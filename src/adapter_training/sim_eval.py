"""at-sim-eval: closed-loop success rate of trained policies in the simulated scene.

    uv run at-sim-eval --checkpoints outputs/train/pick_up_water_bottle/v2 outputs/train/pick_up_water_bottle/v1 \
        --sources outputs/skills/pick_up_water_bottle/datagen/sources.hdf5 \
        --calib data/setup/camera.json --background data/setup/background.png --episodes 50

Each episode puts the objects at new random positions (the same for every checkpoint, so the
comparison is fair), renders the camera image exactly as for training (robot + objects composited on
the real background), runs the policy, steps the physics, and applies the skill's inferred success
check. Placement is a bit wider than during generation and uses different random seeds.

A training dir (vN) means its chosen checkpoint (best.txt, written when checkpoints were compared,
else the last one). With --all-checkpoints, every checkpoint saved in each training dir is evaluated
instead (training with --save-freq) and the best is reported:

    uv run at-sim-eval --all-checkpoints --checkpoints outputs/train/stack_sponges/v1 \
        --sources outputs/skills/stack_sponges/datagen/sources.hdf5 --calib data/setup/camera.json \
        --background data/setup/background.png --episodes 100

Simulation success is a filter and a way to rank adapters, not a prediction of real-robot success:
the policy was trained on data from this simulator, so this is optimistic.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import adapter_training  # noqa: F401  (TLS setup)
from adapter_training.common import default_device
from adapter_training.datagen.appearance import load_image
from adapter_training.datagen.cli import make_compositor, make_env, scene_from_meta
from adapter_training.datagen.dataset import load_sources
from adapter_training.datagen.env import SIDES
from adapter_training.datagen.generate import GenConfig, sample_object_poses
from adapter_training.datagen.task import make_success_fn
from adapter_training.video.pipeline import VideoWriter


def resolve_checkpoint(path: Path) -> Path:
    """A pretrained_model dir, a checkpoint dir, or a training dir (its best.txt checkpoint, else the last)."""
    best = path / "best.txt"
    chosen = path / "checkpoints" / best.read_text().strip() if best.exists() else path / "checkpoints" / "last"
    for candidate in (path, path / "pretrained_model", chosen / "pretrained_model"):
        if (candidate / "config.json").exists():
            return candidate
    raise SystemExit(f"No pretrained_model/config.json under {path}")


def list_checkpoints(run: Path) -> list[Path]:
    """Every checkpoint saved in a training dir (checkpoints/005000, 010000, ...), oldest first."""
    steps = sorted(p for p in (run / "checkpoints").glob("*")
                   if p.name.isdigit() and (p / "pretrained_model" / "config.json").exists())
    return steps or [run]


def pick_best(checkpoints: list[Path], results: dict) -> Path:
    """Highest simulated success; the earliest checkpoint wins a tie."""
    return max(checkpoints, key=lambda c: (results["checkpoints"][str(c)]["success_rate"], -checkpoints.index(c)))


def label(path: Path) -> str:
    """Short unique name for files: v2 or v2_010000."""
    return f"{path.parent.parent.name}_{path.name}" if path.parent.name == "checkpoints" else path.name


def load_policy(ckpt: Path, device: str):
    from lerobot.common.control_utils import predict_action  # noqa: F401  (import check)
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.policies import make_policy, make_pre_post_processors

    train_cfg = json.loads((ckpt / "train_config.json").read_text())
    meta = LeRobotDatasetMetadata(train_cfg["dataset"]["repo_id"], root=train_cfg["dataset"]["root"])
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = ckpt
    cfg.device = device
    policy = make_policy(cfg, ds_meta=meta)
    policy.eval()
    extra = {"dataset_meta": meta, "dataset_stats": meta.stats} if cfg.type == "groot" else {}  # GR00T needs them
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=str(ckpt), preprocessor_overrides={"device_processor": {"device": device}}, **extra
    )
    image_key = next(k for k in cfg.input_features if "image" in k)
    _, h, w = cfg.input_features[image_key].shape
    return policy, pre, post, image_key, (w, h), meta.robot_type


def episode_layouts(sources, env, n: int, seed: int, widen: float) -> list[dict]:
    """Object poses + source initial state per episode, fixed so every checkpoint sees the same layouts."""
    base = GenConfig()
    cfg = GenConfig(
        dx=tuple(widen * v for v in base.dx), dy=tuple(widen * v for v in base.dy), dyaw_deg=tuple(widen * v for v in base.dyaw_deg)
    )
    rng = np.random.default_rng([seed, 991])  # different stream from generation's seeds
    layouts = []
    for _ in range(n):
        src = sources[rng.integers(len(sources))]
        layouts.append({"state": src.initial_state, "poses": sample_object_poses(src, env, cfg, rng)})
    return layouts


def run_episode(env, compositor, background, policy, pre, post, image_key, task, robot_type, layout, max_steps,
                device, hold_steps: int = 5, video: list | None = None) -> dict:
    from lerobot.common.control_utils import predict_action

    env.set_state(layout["state"])
    for name, T in layout["poses"].items():
        env.set_object_pose(T, name)
    for _ in range(5):
        env._physics_step()
    policy.reset()
    streak = 0
    for t in range(max_steps):
        image = compositor.compose(background, env.data)
        if video is not None:
            video.append(image)
        obs = {image_key: image, "observation.state": env.observation()["joint_pos"].astype(np.float32)}
        action = predict_action(obs, policy, torch.device(device), pre, post, use_amp=False, task=task, robot_type=robot_type)
        a = action.squeeze(0).float().cpu().numpy()
        arm, openings = {}, {}
        for k, s in enumerate(SIDES):
            closed, opened = env.cfg.arms[s].gripper_motor_range
            arm[s] = np.radians(a[k * 8 : k * 8 + 7])
            openings[s] = float((a[k * 8 + 7] - closed) / (opened - closed))
        env.step_joints(arm, openings)
        streak = streak + 1 if env.is_success() else 0
        if streak >= hold_steps:  # success has to hold for a moment, not flicker
            return {"success": True, "steps": t + 1}
    return {"success": False, "steps": max_steps}


def evaluate(checkpoints: list[Path], sources_path: Path, calib: Path, background: Path | None, episodes: int = 50,
             seed: int = 0, widen: float = 1.25, max_steps: int | None = None, videos: int = 2, out: Path | None = None,
             device: str | None = None, base_xyz: str | None = None) -> dict:
    device = device or default_device()
    sources, meta = load_sources(sources_path)
    spec = json.loads(meta["task_spec"])
    task = str(meta["task"])
    longest = max(s.target_pose["left"].shape[0] for s in sources)
    max_steps = max_steps or int(longest * 1.5)
    results = {"episodes": episodes, "max_steps": max_steps, "task": task, "checkpoints": {}}
    layouts = None
    for ckpt_path in checkpoints:
        ckpt = resolve_checkpoint(ckpt_path)
        policy, pre, post, image_key, size, robot_type = load_policy(ckpt, device)
        args = argparse.Namespace(base_xyz=base_xyz, image_size=f"{size[0]}x{size[1]}", calib=str(calib),
                                  background=str(background) if background else None, shadows=False)
        env = make_env(args, scene_from_meta(meta))
        env.success_fn = make_success_fn(spec)
        compositor, bg = make_compositor(args, env)
        if compositor is None:  # no real background: composite onto a plain grey image of the right size
            from adapter_training.render.composite import Compositor
            from adapter_training.video.camera import Calibration

            compositor = Compositor(env.model, Calibration.load(calib), size, env.paste_geoms)
            bg = np.full((size[1], size[0], 3), 110, np.uint8)
        layouts = layouts or episode_layouts(sources, env, episodes, seed, widen)
        per_episode, start = [], time.time()
        for i, layout in enumerate(layouts):
            frames = [] if (out and i < videos) else None
            r = run_episode(env, compositor, bg, policy, pre, post, image_key, task, robot_type, layout, max_steps, device, video=frames)
            per_episode.append(r)
            if frames:
                out.mkdir(parents=True, exist_ok=True)
                writer = VideoWriter(out / f"{label(ckpt_path)}_episode{i}_{'success' if r['success'] else 'fail'}.mp4",
                                     env.sim.control_hz, (frames[0].shape[1], frames[0].shape[0]))
                for f in frames:
                    writer.write(f)
                writer.close()
            print(f"  {ckpt_path}: episode {i + 1}/{episodes} {'success' if r['success'] else 'fail'}")
        successes = sum(r["success"] for r in per_episode)
        results["checkpoints"][str(ckpt_path)] = {
            "success_rate": successes / episodes,
            "successes": successes,
            "mean_steps_to_success": float(np.mean([r["steps"] for r in per_episode if r["success"]])) if successes else None,
            "seconds": round(time.time() - start, 1),
        }
        del policy
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "sim_eval.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", type=Path, required=True, help="Training dirs or pretrained_model dirs")
    p.add_argument("--all-checkpoints", action="store_true",
                   help="Evaluate every checkpoint saved in each training dir (not just its chosen one)")
    p.add_argument("--sources", type=Path, required=True, help="sources.hdf5 of the skill (scene + task)")
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--background", type=Path)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--widen", type=float, default=1.25, help="Placement range relative to generation")
    p.add_argument("--max-steps", type=int, help="Default: 1.5x the longest source demo")
    p.add_argument("--videos", type=int, default=2, help="Episode videos to save per checkpoint")
    p.add_argument("--out", type=Path, help="Folder for sim_eval.json and videos")
    p.add_argument("--device")
    p.add_argument("--base-xyz")
    a = p.parse_args()
    checkpoints = [c for run in a.checkpoints for c in list_checkpoints(run)] if a.all_checkpoints else a.checkpoints
    res = evaluate(checkpoints, a.sources, a.calib, a.background, a.episodes, a.seed, a.widen, a.max_steps,
                   a.videos, a.out, a.device, a.base_xyz)
    print(json.dumps({k: v["success_rate"] for k, v in res["checkpoints"].items()}, indent=2))
    if a.all_checkpoints:
        for run in a.checkpoints:
            ckpts = list_checkpoints(run)
            best = pick_best(ckpts, res)
            print(f"best in {run}: {best} ({res['checkpoints'][str(best)]['success_rate']:.0%})")


if __name__ == "__main__":
    main()
