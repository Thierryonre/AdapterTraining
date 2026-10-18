"""Offline evaluation: replay held-out episodes through a trained policy and compare actions.

Runs the exact inference path used on a robot (preprocessor -> select_action -> postprocessor, with
SmolVLA's action chunking) on recorded observations, and reports the error against the recorded
actions. This catches pipeline bugs and lets you compare adapters, but it is not a success rate:
for that, run the policy in simulation or on the robot.

    uv run at-eval --checkpoint outputs/train/smoke_lora --dataset local/mimicgen_stack_d1
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.common.control_utils import predict_action
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors

import adapter_training  # noqa: F401  (TLS setup)
from adapter_training.common import dataset_root, default_device, default_video_backend


def resolve_checkpoint(path: Path) -> Path:
    """Accept a run dir (its best.txt checkpoint, else the last), a checkpoint dir, or a pretrained_model dir."""
    best = path / "best.txt"
    chosen = path / "checkpoints" / best.read_text().strip() if best.exists() else path / "checkpoints" / "last"
    for candidate in (path, path / "pretrained_model", chosen / "pretrained_model"):
        if (candidate / "config.json").exists():
            return candidate
    raise SystemExit(f"No pretrained_model/config.json found under {path}")


def held_out_episodes(meta, eval_split: float) -> list[int]:
    """Same rule as lerobot-train's --dataset.eval_split: the last ceil(n * split) episodes per task."""
    by_task: dict[str, list[int]] = {}
    for ep in range(meta.total_episodes):
        for task in meta.episodes[ep]["tasks"]:
            by_task.setdefault(task, []).append(ep)
    held = set()
    for eps in by_task.values():
        held.update(eps[-math.ceil(len(eps) * eval_split) :])
    return sorted(held)


def to_numpy_observation(item: dict) -> dict[str, np.ndarray]:
    """Dataset item (CHW float images) -> robot-style observation (HWC images)."""
    obs = {}
    for key, value in item.items():
        if not key.startswith("observation."):
            continue
        if "image" in key:
            value = value.permute(1, 2, 0)
        obs[key] = value.numpy().astype(np.float32)
    return obs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Training output dir or pretrained_model dir")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--episodes", help="Comma-separated episode indices (default: held-out split)")
    parser.add_argument("--eval-split", type=float, default=0.1, help="Must match the value used for training")
    parser.add_argument("--max-episodes", type=int, default=5)
    parser.add_argument("--max-frames", type=int, help="Only evaluate the first N frames of each episode")
    parser.add_argument("--device", help="cuda / mps / cpu (default: auto)")
    args = parser.parse_args()

    ckpt = resolve_checkpoint(args.checkpoint)
    device = args.device or default_device()
    root = args.dataset_root or dataset_root(args.dataset)

    probe = LeRobotDataset(args.dataset, root=root, episodes=[0], video_backend=default_video_backend())
    if args.episodes:
        episodes = [int(e) for e in args.episodes.split(",")]
    else:
        episodes = held_out_episodes(probe.meta, args.eval_split)
    episodes = episodes[: args.max_episodes]
    dataset = LeRobotDataset(args.dataset, root=root, episodes=episodes, video_backend=default_video_backend())

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = ckpt
    cfg.device = device
    policy = make_policy(cfg, ds_meta=dataset.meta)
    policy.eval()
    extra = {"dataset_meta": dataset.meta, "dataset_stats": dataset.meta.stats} if cfg.type == "groot" else {}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": device}},
        **extra,
    )

    action_names = dataset.meta.features["action"]["names"]
    mean_action = np.asarray(dataset.meta.stats["action"]["mean"], dtype=np.float32)
    print(f"checkpoint={ckpt}\ndevice={device} episodes={episodes}")

    dataset[0]  # load the reader so the index mapping is available
    to_rel = dataset.absolute_to_relative_idx  # episode metadata uses absolute frame indices

    errors, baseline_errors, per_episode = [], [], []
    start = time.time()
    for ep in episodes:
        policy.reset()
        ep_meta = dataset.meta.episodes[ep]
        first, last = ep_meta["dataset_from_index"], ep_meta["dataset_to_index"]
        if args.max_frames:
            last = min(last, first + args.max_frames)
        ep_errors = []
        for idx in range(first, last):
            item = dataset[to_rel[idx] if to_rel else idx]
            action = predict_action(
                to_numpy_observation(item),
                policy,
                torch.device(device),
                preprocessor,
                postprocessor,
                use_amp=False,
                task=item["task"],
                robot_type=dataset.meta.robot_type,
            )
            pred = action.squeeze(0).float().cpu().numpy()
            target = item["action"].numpy()
            ep_errors.append(pred - target)
            baseline_errors.append(mean_action - target)
        ep_errors = np.stack(ep_errors)
        errors.append(ep_errors)
        per_episode.append({"episode": ep, "frames": len(ep_errors), "mse": float((ep_errors**2).mean())})
        print(f"  episode {ep}: {len(ep_errors)} frames, action MSE {per_episode[-1]['mse']:.4f}")

    errors = np.concatenate(errors)
    baseline_errors = np.stack(baseline_errors)
    results = {
        "checkpoint": str(ckpt),
        "episodes": per_episode,
        "frames": len(errors),
        "seconds_per_frame": (time.time() - start) / len(errors),
        "mse": float((errors**2).mean()),
        "baseline_mean_action_mse": float((baseline_errors**2).mean()),
        "mse_per_dim": {n: float(v) for n, v in zip(action_names, (errors**2).mean(0))},
        "mae_per_dim": {n: float(v) for n, v in zip(action_names, np.abs(errors).mean(0))},
    }

    print(f"\nAction MSE {results['mse']:.4f} (always-predict-mean baseline: {results['baseline_mean_action_mse']:.4f})")
    for name in action_names:
        print(f"  {name:>10}: MSE {results['mse_per_dim'][name]:.4f}  MAE {results['mae_per_dim'][name]:.4f}")

    out = ckpt.parent / "offline_eval.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
