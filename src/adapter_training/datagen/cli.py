"""at-datagen: Part 2 -- multiply video demos with DexMimicGen-style generation in MuJoCo.

    # 1. Video demos (Part 1, with --object-tag-id) -> simulation-validated source demos
    uv run at-datagen prepare outputs/video2robot/handover/demos.hdf5 --calib data/calib/camera.json \
        --out outputs/datagen/handover/sources.hdf5

    # 2. Generate new demos with the object in new places
    uv run at-datagen generate outputs/datagen/handover/sources.hdf5 --n 200 --calib data/calib/camera.json \
        --out outputs/datagen/handover/generated.hdf5

    # 3. Check that saved demos replay successfully from their recorded actions
    uv run at-datagen validate outputs/datagen/handover/generated.hdf5

    # 4. -> LeRobot (same as Part 1)
    uv run at-convert outputs/datagen/handover/generated.hdf5 --repo-id local/handover_gen \
        --cameras front_image:front --state-keys joint_pos --robot-type bi_openarm_follower
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

import adapter_training  # noqa: F401  (TLS setup)


def _json(obj) -> str:
    return json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

from adapter_training.datagen.dataset import load_sources, open_output, render_images, write_episode
from adapter_training.datagen.env import SIDES, HandoverEnv, ObjectSpec, SimConfig
from adapter_training.datagen.generate import GenConfig, generate_episode, sample_object_pose
from adapter_training.datagen.source import load_video_demos, replay_with_correction
from adapter_training.datagen.task import HANDOVER_PHASES, HANDOVER_TASK, check_handover_order, phase_boundaries
from adapter_training.video.camera import Calibration
from adapter_training.video.pipeline import VideoWriter
from adapter_training.video.robot import openarm_config


def make_env(args, object_size) -> HandoverEnv:
    cfg = openarm_config()
    if args.base_xyz:
        cfg.base_xyz = tuple(float(v) for v in args.base_xyz.split(","))
    w, h = (int(v) for v in args.image_size.lower().split("x"))
    camera = None
    if args.calib:
        camera = Calibration.load(args.calib).scaled(w, h).mujoco_camera()
    else:  # default view from across the table
        camera = {"pos": [0.95, 0.05, 0.55], "xyaxes": [-0.05, 1.0, 0, -0.49, -0.03, 0.87], "fovy": 50}
    sim = SimConfig(image_size=(w, h), shadows=args.shadows, obj=ObjectSpec(size=tuple(object_size)))
    return HandoverEnv(cfg, sim, camera=camera)


def save_video(path: Path, frames: np.ndarray, fps: float) -> None:
    writer = VideoWriter(path, fps, (frames.shape[2], frames.shape[1]))
    for f in frames:
        writer.write(f)
    writer.close()


# -- prepare ----------------------------------------------------------------------------------------------
def cmd_prepare(args) -> None:
    with h5py.File(args.demos, "r") as h5:
        attrs = dict(h5["data"].attrs)
    object_size = args.object_size or tuple(attrs.get("object_size", (0.14, 0.05, 0.04)))
    demos, _ = load_video_demos(args.demos, object_size[2])
    overrides = json.loads(Path(args.object_poses).read_text()) if args.object_poses else {}
    env = make_env(args, object_size)
    fps = env.sim.control_hz
    out = open_output(args.out, env, args.task or attrs.get("task", HANDOVER_TASK), [p.name for p in HANDOVER_PHASES])
    report = {}
    for demo in tqdm(demos, desc="replaying"):
        obj = demo.object_pose
        if demo.name in overrides:  # [x, y, yaw_deg] measured by hand
            x, y, yaw = overrides[demo.name]
            c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
            obj = np.eye(4)
            obj[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            obj[:3, 3] = [x, y, object_size[2] / 2]
        if obj is None:
            report[demo.name] = {"kept": False, "reason": "object tag never seen (pass --object-poses)"}
            continue
        problem = check_handover_order(demo.openings)
        if problem:
            report[demo.name] = {"kept": False, "reason": problem}
            continue
        ep, diag = replay_with_correction(env, demo, obj)
        commanded = np.asarray(ep.openings)
        try:
            bounds = phase_boundaries(commanded, HANDOVER_PHASES, fps)
        except ValueError as e:
            diag.update(kept=False, reason=str(e))
            report[demo.name] = diag
            continue
        kept = diag["success"] and all(diag["grasped"].get(s, False) for s in SIDES)
        diag.update(kept=kept, object_xyz=obj[:3, 3].round(4).tolist())
        if not kept:
            diag["reason"] = "replay failed in simulation"
        else:
            images = render_images(env, ep.states)
            name = write_episode(out, ep, env, images, {"source": demo.name, "boundaries": bounds, "success": True})
            diag["saved_as"] = name
            if args.videos:
                save_video(args.out.parent / f"source_{demo.name}.mp4", images, fps)
        report[demo.name] = diag
    kept = sum(r.get("kept", False) for r in report.values())
    out.close()
    (args.out.parent / "prepare_report.json").write_text(_json(report))
    print(_json(report))
    print(f"\n{kept}/{len(demos)} video demos replayed successfully -> {args.out}")


# -- generate ---------------------------------------------------------------------------------------------
def _generate_worker(args, worker: int, n_target: int, max_trials: int, out_path: Path) -> dict:
    """Generate until n_target successes (or max_trials) into out_path. Runs in its own process."""
    sources, meta = load_sources(args.sources)
    env = make_env(args, tuple(meta["object_size"]))
    fps = env.sim.control_hz
    cfg = GenConfig(
        dx=(-args.dx, args.dx), dy=(-args.dy, args.dy), dyaw_deg=(-args.dyaw, args.dyaw), action_noise_m=args.noise
    )
    rng = np.random.default_rng([args.seed, worker])
    out = open_output(out_path, env, str(meta["task"]), json.loads(meta["phases"]), {"sources": str(args.sources)})
    successes, trials, failures = 0, 0, {}
    bar = tqdm(total=n_target, desc=f"worker {worker}", position=worker, leave=False)
    while successes < n_target and trials < max_trials:
        trials += 1
        src = sources[rng.integers(len(sources))]
        obj = sample_object_pose(src.object_pose[0], cfg, rng, env.sim.obj.size[2])
        ep, ok, info = generate_episode(env, src, HANDOVER_PHASES, obj, cfg, rng)
        if not ok:
            reason = next((k for k, v in info.items() if k.endswith("_grasped") and not any(v.values())), "final check")
            failures[reason] = failures.get(reason, 0) + 1
            continue
        images = render_images(env, ep.states)
        attrs = {"source": src.name, "boundaries": info["boundaries"], "success": True, "object_xyz": obj[:3, 3].tolist()}
        write_episode(out, ep, env, images, attrs)
        if worker == 0 and successes < args.videos:
            save_video(args.out.parent / f"generated_{successes}.mp4", images, fps)
        successes += 1
        bar.update(1)
    bar.close()
    out.close()
    return {"generated": successes, "trials": trials, "failures": failures}


def merge_hdf5(parts: list[Path], out: Path) -> int:
    """Concatenate demos from worker files into one file, renumbering demo_0..N."""
    n = 0
    with h5py.File(out, "w") as dst:
        data = dst.create_group("data")
        for part in parts:
            with h5py.File(part, "r") as src:
                if n == 0:
                    for k, v in src["data"].attrs.items():
                        data.attrs[k] = v
                for name in sorted(src["data"], key=lambda d: int(d.split("_")[1])):
                    src.copy(src["data"][name], data, name=f"demo_{n}")
                    n += 1
    return n


def cmd_generate(args) -> None:
    from concurrent.futures import ProcessPoolExecutor

    sources, _ = load_sources(args.sources)
    if not sources:
        raise SystemExit("No source demos.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    workers = max(1, args.workers)
    targets = [args.n // workers + (i < args.n % workers) for i in range(workers)]
    trials = [args.max_trials // workers + (i < args.max_trials % workers) for i in range(workers)]
    if workers == 1:
        results = [_generate_worker(args, 0, args.n, args.max_trials, args.out)]
    else:
        parts = [args.out.with_name(f"{args.out.stem}.part{i}.hdf5") for i in range(workers)]
        with ProcessPoolExecutor(workers) as pool:
            futures = [pool.submit(_generate_worker, args, i, targets[i], trials[i], parts[i]) for i in range(workers)]
            results = [f.result() for f in futures]
        merge_hdf5(parts, args.out)
        for p in parts:
            p.unlink()
    failures: dict = {}
    for r in results:
        for k, v in r["failures"].items():
            failures[k] = failures.get(k, 0) + v
    generated, tried = sum(r["generated"] for r in results), sum(r["trials"] for r in results)
    summary = {
        "generated": generated,
        "trials": tried,
        "success_rate": generated / max(tried, 1),
        "failures": failures,
        "workers": workers,
        "seconds": round(time.time() - start, 1),
    }
    (args.out.parent / "generate_report.json").write_text(_json(summary))
    print(_json(summary))


# -- validate ---------------------------------------------------------------------------------------------
def cmd_validate(args) -> None:
    with h5py.File(args.dataset, "r") as h5:
        meta = dict(h5["data"].attrs)
        env = make_env(args, tuple(meta["object_size"]))
        results = []
        for name in tqdm(sorted(h5["data"], key=lambda n: int(n.split("_")[1])), desc="replaying actions"):
            demo = h5["data"][name]
            env.set_state(demo["states"][0])
            for a in demo["actions"][:]:
                arm, openings = {}, {}
                for k, s in enumerate(SIDES):
                    closed, opened = env.cfg.arms[s].gripper_motor_range
                    arm[s] = np.radians(a[k * 8 : k * 8 + 7])
                    openings[s] = (a[k * 8 + 7] - closed) / (opened - closed)
                env.step_joints(arm, openings)
            results.append((name, env.is_success()))
    ok = sum(r for _, r in results)
    for name, r in results:
        if not r:
            print(f"  {name}: FAILED on action replay")
    print(f"{ok}/{len(results)} demos succeed when replayed from their recorded actions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--calib", type=Path, help="Camera calibration: render images from the real camera's viewpoint")
        p.add_argument("--image-size", default="320x240")
        p.add_argument("--base-xyz", help="Robot base in the table frame (must match Part 1)")
        p.add_argument("--shadows", action="store_true", help="Render shadows (slow without a real GPU)")

    p = sub.add_parser("prepare", help="Video demos -> simulation-validated source demos")
    p.add_argument("demos", type=Path, help="demos.hdf5 from at-video2robot")
    p.add_argument("--out", type=Path, default=Path("outputs/datagen/sources.hdf5"))
    p.add_argument("--object-size", type=lambda s: tuple(float(v) for v in s.split(",")))
    p.add_argument("--object-poses", help='JSON {"demo_0": [x, y, yaw_deg], ...} overriding the tag')
    p.add_argument("--task")
    p.add_argument("--videos", action="store_true", help="Save a video of every kept replay")
    common(p)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("generate", help="Source demos -> many new demos")
    p.add_argument("sources", type=Path)
    p.add_argument("--out", type=Path, default=Path("outputs/datagen/generated.hdf5"))
    p.add_argument("--n", type=int, default=100, help="Successful demos to generate")
    p.add_argument("--max-trials", type=int, default=1000)
    p.add_argument("--dx", type=float, default=0.04, help="Object x randomisation (+/- m)")
    p.add_argument("--dy", type=float, default=0.05, help="Object y randomisation (+/- m)")
    p.add_argument("--dyaw", type=float, default=20.0, help="Object yaw randomisation (+/- deg)")
    p.add_argument("--noise", type=float, default=0.0, help="Gripper position noise (m) on commands")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--videos", type=int, default=3, help="Save videos of the first N generated demos")
    p.add_argument("--workers", type=int, default=1, help="Parallel simulation processes (use ~CPU cores)")
    common(p)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("validate", help="Replay saved demos from their actions and check success")
    p.add_argument("dataset", type=Path)
    common(p)
    p.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
