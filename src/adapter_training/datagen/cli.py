"""at-datagen: multiply video demos with DexMimicGen-style generation in MuJoCo.

    # 0. Photo of the empty workspace (for realistic images)
    uv run at-datagen background data/calib/empty_workspace.mp4 --out data/calib/background.png

    # 1. Video demos (Part 1) + object scenes (at-skill / perception) -> simulation-validated sources.
    #    The task (phases, references, success check) is inferred from the replay -- no per-skill code.
    uv run at-datagen prepare outputs/video2robot/x/demos.hdf5 --scenes outputs/video2robot/x/scenes \
        --calib data/calib/camera.json --background data/calib/background.png --out outputs/datagen/x/sources.hdf5

    # 2. Generate new demos with objects in new places
    uv run at-datagen generate outputs/datagen/x/sources.hdf5 --n 200 --workers 8 \
        --calib data/calib/camera.json --background data/calib/background.png --out outputs/datagen/x/generated.hdf5

    # 3. Check that saved demos replay successfully from their recorded actions
    uv run at-datagen validate outputs/datagen/x/generated.hdf5

    # 4. -> LeRobot
    uv run at-convert outputs/datagen/x/generated.hdf5 --repo-id local/x_gen \
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
from adapter_training.datagen.appearance import capture_background, load_image, sample_appearance
from adapter_training.datagen.dataset import load_sources, open_output, render_images, write_episode
from adapter_training.datagen.env import SIDES, SceneEnv, SceneObject, SimConfig
from adapter_training.datagen.generate import GenConfig, generate_episode, sample_object_poses
from adapter_training.datagen.source import load_video_demos, replay_with_correction
from adapter_training.datagen.task import (
    check_events_present,
    infer_task,
    make_success_fn,
    phase_boundaries,
    phases_from_spec,
    structure_signature,
)
from adapter_training.render.composite import CameraWarp, Compositor, output_size
from adapter_training.video.camera import Calibration
from adapter_training.video.pipeline import VideoWriter
from adapter_training.video.robot import openarm_config

DEFAULT_CAMERA = {"pos": [0.95, 0.05, 0.55], "xyaxes": [-0.05, 1.0, 0, -0.49, -0.03, 0.87], "fovy": 50}
DEFAULT_TAG_OBJECT = SceneObject()  # the 14 x 5 x 4 cm bar used with object tags


def _json(obj) -> str:
    return json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def image_size(args, calib: Calibration | None) -> tuple[int, int]:
    """'auto' = 320 px wide with the camera's aspect ratio (so training frames match the live camera)."""
    if args.image_size != "auto":
        return tuple(int(v) for v in args.image_size.lower().split("x"))
    width = getattr(args, "image_width", 320)
    return output_size(calib, width) if calib else (width, width * 3 // 4)


def make_env(args, objects: list[SceneObject]) -> SceneEnv:
    cfg = openarm_config()
    if args.base_xyz:
        cfg.base_xyz = tuple(float(v) for v in args.base_xyz.split(","))
    calib = Calibration.load(args.calib) if args.calib else None
    w, h = image_size(args, calib)
    camera = CameraWarp(calib, (w, h)).camera if calib else DEFAULT_CAMERA
    # Without a real background the env's own renderer makes the images; with one, the compositor does.
    sim = SimConfig(image_size=(0, 0) if args.background else (w, h), shadows=args.shadows, objects=objects)
    return SceneEnv(cfg, sim, camera=camera)


def make_compositor(args, env: SceneEnv):
    """(compositor, background image) when --background is given, else (None, None)."""
    if not args.background:
        return None, None
    if not args.calib:
        raise SystemExit("--background needs --calib (the camera that took the photo).")
    calib = Calibration.load(args.calib)
    compositor = Compositor(env.model, calib, image_size(args, calib), env.paste_geoms, shadows=args.shadows)
    return compositor, load_image(args.background)


def scene_from_meta(meta: dict) -> list[SceneObject]:
    return [SceneObject.from_dict(o) for o in json.loads(meta["scene"])]


def save_video(path: Path, frames: np.ndarray, fps: float) -> None:
    writer = VideoWriter(path, fps, (frames.shape[2], frames.shape[1]))
    for f in frames:
        writer.write(f)
    writer.close()


# -- prepare ----------------------------------------------------------------------------------------------
def cmd_prepare(args) -> dict:
    with h5py.File(args.demos, "r") as h5:
        attrs = dict(h5["data"].attrs)
    tag_object = DEFAULT_TAG_OBJECT
    if "object_size" in attrs:
        tag_object = SceneObject(size=tuple(float(v) for v in attrs["object_size"]))
    demos = load_video_demos(args.demos, args.scenes, tag_object)
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    with_objects = [d for d in demos if d.objects]
    for d in demos:
        if not d.objects:
            report[d.name] = {"kept": False, "reason": "no objects for this clip (no scene file or object tag)"}
    if not with_objects:
        (out_dir / "prepare_report.json").write_text(_json(report))
        raise SystemExit("No demo has objects to simulate. See prepare_report.json.")

    # One simulated scene for the skill: the objects of the first clip (same physical objects in every clip).
    scene = with_objects[0].objects
    names = sorted(o.name for o in scene)
    env = make_env(args, scene)
    compositor, background = make_compositor(args, env)
    fps = env.sim.control_hz
    task_text = args.task or attrs.get("task", "")
    out, reference, spec = None, None, None
    for demo in tqdm(with_objects, desc="replaying"):
        if sorted(o.name for o in demo.objects) != names:
            report[demo.name] = {"kept": False, "reason": f"different objects from the first clip ({names})"}
            continue
        problem = check_events_present(demo.openings)
        if problem:
            report[demo.name] = {"kept": False, "reason": problem}
            continue
        ep, diag = replay_with_correction(env, demo)
        if not diag["ok"]:
            diag.update(kept=False, reason="replay failed in simulation (a grasp didn't hold or a release didn't let go)")
            report[demo.name] = diag
            continue
        openings = np.asarray(ep.openings)
        sizes = {n: env.object_size(n) for n in env.object_names}
        poses = {n: np.asarray(P) for n, P in ep.object_pose.items()}
        eef = {s: np.asarray(ep.eef_pose[s]) for s in SIDES}
        final_holders = {n: env.holders(n) for n in names}  # env is at the end of the replay
        demo_spec = infer_task(openings, eef, poses, sizes, holders_at=lambda t, h=final_holders: h)
        if reference is None:
            reference, spec = structure_signature(demo_spec), demo_spec
            out = open_output(args.out, env, task_text, spec, {"demos": str(args.demos)})
        elif structure_signature(demo_spec) != reference:
            diag.update(kept=False, reason="different structure from the first kept clip (grasp/release order or objects)")
            report[demo.name] = diag
            continue
        bounds = phase_boundaries(openings, phases_from_spec(spec), fps)
        images = render_images(env, ep.states, compositor, background)
        name = write_episode(out, ep, env, images, {"source": demo.name, "boundaries": bounds, "success": True})
        diag.update(kept=True, saved_as=name)
        if args.videos:
            save_video(out_dir / f"source_{demo.name}.mp4", images, fps)
        report[demo.name] = diag
    if out is not None:
        out.close()
    kept = sum(r.get("kept", False) for r in report.values())
    (out_dir / "prepare_report.json").write_text(_json(report))
    if spec is not None:
        (out_dir / "task_spec.json").write_text(_json(spec))
    print(_json({k: {kk: v.get(kk) for kk in ("kept", "reason", "saved_as")} for k, v in report.items()}))
    print(f"\n{kept}/{len(demos)} video demos replayed successfully -> {args.out}")
    if spec is not None:
        print("Inferred phases: " + " | ".join(f"{p['name']} {p['refs']}" for p in spec["phases"]))
        print("Success check: " + json.dumps(spec["success"]))
    return {"kept": kept, "total": len(demos), "spec": spec, "report": report}


# -- generate ---------------------------------------------------------------------------------------------
def _failure_reason(info: dict, phases) -> str:
    for p in phases:
        held = info.get(f"{p.name}_held")
        if held is None or not p.end_event:
            continue
        side, kind = p.end_event.split("_")
        ref = p.refs.get(side)
        if kind == "close" and ref is not None and side not in held.get(ref, []):
            return f"{p.name}: {side} gripper didn't hold {ref}"
    return "final check"


def _generate_worker(args, worker: int, n_target: int, max_trials: int, out_path: Path) -> dict:
    """Generate until n_target successes (or max_trials) into out_path. Runs in its own process."""
    sources, meta = load_sources(args.sources)
    spec = json.loads(meta["task_spec"])
    phases = phases_from_spec(spec)
    env = make_env(args, scene_from_meta(meta))
    env.success_fn = make_success_fn(spec)
    compositor, background = make_compositor(args, env)
    base_colours = {o.name: (None if o.texture else o.rgba) for o in env.sim.objects}
    fps = env.sim.control_hz
    cfg = GenConfig(
        dx=(-args.dx, args.dx), dy=(-args.dy, args.dy), dyaw_deg=(-args.dyaw, args.dyaw), action_noise_m=args.noise
    )
    rng = np.random.default_rng([args.seed, worker])
    out = open_output(out_path, env, str(meta["task"]), spec, {"sources": str(args.sources)})
    successes, trials, failures = 0, 0, {}
    bar = tqdm(total=n_target, desc=f"worker {worker}", position=worker, leave=False)
    while successes < n_target and trials < max_trials:
        trials += 1
        src = sources[rng.integers(len(sources))]
        poses = sample_object_poses(src, env, cfg, rng)
        ep, ok, info = generate_episode(env, src, phases, poses, cfg, rng)
        if not ok:
            reason = _failure_reason(info, phases)
            failures[reason] = failures.get(reason, 0) + 1
            continue
        appearance = sample_appearance(rng, base_colours, args.randomize) if compositor and args.randomize else None
        images = render_images(env, ep.states, compositor, background, appearance)
        attrs = {
            "source": src.name,
            "boundaries": info["boundaries"],
            "success": True,
            "object_xyz": {n: T[:3, 3].tolist() for n, T in poses.items()},
        }
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
                if n == 0 or not data.attrs:
                    for k, v in src["data"].attrs.items():
                        data.attrs[k] = v
                for name in sorted(src["data"], key=lambda d: int(d.split("_")[1])):
                    src.copy(src["data"][name], data, name=f"demo_{n}")
                    n += 1
    return n


def cmd_generate(args) -> dict:
    import multiprocessing
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
        # "spawn", not Linux's default "fork": this process has usually used the GPU already (SAM 3, EGL
        # rendering during replay), and forked children inheriting that state deadlock in the NVIDIA driver.
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
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
    return summary


# -- background -------------------------------------------------------------------------------------------
def cmd_background(args) -> None:
    import cv2

    img = capture_background(args.video, args.max_frames, args.stride)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved {args.out} ({img.shape[1]}x{img.shape[0]}). Check nobody/nothing moving is left in it.")


# -- validate ---------------------------------------------------------------------------------------------
def cmd_validate(args) -> tuple[int, int]:
    with h5py.File(args.dataset, "r") as h5:
        meta = dict(h5["data"].attrs)
        env = make_env(args, scene_from_meta(meta))
        env.success_fn = make_success_fn(json.loads(meta["task_spec"]))
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
    return ok, len(results)


def add_common(p) -> None:
    p.add_argument("--calib", type=Path, help="Camera calibration: render images from the real camera's viewpoint")
    p.add_argument("--image-size", default="auto", help="WxH, or auto = --image-width wide with the camera's aspect")
    p.add_argument("--image-width", type=int, default=320, help="Width for --image-size auto (SmolVLA works at 512)")
    p.add_argument("--base-xyz", help="Robot base in the table frame (must match Part 1)")
    p.add_argument("--shadows", action="store_true", help="Render shadows (slow without a real GPU)")
    p.add_argument("--background", type=Path, help="Photo of the empty workspace (at-datagen background)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="Video demos -> simulation-validated source demos (+ inferred task)")
    p.add_argument("demos", type=Path, help="demos.hdf5 from at-video2robot")
    p.add_argument("--scenes", type=Path, help="Folder of <video stem>.json scenes from object perception")
    p.add_argument("--out", type=Path, default=Path("outputs/datagen/sources.hdf5"))
    p.add_argument("--task", help="Language instruction (default: the one stored with the demos)")
    p.add_argument("--videos", action="store_true", help="Save a video of every kept replay")
    add_common(p)
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
    p.add_argument("--randomize", type=float, default=1.0, help="Appearance randomisation strength (0 = off)")
    add_common(p)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("background", help="Median frame of an empty-workspace clip -> background photo")
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, default=Path("data/calib/background.png"))
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--stride", type=int, default=3)
    p.set_defaults(func=cmd_background)

    p = sub.add_parser("validate", help="Replay saved demos from their actions and check success")
    p.add_argument("dataset", type=Path)
    add_common(p)
    p.set_defaults(func=cmd_validate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
