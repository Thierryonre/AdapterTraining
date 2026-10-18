"""at-run: one command from clips to a trained, evaluated skill adapter.

    uv run at-run --setup data/setup --name "Pick up water bottle" \
        --description "Pick up the water bottle with the right hand and lift it." --videos clips/*.mp4 --train

    uv run at-run --job job.json          # the same, from a JSON file (what an app would write)

Stages: camera calibration -> background photo -> at-skill (hands -> robot trajectories, objects ->
3D models, simulation replay + task inference, generation, validation, LeRobot dataset, LoRA training)
-> offline evaluation. Setup results are cached in the setup folder and redone when its clips change.

Setup folder (record these once per camera placement):
  tag.mp4           a few seconds with the table AprilTag visible, nobody in view       (required)
  empty.mp4         a few seconds of the empty workspace, tag removed if you like         (optional:
                    the background for generated images; tag.mp4 is used if missing)
  checkerboard.mp4  a checkerboard moved around the view (better intrinsics)              (optional)
  setup.json        {"tag_size": 0.16, "tag_id": 0, "hfov": 70, "pattern": "9x6", "square": 0.025,
                     "base_xyz": "-0.30,0,-0.45"}                                          (optional)
                    tag too small/blurred to decode: add "robot_side": "top" (image edge nearest the
                    robot) to use its black square instead; or "tag_corners_px": [[x,y] x 4] by hand

Job file: {"name": ..., "description": ..., "videos": [...], "setup": ..., "train": true, ...}; any
at-skill option can be given with underscores (e.g. "image_width": 512, "segmenter": "sam3").
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import adapter_training  # noqa: F401  (TLS setup)
from adapter_training import skill
from adapter_training.common import OUTPUTS_DIR
from adapter_training.perception.perceive import slug

SETUP_DEFAULTS = {"tag_size": 0.16, "tag_id": 0, "tag_family": "36h11", "hfov": 70.0, "pattern": "9x6", "square": 0.025}


def _stale(output: Path, *sources: Path) -> bool:
    """True if output is missing or older than any existing source (or the setup.json next to it)."""
    if not output.exists():
        return True
    newest = max((s.stat().st_mtime for s in sources if s and s.exists()), default=0)
    return newest > output.stat().st_mtime


def prepare_setup(setup: Path) -> dict:
    """Calibrate the camera and make the background photo if needed. Returns paths and settings."""
    from types import SimpleNamespace

    import cv2

    from adapter_training.datagen.appearance import capture_background
    from adapter_training.video import calibrate

    cfg_file = setup / "setup.json"
    cfg = {**SETUP_DEFAULTS, **(json.loads(cfg_file.read_text()) if cfg_file.exists() else {})}
    tag_clip, empty_clip, board_clip = setup / "tag.mp4", setup / "empty.mp4", setup / "checkerboard.mp4"
    if not tag_clip.exists():
        raise SystemExit(f"{tag_clip} is missing: record a few seconds with the table AprilTag visible.")

    intrinsics = setup / "intrinsics.json"
    if board_clip.exists() and _stale(intrinsics, board_clip, cfg_file):
        print("[setup] camera intrinsics from checkerboard.mp4")
        calibrate.cmd_intrinsics(SimpleNamespace(video=board_clip, pattern=cfg["pattern"], square=cfg["square"],
                                                 stride=10, out=intrinsics))
    camera = setup / "camera.json"
    if _stale(camera, tag_clip, intrinsics if board_clip.exists() else None, cfg_file):
        print("[setup] camera pose from tag.mp4")
        calibrate.cmd_table(SimpleNamespace(
            video=tag_clip, tag_size=cfg["tag_size"], tag_family=cfg["tag_family"], tag_id=cfg["tag_id"],
            intrinsics=intrinsics if board_clip.exists() else None, hfov=cfg["hfov"], stride=1, max_frames=60, out=camera,
            robot_side=cfg.get("robot_side"), corners=cfg.get("tag_corners_px"),
        ))
    background = setup / "background.png"
    bg_source = empty_clip if empty_clip.exists() else tag_clip
    if _stale(background, bg_source):
        print(f"[setup] background photo from {bg_source.name}")
        cv2.imwrite(str(background), cv2.cvtColor(capture_background(bg_source), cv2.COLOR_RGB2BGR))
    return {"camera": camera, "background": background, "base_xyz": cfg.get("base_xyz"),
            "intrinsics": "checkerboard" if board_clip.exists() else f"approximate (hfov {cfg['hfov']} deg)"}


def skill_argv(job: dict, setup: dict) -> list[str]:
    """at-skill arguments from the job dict (any at-skill option, underscores for dashes)."""
    argv = ["--name", job["name"], "--description", job.get("description", ""),
            "--videos", *map(str, job["videos"]), "--calib", str(setup["camera"]),
            "--background", str(setup["background"])]
    if setup.get("base_xyz") and "base_xyz" not in job:
        argv += [f"--base-xyz={setup['base_xyz']}"]  # "=": values like -0.30,0,-0.45 start with "-"
    skip = {"name", "description", "videos", "setup", "calib", "background", "evaluate"}
    for key, value in job.items():
        if key in skip or value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        argv += [flag] if value is True else [f"{flag}={value}"]
    return argv


def evaluate(adapter: Path, name_slug: str) -> dict | None:
    """Offline evaluation of the trained adapter on held-out episodes (action error vs a baseline)."""
    if not (adapter / "config.json").exists():
        return None
    subprocess.run([sys.executable, "-m", "adapter_training.evaluate", "--checkpoint", str(adapter),
                    "--dataset", f"local/{name_slug}", "--max-episodes", "3"], check=True)
    result = json.loads((adapter.parent / "offline_eval.json").read_text())
    return {"action_mse": result["mse"], "mean_action_baseline_mse": result["baseline_mean_action_mse"],
            "frames": result["frames"]}


def main() -> None:
    # allow_abbrev=False: unknown flags (e.g. --n) must reach at-skill, not match --name / --no-evaluate.
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    p.add_argument("--job", type=Path, help="Job JSON file (alternative to the options below)")
    p.add_argument("--setup", type=Path, default=Path("data/setup"), help="Setup folder (see above)")
    p.add_argument("--name")
    p.add_argument("--description", default="")
    p.add_argument("--videos", nargs="+", type=Path)
    p.add_argument("--train", action="store_true")
    p.add_argument("--no-evaluate", action="store_true", help="Skip offline evaluation after training")
    args, extra = p.parse_known_args()  # anything else goes straight to at-skill

    if args.job:
        job = json.loads(args.job.read_text())
    else:
        if not args.name or not args.videos:
            p.error("give --job, or --name and --videos")
        job = {"name": args.name, "description": args.description, "videos": args.videos, "train": args.train,
               "setup": str(args.setup)}
    setup_dir = Path(job.get("setup", args.setup))

    setup = prepare_setup(setup_dir)
    # Extra command-line options go straight to at-skill (e.g. --tracker mediapipe for a quick CPU run).
    s_args = skill.build_parser().parse_args([*skill_argv(job, setup), *extra])
    code = skill.run(s_args)

    out = s_args.out or OUTPUTS_DIR / "skills" / slug(s_args.name)
    report_path = out / "report.json"
    report = json.loads(report_path.read_text())
    report["setup"] = {k: str(v) for k, v in setup.items()}
    adapter = report["stages"].get("training", {}).get("adapter")
    if adapter and not (args.no_evaluate or job.get("evaluate") is False):
        report["stages"]["offline_evaluation"] = {"status": "done", **(evaluate(Path(adapter), slug(s_args.name)) or {})}
    report_path.write_text(json.dumps(report, indent=2, default=str))

    result = report.get("result", {})
    print("\n" + "=" * 70)
    print(f"Skill: {s_args.name}  ->  {result.get('status')}")
    for stage, info in report["stages"].items():
        print(f"  {stage:18s} {info.get('status', ''):7s} {info.get('reason', '')}")
    if result.get("status") == "done":
        print(f"  dataset: {result.get('dataset')}")
        if "training" in report["stages"]:
            print(f"  adapter: {report['stages']['training']['adapter']} (promoted: {result.get('promoted')})")
        if "sim_evaluation" in report["stages"]:
            se = report["stages"]["sim_evaluation"]
            prev = se.get("previous_success_rate")
            print(f"  simulated success: {se['new_success_rate']:.0%}" + (f" (previous {prev:.0%})" if prev is not None else ""))
    print(f"  report:  {report_path}")
    sys.exit(code)


if __name__ == "__main__":
    main()
