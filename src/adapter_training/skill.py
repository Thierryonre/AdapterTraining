"""at-skill: clip(s) + skill name + description -> training data (and optionally a LoRA adapter).

No per-skill code: objects are found from the description, reconstructed in 3D, placed on the table;
the task (phases, which object each arm works relative to, success check) is inferred from the
replayed demonstration. Every stage has a quality gate; the job stops with a reason instead of
training on bad data.

    uv run at-skill --name "Pick up water bottle" --description "Pick up the water bottle and lift it." \
        --videos clips/*.mp4 --calib data/calib/camera.json --background data/calib/background.png \
        --tracker wilor --segmenter sam3 --reconstructor sam3d --n 200 --workers 8 --train

Writes <out>/report.json (stage-by-stage status, QA numbers, reasons) -- what an app should show.
Exit code 0 = dataset (and adapter) produced, 1 = stopped by a quality gate, 2 = error.

With --train, adapters are versioned (outputs/train/<skill>/v1, v2, ...). After training, the new adapter
and the previously promoted one run closed-loop in simulation on the same object layouts; the new one is
promoted (latest.txt) only if it reaches --min-sim-success and isn't more than --max-regression worse.
With --save-freq, every saved checkpoint of the new version is evaluated and the best one is used
(written to vN/best.txt).
"""

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import adapter_training  # noqa: F401  (TLS setup)
from adapter_training.common import OUTPUTS_DIR, dataset_root
from adapter_training.perception.objects import describe_objects
from adapter_training.perception.perceive import PerceptionConfig, perceive_clip, read_frame, slug
from adapter_training.video.camera import Calibration


class GateFailed(Exception):
    pass


class Job:
    def __init__(self, out: Path):
        self.out = out
        self.report = {"stages": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        out.mkdir(parents=True, exist_ok=True)

    def stage(self, name: str, **info) -> None:
        self.report["stages"][name] = {**self.report["stages"].get(name, {}), **info}
        self.save()

    def fail(self, stage: str, reason: str, **info) -> None:
        self.stage(stage, status="failed", reason=reason, **info)
        self.report["result"] = {"status": "stopped", "stage": stage, "reason": reason}
        self.save()
        raise GateFailed(f"{stage}: {reason}")

    def save(self) -> None:
        (self.out / "report.json").write_text(json.dumps(self.report, indent=2, default=str))


def make_segmenter(spec: str, clip_stem: str):
    from adapter_training.perception.segment import MaskDirSegmenter, Sam3Segmenter

    if spec == "sam3":
        if not hasattr(make_segmenter, "_sam3"):
            make_segmenter._sam3 = Sam3Segmenter()
        return make_segmenter._sam3
    if spec.startswith("masks:"):
        root = Path(spec.split(":", 1)[1])
        return MaskDirSegmenter(root / clip_stem if (root / clip_stem).is_dir() else root)
    raise ValueError(f"unknown segmenter {spec!r} (sam3 | masks:<dir>)")


def make_reconstructor(spec: str):
    from adapter_training.perception.reconstruct import MeshFileReconstructor, Sam3DObjectsReconstructor

    if spec == "sam3d":
        return Sam3DObjectsReconstructor()
    if spec == "none":
        return None
    if spec.startswith("mesh:"):
        return MeshFileReconstructor(Path(spec.split(":", 1)[1]))
    raise ValueError(f"unknown reconstructor {spec!r} (sam3d | none | mesh:<file>)")


def run(args) -> int:
    name_slug = slug(args.name)
    out = args.out or OUTPUTS_DIR / "skills" / name_slug
    job = Job(out)
    job.report.update(skill=args.name, description=args.description, videos=[str(v) for v in args.videos])
    task_text = args.description.strip() or args.name
    try:
        calib = Calibration.load(args.calib)

        # 1. Video -> robot trajectories (Part 1)
        from adapter_training.video import pipeline

        part1 = out / "part1"
        v_args = pipeline.build_parser().parse_args(
            [*map(str, args.videos), "--calib", str(args.calib), "--task", task_text, "--out", str(part1),
             "--tracker", args.tracker, *(["--reuse-tracks"] if args.reuse_tracks else []),
             *([f"--base-xyz={args.base_xyz}"] if args.base_xyz else [])]
        )
        summaries = pipeline.run(v_args)
        tracked = {
            v: min(s["left"]["tracked_fraction"], s["right"]["tracked_fraction"]) for v, s in summaries.items()
        }
        job.stage("video", status="done", tracked_fraction=tracked, check_videos=str(part1))
        if all(t < args.min_tracked for t in tracked.values()):
            job.fail("video", f"hands tracked in < {args.min_tracked:.0%} of frames in every clip")

        # 2. Objects: names from the description (+ first frame), then per-clip perception
        first = read_frame(args.videos[0], 0)
        objects, source = describe_objects(args.name, args.description, first, use_claude=not args.no_claude)
        job.stage("objects", status="done", names=[o.name for o in objects], source=source)
        if not objects:
            job.fail("objects", "no objects identified from the description")
        cfg = PerceptionConfig()
        reconstructor = make_reconstructor(args.reconstructor)
        per_clip = {}
        for video in args.videos:
            seg = make_segmenter(args.segmenter, Path(video).stem)
            scene = perceive_clip(Path(video), calib, objects, seg, reconstructor, part1, cfg)
            per_clip[Path(video).name] = scene["qa"]
        ok_clips = [v for v, qa in per_clip.items() if qa and all(o.get("ok") for o in qa.values())]
        job.stage("perception", status="done", objects=per_clip, clips_ok=ok_clips, overlays=str(part1 / "scenes"))
        if not ok_clips:
            job.fail("perception", "objects could not be modelled in any clip (see objects in this stage)")
        if args.stop_after == "perception":
            return 0

        # 3-5. Simulation: replay + infer task, generate, validate
        from adapter_training.datagen import cli as datagen

        common = ["--calib", str(args.calib), "--image-width", str(args.image_width),
                  *(["--background", str(args.background)] if args.background else []),
                  *([f"--base-xyz={args.base_xyz}"] if args.base_xyz else []), *(["--shadows"] if args.shadows else [])]
        sources = out / "datagen" / "sources.hdf5"
        prep = datagen.cmd_prepare(datagen.build_parser().parse_args(
            ["prepare", str(part1 / "demos.hdf5"), "--scenes", str(part1 / "scenes"), "--out", str(sources), "--videos", *common]
        ))
        job.stage("simulation_replay", status="done", kept=prep["kept"], total=prep["total"],
                  task=prep["spec"], report=str(sources.parent / "prepare_report.json"))
        if prep["kept"] < args.min_sources:
            job.fail("simulation_replay", f"only {prep['kept']} clip(s) reproduced in simulation (need {args.min_sources})")
        if args.stop_after == "prepare":
            return 0

        generated = out / "datagen" / "generated.hdf5"
        gen = datagen.cmd_generate(datagen.build_parser().parse_args(
            ["generate", str(sources), "--out", str(generated), "--n", str(args.n), "--max-trials", str(args.n * 4),
             "--workers", str(args.workers), *common]
        ))
        job.stage("generation", status="done", **gen)
        if gen["generated"] < max(1, args.n // 4) or gen["success_rate"] < args.min_success_rate:
            job.fail("generation", f"{gen['generated']} demos at {gen['success_rate']:.0%} success "
                     f"(need >= {max(1, args.n // 4)} and >= {args.min_success_rate:.0%})")
        ok, total = datagen.cmd_validate(datagen.build_parser().parse_args(["validate", str(generated), *common]))
        job.stage("validation", status="done", replayed_ok=ok, total=total)
        if ok < total * 0.9:
            job.fail("validation", f"only {ok}/{total} demos replay successfully from their actions")

        # 6. LeRobot dataset
        repo_id = f"local/{name_slug}"
        root = dataset_root(repo_id)
        subprocess.run([sys.executable, "-m", "adapter_training.convert", str(generated), "--repo-id", repo_id,
                        "--cameras", "front_image:front", "--state-keys", "joint_pos", "--robot-type",
                        "bi_openarm_follower", "--task", task_text, "--overwrite"], check=True)
        job.stage("dataset", status="done", repo_id=repo_id, root=str(root))

        # 7. LoRA adapter (optional; needs a GPU in practice)
        result = {"status": "done", "dataset": repo_id, "task": task_text}
        if args.train:
            # Versioned: outputs/train/<skill>/v1, v2, ... and latest.txt naming the promoted version.
            train_root = OUTPUTS_DIR / "train" / name_slug
            versions = sorted(int(p.name[1:]) for p in train_root.glob("v*") if p.name[1:].isdigit())
            version = versions[-1] + 1 if versions else 1
            train_out = train_root / f"v{version}"
            latest_file = train_root / "latest.txt"
            previous = train_root / latest_file.read_text().strip() if latest_file.exists() else None
            subprocess.run([sys.executable, "-m", "adapter_training.train", "--policy", args.policy, "--dataset", repo_id,
                            "--job-name", f"{name_slug}_v{version}", "--output-dir", str(train_out), "--steps",
                            str(args.steps), "--batch-size",
                            str(args.batch_size or (32 if args.policy == "smolvla" else 64)),
                            *(["--save-freq", str(args.save_freq)] if args.save_freq else []),
                            *(["--wandb"] if args.wandb else [])], check=True)
            (train_out / "policy.txt").write_text(args.policy)
            adapter = train_out / "checkpoints" / "last" / "pretrained_model"
            job.stage("training", status="done", adapter=str(adapter), version=f"v{version}", policy=args.policy,
                      previous=str(previous) if previous else None)
            result["adapter"] = str(adapter)

            # 8. Closed-loop simulation evaluation: new adapter (and the previous one, same layouts)
            if args.sim_episodes > 0:
                from adapter_training import sim_eval

                # With --save-freq: every saved checkpoint of the new version, then keep the best.
                new_ckpts = sim_eval.list_checkpoints(train_out) if args.save_freq else [train_out]
                has_prev = bool(previous and previous.exists())
                ckpts = new_ckpts + ([previous] if has_prev else [])
                res = sim_eval.evaluate(ckpts, sources, args.calib, args.background, episodes=args.sim_episodes,
                                        videos=2, out=out / "sim_eval", base_xyz=args.base_xyz)
                best = sim_eval.pick_best(new_ckpts, res)
                if best != train_out:
                    (train_out / "best.txt").write_text(best.name)
                    adapter = sim_eval.resolve_checkpoint(train_out)
                    job.stage("training", adapter=str(adapter), best_checkpoint=best.name)
                    result["adapter"] = str(adapter)
                new_rate = res["checkpoints"][str(best)]["success_rate"]
                prev_rate = res["checkpoints"][str(previous)]["success_rate"] if has_prev else None
                info = {"new_success_rate": new_rate, "previous_success_rate": prev_rate,
                        "episodes": args.sim_episodes, "videos": str(out / "sim_eval")}
                if len(new_ckpts) > 1:
                    info["checkpoints"] = {c.name: res["checkpoints"][str(c)]["success_rate"] for c in new_ckpts}
                    info["best_checkpoint"] = best.name
                if new_rate < args.min_sim_success:
                    job.fail("sim_evaluation", f"new adapter succeeds in {new_rate:.0%} of simulated episodes "
                             f"(need {args.min_sim_success:.0%}); kept as v{version} but not promoted", **info)
                if prev_rate is not None and new_rate < prev_rate - args.max_regression:
                    job.fail("sim_evaluation", f"new adapter ({new_rate:.0%}) is worse than {previous.name} "
                             f"({prev_rate:.0%}); kept as v{version} but not promoted", **info)
                job.stage("sim_evaluation", status="done", **info)
            latest_file.parent.mkdir(parents=True, exist_ok=True)
            latest_file.write_text(f"v{version}")  # promote
            result["promoted"] = f"v{version}"

        job.report["result"] = result
        job.save()
        return 0
    except GateFailed as e:
        print(f"Stopped: {e}")
        return 1
    except Exception as e:
        job.report["result"] = {"status": "error", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]}
        job.save()
        raise


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="Skill name (from the app)")
    p.add_argument("--description", default="", help="What the skill does (from the app); also the robot's instruction")
    p.add_argument("--videos", nargs="+", type=Path, required=True)
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--background", type=Path, help="Photo of the empty workspace (realistic training images)")
    p.add_argument("--out", type=Path, help="Job folder (default outputs/skills/<skill name>)")
    p.add_argument("--tracker", default="wilor", choices=["wilor", "mediapipe"])
    p.add_argument("--segmenter", default="sam3", help="sam3 | masks:<dir>")
    p.add_argument("--reconstructor", default="sam3d", help="sam3d | none (boxes/cylinders) | mesh:<file>")
    p.add_argument("--no-claude", action="store_true", help="Don't call Claude to identify objects")
    p.add_argument("--base-xyz", help="Robot base in the table frame")
    p.add_argument("--shadows", action="store_true")
    p.add_argument("--image-width", type=int, default=512, help="Generated image width (camera aspect ratio kept)")
    p.add_argument("--reuse-tracks", action="store_true", help="Reuse <clip>_tracks.npz in <out>/part1 (tests)")
    p.add_argument("--n", type=int, default=200, help="Demos to generate")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--train", action="store_true", help="Also train a policy (SmolVLA LoRA adapter or GR00T)")
    p.add_argument("--policy", default="pi05", choices=["pi05", "smolvla", "groot"],
                   help="pi05 (π0.5 LoRA adapter, default; GPU), smolvla (small LoRA adapter, runs anywhere) or "
                        "groot (GR00T N1.7: GPU and `uv sync --extra groot`)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, help="Default: 64 for pi05/groot (80 GB GPU), 32 for smolvla")
    p.add_argument("--save-freq", type=int, help="Also save a checkpoint every N steps (default: only at the end); "
                                                 "all are compared in simulation and the best is used")
    p.add_argument("--wandb", action="store_true", help="Log training to Weights & Biases (run `wandb login` once)")
    p.add_argument("--stop-after", choices=["perception", "prepare"], help="Stop early (debugging)")
    gates = p.add_argument_group("quality gates")
    gates.add_argument("--min-tracked", type=float, default=0.8, help="Min fraction of frames with both hands tracked")
    gates.add_argument("--min-sources", type=int, default=1, help="Min clips that must replay in simulation")
    gates.add_argument("--min-success-rate", type=float, default=0.1, help="Min generation success rate")
    gates.add_argument("--sim-episodes", type=int, default=50, help="Closed-loop simulation episodes after training (0 = skip)")
    gates.add_argument("--min-sim-success", type=float, default=0.2, help="Min simulated success rate to promote an adapter")
    gates.add_argument("--max-regression", type=float, default=0.05, help="Max allowed drop vs the previous adapter")
    return p


def main() -> None:
    sys.exit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
