# AdapterTraining

Turn **human demonstration videos** into **OpenArm** robot trajectories, and train **SmolVLA LoRA
adapters** on them (or on MimicGen-style data) with LeRobot.

```
human RGB video ─► at-video2robot (hands → grippers → OpenArm IK) ─┐
teleop demos ─► MimicGen / DexMimicGen / Isaac Lab Mimic ──────────┴─► robomimic HDF5
                                                              │  at-convert
                                                              ▼
                                                     LeRobot dataset (data/lerobot/…)
                                                              │  at-train  (lerobot-train + PEFT LoRA)
                                                              ▼
                                         adapter checkpoint (outputs/train/…/pretrained_model)
                                                              │  at-eval   (offline action error)
                                                              ▼
                                                  lerobot-rollout on the robot
```

Built on `lerobot==0.6.1` (Python 3.12, managed with `uv`).

## Setup

```bash
uv sync
```

`pyproject.toml` sets `native-tls = true` and the package injects `truststore`, so downloads use the
Windows certificate store. That's needed on this machine because something re-signs HTTPS traffic.

---

# Part 1: Human video → OpenArm trajectories

```
RGB video ─► AprilTag calibration ─► 3D hand tracking ─► thumb/index → gripper pose + opening
          ─► OpenArm v1 IK (MuJoCo + mink) ─► check video + demos.hdf5 ─► at-convert ─► LeRobot
```

Code: `src/adapter_training/video/` (`camera.py`, `hands.py`, `retarget.py`, `robot.py`, `ik.py`, `pipeline.py`).

### One-time setup

```bash
uv run at-fetch-robot          # clones enactic/openarm_description, builds the MuJoCo model
bash scripts/install_wilor.sh  # GPU machine only: WiLoR hand tracker (MediaPipe works without it)
```

### Recording guide

1. **Camera:** fixed on a tripod for the whole session. It must not move after calibration.
2. **AprilTag:** print a 36h11 tag (id 0) as large as fits on A4 (~16 cm black square) and measure the
   black square. In the synthetic test, a 16 cm tag gave ~8 mm / 0.4° camera error versus ~25 mm / 1.3° for a 6 cm one.
   Lay it flat on the table with its **top edge pointing away from the demonstrator**. That defines the
   table frame: origin at the tag centre, x away from the robot side, y to the robot's left, z up.
3. **Demonstrator:** stand where the robot will stand (behind the table, facing the camera). Your left
   hand drives the robot's left arm.
4. **Grasps:** pinch with thumb and index only, on a small block (< 3 cm). Approach from the front or
   side. OpenArm's wrist can rarely point straight down, but forward grasps are easy.
5. **Workspace:** keep your hands where the arms can reach. With the default robot placement that's
   roughly 20 cm behind to 30 cm in front of the tag, within ±40 cm sideways and up to ~30 cm high.
6. **Clips:** one handover per clip, 5–15 s, both hands visible the whole time. Vary the block position
   between clips.

### Run

```bash
# Calibrate (once per camera placement). A checkerboard gives better intrinsics than --hfov.
uv run at-calibrate intrinsics data/calib/checkerboard.mp4 --pattern 9x6 --square 0.025
uv run at-calibrate table data/calib/tag.mp4 --tag-size 0.16 --intrinsics data/calib/intrinsics.json
#   -> data/calib/camera.json (+ camera.png with the table axes drawn: red x away from robot, blue z up)

# Videos -> robot trajectories (all clips into one demos.hdf5)
uv run at-video2robot data/videos/handover_*.mp4 --calib data/calib/camera.json --tracker wilor \
    --task "Pass the block from the left hand to the right hand." --out outputs/video2robot/handover

# -> LeRobot dataset (exact command is printed at the end of at-video2robot)
uv run at-convert outputs/video2robot/handover/demos.hdf5 --repo-id local/handover \
    --cameras front_image:front --state-keys joint_pos --robot-type bi_openarm_follower
```

Per video you get `<video>_check.mp4` (tracked hands and gripper targets next to the robot rendered
from the same viewpoint, with the per-frame IK error), `<video>_tracks.npz` (re-run later stages with
`--reuse-tracks`), and an entry in `summary.json` with tracking coverage and IK error. **Watch the check
videos** and drop clips where the robot doesn't do what you did.

`demos.hdf5` per demo: `obs/front_image`, `obs/joint_pos` (16 values in degrees, named like LeRobot's
`bi_openarm_follower`: `left_joint_1.pos` … `left_gripper.pos`, `right_…`), `obs/{left,right}_eef_pos/quat`,
`obs/{left,right}_gripper_opening`, and `actions` (next-step joint targets, same names).

### Test without filming

```bash
uv run at-synthetic-demo
uv run at-calibrate table data/synthetic/tag.mp4 --tag-size 0.16 --hfov 70 --out data/synthetic/camera.json
uv run at-video2robot data/synthetic/handover.mp4 --calib data/synthetic/camera.json --reuse-tracks \
    --out data/synthetic --task "Pass the block from the left hand to the right hand."
```

A scripted handover with known hand positions (3 mm keypoint noise and a 6-frame dropout). Verified
here: camera calibrated from the tag video, both arms track their targets with ~1 mm median / 1.2 mm
95th-percentile error, all joints inside LeRobot's OpenArm limits, and the result converts to a
LeRobot dataset with the driver's motor names.

### Check before using on the real robot

- **Robot placement:** the robot base is assumed 30 cm behind the tag, 45 cm below the table top
  (`--base-xyz -0.30,0,-0.45`, in `robot.py`). Measure your real setup and pass it.
- **Joint conventions:** joint angles are exported as URDF angles in degrees, assuming LeRobot's
  OpenArm motor zero and direction match the URDF. Verify by moving the real robot slowly to an
  exported pose before running anything fast.
- **Gripper command:** assumed 0° = closed and -65° = open (LeRobot's default range). Swap
  `gripper_motor_range` in `robot.py` if yours is reversed.
- **Camera frames show the human, not the robot.** Fine for trajectories (and for DexMimicGen, which
  renders its own images), but a policy trained on these frames sees hands at training time and a
  robot at test time. The robot overlay (inpainting + rendering) is the next stage to build.
- **Depth from one RGB camera is approximate.** MediaPipe is the CPU fallback. Use `--tracker wilor`
  on the GPU. WiLoR is wired in but untested here (no GPU), so check its first check video carefully.
- **Orientation:** OpenArm's reachable wrist orientations are limited. The IK weights position over
  orientation and picks the reachable one of the two equivalent gripper orientations.

## 1. Get data

Example data is the official MimicGen release (1000 generated demos per task, Panda arm, 84×84 images):

```bash
uv run at-download --list                 # files and sizes (1–13 GB each)
uv run at-download core/stack_d1.hdf5     # smallest generated task, 1.07 GB -> data/raw/
```

Your own MimicGen / Isaac Lab Mimic output uses the same robomimic HDF5 layout.

## 2. Convert to LeRobot format

```bash
uv run at-convert data/raw/core/stack_d1.hdf5 --max-episodes 50
# -> data/lerobot/mimicgen_stack_d1  (repo id local/mimicgen_stack_d1)
```

| Option | Default | Notes |
|---|---|---|
| `--cameras` | `agentview_image:front,robot0_eye_in_hand_image:wrist` | `hdf5_key:name` pairs |
| `--state-keys` | `robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos` | concatenated into `observation.state` |
| `--task` | looked up from env name (`tasks.py`) | **must equal the prompt used at deployment** |
| `--flip` | off | raw robosuite renders are upside down; MimicGen releases are already upright |
| `--no-videos` | off | store PNG frames instead of MP4 |

For a bimanual robot, list both arms' keys, e.g.
`--state-keys robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos,robot1_eef_pos,robot1_eef_quat,robot1_gripper_qpos`.
SmolVLA pads state/action up to 32 dims, so two arms fit.

Streaming video encoding is on by default (~0.5 s per episode vs ~15 s without it on Windows).

## 3. Train a LoRA adapter

```bash
# GPU run (≈4 h for 20k steps on an A100 per the LeRobot docs)
uv run at-train --dataset local/mimicgen_stack_d1 --steps 20000 --batch-size 64

# CPU smoke test (checks the pipeline end to end, not a useful model)
uv run at-train --dataset local/mimicgen_stack_d1 --steps 4 --batch-size 2 --num-workers 0 \
    --warmup-steps 1 --decay-steps 4 --job-name smoke_lora --output-dir outputs/train/smoke_lora
```

Defaults follow the [LeRobot PEFT guide](https://huggingface.co/docs/lerobot/peft_training): LoRA r=64,
alpha=64, lr 1e-3 → 1e-4, targeting the action expert's `q_proj`/`v_proj` plus state/action projections.
10% of episodes are held out and validation loss is logged at each checkpoint.

| Option | Purpose |
|---|---|
| `--rank`, `--alpha` | LoRA size/scaling; try 16–32 with little data |
| `--init pissa` | alternative LoRA init (any PEFT `init_lora_weights` value) |
| `--full-train-projections` | fully train state/action projections (new robot layout), LoRA elsewhere |
| `--no-peft` | full fine-tune of the action expert, for comparison against LoRA |
| `--dry-run` | print the underlying `lerobot-train` command |
| `-- <args>` | pass anything else to `lerobot-train`, e.g. `-- --wandb.enable=true` |

One adapter per task: convert each task to its own dataset and train with its own `--job-name`.
For a shared multi-task adapter, write all tasks into one dataset (each episode keeps its own `task` string).

## 4. Evaluate offline

```bash
uv run at-eval --checkpoint outputs/train/smoke_lora --dataset local/mimicgen_stack_d1
```

Replays held-out episodes through the same inference path as the robot (preprocessor → `select_action`
with action chunking → postprocessor) and reports action MSE/MAE per dimension, plus an
always-predict-the-mean baseline. Results are saved next to the checkpoint in `offline_eval.json`.
This is a sanity check and a way to compare adapters. It isn't a success rate.

## 5. Deploy

```bash
lerobot-rollout --strategy.type=base \
  --robot.type=<your_robot> \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --task="Stack the red cube on top of the green cube." \
  --policy.path=outputs/train/<run>/checkpoints/last/pretrained_model
```

Camera names, state layout and task text must match the training dataset. Your robot needs a LeRobot
`Robot` driver.

## Verified smoke test (this laptop, CPU only)

| Step | Result |
|---|---|
| Convert 50 demos of `stack_d1` | ~25 s, 5 MB; decoded frames match the HDF5 (mean abs error ~1.8/255), actions exact |
| `at-train`, 4 steps, batch 2 | ~17 s/step; 3.0M trainable / 453M total params; adapter saved |
| `at-eval`, 60 frames | ~1 min; adapter + base model reload and run through the deployment inference path |

After 8 training frames the model is (as expected) worse than the predict-the-mean baseline, so the
numbers only show the plumbing works.

## Notes / limitations

- **This machine has no CUDA GPU**, so only the CPU smoke test runs here. Use a cloud GPU or Colab for real training.
- The MimicGen example data is a single-arm Panda in simulation; it validates the pipeline, not your robot.
- **Windows workaround:** LeRobot 0.6.1 mangles Hub ids passed to `--policy.path` into `lerobot\smolvla_base`.
  `at-train` downloads the base model first and passes the local path. The adapter's `adapter_config.json`
  then points at that local snapshot, so copying an adapter to another machine requires that machine to
  have `lerobot/smolvla_base` too (or editing `base_model_name_or_path`).
- A `libtorchcodec ... Could not load this library` traceback in the logs is harmless: on Windows the
  scripts decode video with PyAV instead.
- LeRobot's validation pass uses all held-out frames unless capped; `at-train` caps it with
  `--max-eval-samples` (default 1000).
- **Memory:** loading SmolVLA needs several GB of free RAM. On this 16 GB laptop, training segfaulted
  while loading the model until other applications were closed.
- `smolvla_base` loads `SmolVLM2-500M-Video-Instruct` weights on first use, so the first run downloads ~2 GB.
- Converting all 1000 demos of `stack_d1` takes ~10 min and ~100 MB.
