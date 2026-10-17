# AdapterTraining

Turn **human demonstration videos** into **OpenArm** robot trajectories, and train **SmolVLA LoRA
adapters** on them (or on MimicGen-style data) with LeRobot.

```
human RGB video ─► at-video2robot (hands → grippers → OpenArm IK) ─┬────────────────────────┐
                   at-datagen (DexMimicGen-style generation in MuJoCo) ◄┘                        │
teleop demos ─► MimicGen / DexMimicGen / Isaac Lab Mimic ─────────────────────────────────────┴─► robomimic HDF5
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
4. **Object (needed for Part 2):** a box-shaped bar (default 14 × 5 × 4 cm) with a second 36h11 tag
   (id 1, ~4.4 cm) centred on its top face, the tag's top edge pointing along the bar's long axis. Keep
   the tag visible for the first second of each clip, before your hand reaches it.
5. **Grasps:** pinch with thumb and index only. Left hand: pinch the bar's long sides from behind.
   Right hand: take the far end from the side, top and bottom, with clear space between the hands.
   OpenArm's wrist can rarely point straight down, but forward grasps are easy.
6. **Workspace:** keep your hands where the arms can reach. With the default robot placement that's
   roughly 20 cm behind to 30 cm in front of the tag, within ±40 cm sideways and up to ~30 cm high.
7. **Clips:** one handover per clip, 5–15 s, both hands visible the whole time. Vary the object
   position between clips.

### Run

```bash
# Calibrate (once per camera placement). A checkerboard gives better intrinsics than --hfov.
uv run at-calibrate intrinsics data/calib/checkerboard.mp4 --pattern 9x6 --square 0.025
uv run at-calibrate table data/calib/tag.mp4 --tag-size 0.16 --intrinsics data/calib/intrinsics.json
#   -> data/calib/camera.json (+ camera.png with the table axes drawn: red x away from robot, blue z up)

# Videos -> robot trajectories (all clips into one demos.hdf5), tracking the object's tag for Part 2
uv run at-video2robot data/videos/handover_*.mp4 --calib data/calib/camera.json --tracker wilor \
    --object-tag-id 1 --object-tag-size 0.044 --object-size 0.14,0.05,0.04 \
    --task "Pass the bar from the left hand to the right hand." --out outputs/video2robot/handover

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
uv run at-synthetic-demo --demos 3
uv run at-calibrate table data/synthetic/tag.mp4 --tag-size 0.16 --hfov 70 --out data/synthetic/camera.json
uv run at-video2robot data/synthetic/handover_*.mp4 --calib data/synthetic/camera.json --reuse-tracks \
    --out data/synthetic --object-tag-id 1 --object-tag-size 0.044 \
    --task "Pass the bar from the left hand to the right hand."
```

Scripted bar handovers with known hand positions (3 mm keypoint noise and a 6-frame dropout), the bar
moved between demos. Verified here: camera calibrated from the tag video; arms track their targets to
~1 mm during the grasp, carry and handover (up to ~3 cm near the robot's body at the start and end,
the edge of the reachable area); bar tag found before the grasp (1–18 mm single-frame position error,
reduced in Part 2 by taking the median); converts to a LeRobot dataset with the driver's motor names.

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

---

# Part 2: Multiply demos in simulation (DexMimicGen-style)

NVIDIA's public DexMimicGen repo ships environments and datasets but no generation code (that lives in
Isaac Lab Mimic, which needs Linux + Isaac Sim). This is a reimplementation of the method on MuJoCo,
reusing the OpenArm model and IK from Part 1. Code: `src/adapter_training/datagen/`.

```
demos.hdf5 (Part 1, with object tag)
  ─► at-datagen prepare    replay each demo in physics, nudge grasps onto the object, keep successes
  ─► sources.hdf5          simulation-validated demos + gripper/object poses + phase boundaries
  ─► at-datagen generate   new object poses → transform each phase's motion → execute → keep successes
  ─► generated.hdf5        ─► at-datagen validate ─► at-convert ─► at-train
```

```bash
uv run at-datagen prepare outputs/video2robot/handover/demos.hdf5 --calib data/calib/camera.json \
    --out outputs/datagen/handover/sources.hdf5 --videos
uv run at-datagen generate outputs/datagen/handover/sources.hdf5 --n 200 --workers 8 \
    --calib data/calib/camera.json --out outputs/datagen/handover/generated.hdf5
uv run at-datagen validate outputs/datagen/handover/generated.hdf5
uv run at-convert outputs/datagen/handover/generated.hdf5 --repo-id local/handover_gen \
    --cameras front_image:front --state-keys joint_pos --robot-type bi_openarm_follower
```

**Simulation** (`env.py`): OpenArm with position-controlled, gravity-compensated joints and
position-controlled fingers (the URDF's finger mimic is rebuilt as an equality constraint), a
collidable table, and a free box object. Robot links collide with the table and object but not with
each other. Success = right gripper holds the object off the table and the left has let go.

**Prepare** (`source.py`): the object is placed at the median of its tag detections before the first
grasp, flat on the table. Each gripper trajectory is replayed through IK, with the gripper fully open
until a grasp and fully closed until a release (as in MimicGen). Over the second before each grasp,
the gripper is shifted so its fingers straddle the object's centre along the closing axis and sit
≥ 1.5 cm inside its edges. Demos are kept only if both grasps hold and the handover succeeds.
`prepare_report.json` lists every demo, its grasp corrections and why rejected ones failed.

**Generate** (`generate.py`, `task.py`): phases come from gripper events. `left_grasp` (left arm
relative to the object, right arm replayed as-is), `handover` (both arms relative to the object in the
left hand, sharing one transform so their relative motion is kept — DexMimicGen's coordination
subtask), `retreat` (replayed as-is). Each new episode samples an object pose (±4 cm, ±5 cm, ±20° yaw
around a source demo's), transforms each phase's commanded gripper poses, interpolates between phases,
runs IK + physics, and keeps it only on success. Images are re-rendered only for kept episodes, from
the calibrated camera if `--calib` is given. Options: `--dx/--dy/--dyaw`, `--noise`, `--workers`,
`--shadows`.

**Validate** replays each saved demo's recorded actions from its first state and checks success again.

### Verified on synthetic demos (this laptop, CPU)

| Step | Result |
|---|---|
| prepare, 3 synthetic video demos | 3/3 kept (left grasp corrections ≤ 2 mm, right 3–4 cm because the scripted handover point was rough) |
| generate | 77–80% of attempts succeed; failures are missed left grasps or failed handovers |
| speed | ~4 s physics per episode + ~15 s rendering 244 frames at 320×240 without shadows (integrated GPU); 8 demos in 104 s with 4 workers |
| validate | 10/10 and 8/8 generated demos succeed again from their recorded actions |
| at-convert | `bi_openarm_follower` names, 30 fps, task text, 320×240 images |

### What went wrong on the way (and how it's handled)

- **Grasps on edges:** the first correction only put the grasp point *inside* the object, so the right
  gripper pinched the bar's edge and dropped it. It now centres the fingers along the closing axis.
- **Fingers clipping the object on approach:** the human's pinch width maps to only ~85% open, leaving
  < 1 cm clearance. Grippers are now commanded fully open/closed in simulation.
- **Reachability:** OpenArm's right arm can only reach the handover point with the gripper pointing
  mostly forward (sideways approaches had 3–10 cm IK error). Demonstrate handovers accordingly.

### Limitations

- **Sim-to-real:** generated images are MuJoCo renders (grey robot meshes, plain table). Mix real data
  in for training, e.g. Part 1 demos once the robot overlay exists, or some teleop.
- **Approximate physics:** gripper-gripper collisions are off, so the generator can't detect the two
  hands clashing. Keep handover grasps on opposite ends of the object.
- **One task:** phases and the success check are written for the handover (`task.py`, `env.py`).
  Other tasks need their own phases, object and success check.
- **Real video demos aren't tested yet.** Expect a lower `prepare` keep rate than with synthetic
  demos, and check `prepare_report.json` and the `source_*.mp4` videos.

---

# Training (LeRobot)

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
