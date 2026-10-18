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

# Cloud: Modal H200 (`modal_app.py`)

```bash
uvx modal run --detach modal_app.py --job jobs/example_stack_sponges.json \
  --args "--n 1000 --steps 30000 --save-freq 5000 --workers 16"
```

Same `at-run`, on an H200 with everything preinstalled (SAM 3D Objects, WiLoR, ...). Checkpoints are kept
on the Modal volume `adapter-training-work` (optionally also copied to Google Drive). One-time setup (Modal
login, a secret with `HF_TOKEN`) and details: QUICKSTART.md, "Run on Modal (H200) instead". Test the image
with `AT_MODAL_NO_SECRET=1 uvx modal run modal_app.py::check`.

# One command: `at-run`

```bash
uv run at-run --setup data/setup --name "Pick up water bottle" \
    --description "Pick up the water bottle with the right hand and lift it." --videos clips/*.mp4 --train

uv run at-run --job job.json     # same thing from a JSON file -- the easiest way for an app to call it
```

It chains everything: **camera calibration → background photo → hands → robot trajectories → objects →
3D models + textures → simulation replay + task inference → generation → validation → LeRobot dataset
→ LoRA training → offline evaluation**. It finishes with a stage-by-stage summary and `report.json`.
Exit code 0 means done, 1 means stopped by a quality gate (the reason is in the report).

**Setup folder**, recorded once per camera placement. Results are cached there and redone when a clip changes:

| File | What | |
|---|---|---|
| `tag.mp4` | a few seconds with the table AprilTag visible, nobody in view | required |
| `empty.mp4` | a few seconds of the empty workspace (background for generated images) | optional, else `tag.mp4` is used |
| `checkerboard.mp4` | a checkerboard moved around the view (accurate intrinsics) | optional, else `hfov` is used |
| `setup.json` | `{"tag_size": 0.16, "tag_id": 0, "hfov": 70, "pattern": "9x6", "square": 0.025, "base_xyz": "-0.30,0,-0.45"}` | optional |

**Policy:** `"policy": "pi05"` (default, π0.5 LoRA adapter), `"smolvla"` (small LoRA adapter, runs
anywhere, use it for CPU tests) or `"groot"` (GR00T N1.7). See "Policies" under Training.

**Job file:** `{"name", "description", "videos", "setup", "train"}` plus any `at-skill` option with
underscores, e.g. `"image_width": 512`, `"n": 200`, `"workers": 8`, `"steps": 20000`. Extra command-line
flags are passed to `at-skill` as well (e.g. `--tracker mediapipe` for a CPU run).

Verified here end to end on the synthetic bottle clips (GPU models replaced by stand-ins, 1 training step):
every stage reached `done`, setup was cached on the second run and redone after `tag.mp4` changed.

---

# Skills from a clip: `at-skill` (no per-skill code)

The entry point for an app: one or more clips, a skill name and a description, and nothing else. No
tags on objects, no code per skill.

```bash
uv run at-skill --name "Pick up water bottle" \
    --description "Pick up the water bottle with the right hand and lift it." \
    --videos clips/bottle_*.mp4 --calib data/calib/camera.json --background data/calib/background.png \
    --tracker wilor --segmenter sam3 --reconstructor sam3d --n 200 --workers 8 --train
```

| Stage | What happens | Quality gate (job stops with a reason) |
|---|---|---|
| video | Part 1: hands → gripper poses → OpenArm joint trajectories (table-aware IK) | both hands tracked in ≥ 80% of frames |
| objects | Claude reads the name, description and first frame and lists the objects with a shape hint and mass (offline fallback: noun phrases) | at least one object |
| perception | SAM 3 masks each object; SAM 3D Objects reconstructs its shape; the mesh is fitted to the mask on the calibrated table (stable resting pose, position, yaw, scale); box/cylinder fallback if the fit is poor; texture baked from the clip (see below); CoACD collision parts | outline IoU ≥ 0.8 (mesh) / 0.65 (shape), plausible size |
| simulation_replay | each clip replayed in physics with grasp correction; **task inferred**: phases from grasps/releases, reference object per arm, success = the demo's final arrangement | ≥ 1 clip reproduced; all clips must share one structure |
| generation | DexMimicGen-style variations with objects moved | ≥ 10% success, ≥ n/4 demos |
| validation | every demo replayed from its actions | ≥ 90% succeed |
| dataset / training | LeRobot dataset `local/<skill>`; optional SmolVLA LoRA adapter | — |
| sim_evaluation | the new adapter (and the previously promoted one) run closed-loop in the simulated scene on the same 50 unseen object layouts | new adapter ≥ 20% success and ≤ 5 points worse than the previous one, else kept but not promoted |

`<out>/report.json` (default `outputs/skills/<skill name>/`) records every stage's status, the QA
numbers, the inferred task and the stop reason. The files for human review are
`part1/*_check.mp4`, `part1/scenes/*_objects.png` and `datagen/source_*.mp4` / `generated_*.mp4`.

### Adapter versions and simulated evaluation (`sim_eval.py`)

Each training run is a new version: `outputs/train/<skill>/v1`, `v2`, …, and `latest.txt` names the promoted
one, which is what to deploy. After training, `at-skill` / `at-run` run the new adapter and the current
`latest` in simulation:
- the objects go to 50 new random layouts, placed a bit wider than in generation and with different seeds;
  both adapters get the same layouts
- the camera image is rendered exactly as for training (robot and objects composited on the real background)
- the policy acts, the physics steps, and the skill's inferred success check decides (it must hold for 5 steps)

The new version is promoted only if it reaches `--min-sim-success` (default 20%) and isn't more than
`--max-regression` (5 points) worse than the previous one. Otherwise the job stops with the reason, and the
adapter is kept but not promoted. Results and two videos per adapter go to `<job>/sim_eval/`. Run it on its
own to compare any checkpoints:

```bash
uv run at-sim-eval --checkpoints outputs/train/pick_up_water_bottle/v2 outputs/train/pick_up_water_bottle/v1     --sources outputs/skills/pick_up_water_bottle/datagen/sources.hdf5 --calib data/setup/camera.json     --background data/setup/background.png --episodes 50 --out outputs/sim_eval_compare
```

A plain "base model without adapter" comparison isn't included: on OpenArm the base SmolVLA has untrained
input/output layers for this robot, so it would always score ~0%. Comparing against the previous version
is what tells you whether new clips or settings helped.

**Choosing the best checkpoint.** Add `--save-freq 5000` to `at-run`/`at-skill` to also save every 5k steps
(`vN/checkpoints/005000`, …). All of them are then evaluated in simulation, the best is written to
`vN/best.txt`, and that's the adapter that is promoted and reported. `at-sim-eval --all-checkpoints
--checkpoints outputs/train/<skill>/vN ...` re-ranks them on demand without listing them (see QUICKSTART.md,
"Choosing the best checkpoint").

**Simulation success is optimistic.** The policy was trained on data from this simulator. Use it to catch broken
training and to rank adapters, then confirm on the real robot.

Verified here (CPU, 1-step adapters, so 0% success as expected): two consecutive `at-run` jobs created
`v1` then `v2`. The second evaluated both on identical layouts (about 1 min per adapter for 2 episodes, plus
model loading), and promotion and the `latest.txt` update behaved as configured. The images in the evaluation
match the training images (512 × 288, textured object on the real background). The "reject and don't promote"
branch is simple but wasn't exercised end to end.

### How the task is inferred (`datagen/task.py`)

- **Phases** end at each gripper event (either arm closing or opening), in order.
- **Grasping:** a closing arm works relative to the object it grasps.
- **Releasing:** an opening arm works relative to the object the released one lands on or in (within 15 cm).
  In a handover, it works relative to the released object, since the other hand still holds it. Otherwise
  it replays its recorded motion (putting something down on the table).
- **Coordination:** an arm holding an object that the other arm works relative to moves with it, sharing one transform.
- **Success:** who holds what at the end (and lifted or not). Placed objects must end within 4 cm of where they
  were put, relative to what they were put on. Untouched objects must stay upright.

Inferred on the synthetic tests with no task-specific code:

| Skill | Inferred phases | Inferred success check |
|---|---|---|
| Bar handover | left grasps bar → right grasps bar (both relative to bar) → left releases → retreat | bar held by right, lifted |
| Bottle pickup | right grasps water bottle → retreat | water bottle held by right, lifted |

### Object textures (`perception/texture.py`)

The mesh is UV-unwrapped (xatlas) and a texture is baked in layers:
1. **Real pixels from the clip** wherever the fixed camera saw the surface (inside the object's mask, and
   in front according to a depth test against the object itself). That's the side the policy mostly
   sees at deployment, since the same camera is used.
2. **The reconstruction's colours** elsewhere. The SAM 3D worker tries the pipeline's own texture baking
   first and falls back to vertex colours (`object.json` records which).
3. **The median colour** of the object's pixels if neither exists.

The material is partly self-lit (the baked pixels already contain real shading). Randomisation only tints
textured objects mildly (±12% brightness, slight colour cast) so labels stay recognisable. Box/cylinder
fallbacks get textures the same way. `at-skill` generates 512-px-wide images by default (`--image-width`),
matching SmolVLA's input size, so object details survive.

### GPU setup (cloud machine; not testable on this laptop)

```bash
bash scripts/install_wilor.sh                    # hand tracker
hf auth login                                    # after requesting access to facebook/sam3 (gated)
# SAM 3D Objects: Linux, NVIDIA GPU with >= 32 GB VRAM, its own conda env, checkpoints from
# facebook/sam-3d-objects (gated). Step-by-step install: QUICKSTART.md, section 1. Installed there it's
# found automatically:
#   code + checkpoints: AdapterTraining/third_party/sam-3d-objects
#   python:             ~/miniforge3/envs/sam3d-objects/bin/python  (also ~/miniconda3, ~/mambaforge, ~/anaconda3)
uv run python -m adapter_training.perception.reconstruct     # shows where it was found
# Elsewhere, set e.g. SAM3D_PYTHON=/opt/conda/envs/sam3d-objects/bin/python SAM3D_REPO=/workspace/sam-3d-objects
```

SAM 3D Objects runs as a separate process (`scripts/sam3d_worker.py`), because its environment
(PyTorch 2.5, Kaolin, PyTorch3D) can't share ours. Only its **shape** is used. Size and pose are always fitted
to the calibrated table (`perception/fit.py`), so its camera conventions don't matter. Claude needs
Anthropic credentials in the environment (`ANTHROPIC_API_KEY` or an `ant auth login` profile). Pass
`--no-claude` to use the offline fallback.

### Verified here (synthetic "pick up the water bottle", 3 clips)

GPU models replaced by stand-ins: ground-truth masks (`--segmenter masks:<dir>`) and an arbitrarily
rotated, scaled and offset copy of the bottle mesh (`--reconstructor mesh:<file>`).

| Stage | Result |
|---|---|
| perception | mesh chosen in all 3 clips, outline IoU 0.978–0.98. Size 6.8–6.9 × 21.4–21.6 cm (true 7 × 7 × 22). Recovered from a mesh lying on its side at 3.7× scale. ~3 s per fit. Stands stably in physics. Textured render vs the real frame: mean colour error 4.3/255 (44% of texels from real pixels). |
| simulation_replay | 3/3 clips kept (after the two fixes below) |
| generation | 12 demos, 71% success (12/17), 99 s with 4 workers (512 px images, textured bottle) |
| validation | 12/12 |
| dataset | `local/pick_up_water_bottle` |

Handover regression with the same changes: 3/3 kept, same inferred structure, 77.5% generation success
(up from 67.5% before the table-aware IK).

Fitting check on its own: from the arbitrarily posed mesh, size within 1 mm and position within 3 mm. A box
fits the bottle's outline from one view almost as well as a cylinder (0.85 vs 0.88), which is why the
fallback uses Claude's shape hint.

### Problems found and fixed while testing

- **Idle hand making phantom grasps:** a hand that never closes had its pinch noise stretched into
  repeated open/close events. Hands whose pinch width varies by < 1.5 cm are now treated as always open.
- **Elbow through the table:** gripper-pose IK picked elbow-down solutions for objects close to the robot,
  and the forearm hit the table in simulation. The IK now keeps links 3–7 and the grippers ≥ 1 cm above the table.
- **Silhouette renderer holes:** OpenCV's multi-polygon fill uses even-odd filling and punched holes where front
  and back faces overlap. Triangles are now filled one by one.
- **Re-anchoring twitch:** consecutive phases referencing the same object now keep their transform.
- **Upright objects fitted lying down:** a bottle is more *stable* on its side, so the top-ranked resting poses
  were all side-lying variants. Candidates are now one per distinct "up" direction.
- **Mesh simplification ate thin parts** (the bottle's neck), so it's only used if the bounding box is preserved.
  Fitting renders the outline from ~5,000 surface samples instead of triangles: 87 s → 3 s per fit at the same accuracy.

### What's not covered

- **Never run here:** SAM 3, SAM 3D Objects and WiLoR (no GPU), and real videos. Claude worked in one
  quick test call. Check the overlays and videos of the first real jobs.
- **Skills without grasps** (pushing, wiping, pressing) have no events to split on.
- **Hinged or flexible objects** (drawers, lids, cloth) and **liquids**.
- **Several identical objects:** SAM 3 keeps the highest-scoring instance per name.
- **Objects hidden in the first frame** (perception uses frame 0).
- **Texture on sides the camera never saw** comes from the reconstruction (SAM 3D Objects' own baked
  texture or vertex colours), so it's a guess. With a stand-in mesh that has no colours, it's the median colour.
- **Speed:** CoACD took ~40 s per object on this CPU. It runs once per object per skill.

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
# Photo of the empty workspace from the fixed camera (median of a short clip; no robot, no people)
uv run at-datagen background data/calib/empty_workspace.mp4 --out data/calib/background.png

uv run at-datagen prepare outputs/video2robot/handover/demos.hdf5 --calib data/calib/camera.json \
    --background data/calib/background.png --out outputs/datagen/handover/sources.hdf5 --videos
uv run at-datagen generate outputs/datagen/handover/sources.hdf5 --n 200 --workers 8 \
    --calib data/calib/camera.json --background data/calib/background.png --object-rgba 0.85,0.35,0.1,1 \
    --shadows --out outputs/datagen/handover/generated.hdf5
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

**Generate** (`generate.py`, `task.py`): phases, reference objects and the success check are inferred
from the replay (see the `at-skill` section). For the handover, the inferred phases are: left grasps the bar,
right grasps it (both relative to the bar, sharing one transform: DexMimicGen's coordination subtask),
left releases, retreat. Objects come from perception scene files (`--scenes`) or, for older data, the bar's
tag. Each new episode samples every object's pose (±4 cm, ±5 cm, ±20° yaw around a source demo's,
footprints not overlapping), transforms each phase's commanded gripper poses (an arm keeps its
transform while its reference doesn't change), interpolates between phases, runs IK + physics, and keeps
it only on success. Images are rendered only for kept episodes. Options:
`--dx/--dy/--dyaw`, `--noise`, `--workers`, `--shadows`, `--randomize`.

**Validate** replays each saved demo's recorded actions from its first state and checks success again.

### Making generated images match the real camera

Code: `src/adapter_training/render/` (`composite.py`, `overlay.py`), `datagen/appearance.py`,
`plugins/lerobot_robot_openarm_overlay/`.

1. **Real background:** with `--background`, only the robot and object are rendered (from the calibrated
   camera, including its principal point and lens distortion) and pasted onto your photo of the
   empty workspace, so the table, lighting and surroundings are real. `--shadows` also darkens
   the background where the robot and object would cast shadows (fast on a GPU, slow on integrated graphics).
2. **Same robot at deployment (Phantom's trick):** run the policy with `--robot.type=bi_openarm_overlay`
   instead of `bi_openarm_follower`. It's the same driver, but each camera frame gets the simulated robot drawn
   at its measured joint angles over the real one. So the policy sees the rendered robot in training and on
   the robot. The object is never overlaid: the policy sees the real one. Use it for `lerobot-record`
   too if you record real demos to mix in.
3. **Randomisation** (`--randomize`, default 1.0): per episode, light direction and intensity, object hue,
   saturation and brightness around `--object-rgba` (set it to your real object's colour), image
   brightness and contrast ±15%, and ~0.5 cm / 0.5° camera jitter to absorb calibration error between the
   training images and the deployment overlay.
4. **Aspect ratio:** images default to 320 px wide with the camera's aspect ratio (`--image-size auto`),
   in Part 1 and Part 2, so frames match the live camera instead of being squeezed to 4:3.

Verified on the synthetic scene: the warp puts 3D points within 0.7 px of OpenCV's projection with
distortion and an off-centre principal point. A pasted marker lands 4.6 px (at 640 px wide) from
the real tag, consistent with the synthetic calibration's ~8 mm error. For the same joint state, the
deployment overlay and the training composite differ on < 0.25% of pixels. The plugin is discovered
by LeRobot and constructs a `BiOpenArmOverlay` robot. The overlay takes ~130 ms/frame at 320 px on this
laptop's integrated graphics (expect far less on a GPU; lower `--robot.overlay_render_width` if needed).

### Verified on synthetic demos (this laptop, CPU)

| Step | Result |
|---|---|
| prepare, 3 synthetic video demos | 3/3 kept (left grasp corrections ≤ 2 mm, right 3–4 cm because the scripted handover point was rough) |
| generate | ~68% of attempts succeed (27/40); failures are missed left grasps or failed handovers |
| speed | ~4 s physics per episode + ~15 s rendering 244 frames without shadows (integrated GPU); 8 demos in 104 s with 4 workers |
| validate | 10/10 and 8/8 generated demos succeed again from their recorded actions |
| at-convert | `bi_openarm_follower` names, 30 fps, task text, camera-aspect images |

### What went wrong on the way (and how it's handled)

- **Grasps on edges:** the first correction only put the grasp point *inside* the object, so the right
  gripper pinched the bar's edge and dropped it. It now centres the fingers along the closing axis.
- **Fingers clipping the object on approach:** the human's pinch width maps to only ~85% open, leaving
  < 1 cm clearance. Grippers are now commanded fully open/closed in simulation.
- **Reachability:** OpenArm's right arm can only reach the handover point with the gripper pointing
  mostly forward (sideways approaches had 3–10 cm IK error). Demonstrate handovers accordingly.

### Limitations

- **Remaining appearance gaps:** the object is still rendered in training but real at deployment
  (match its colour with `--object-rgba`; randomisation covers some difference). Parts of the real
  robot the model doesn't include (cables, mounts) stay visible under the overlay. Without
  `--shadows`, the real robot casts shadows the training images lack. Mixing in a few real demos
  recorded with `bi_openarm_overlay` is still worthwhile.
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
| `--wandb` | log to Weights & Biases (once per machine: `uv run wandb login`, or set `WANDB_API_KEY`); also on `at-run`/`at-skill` |
| `-- <args>` | pass anything else to `lerobot-train`, e.g. `-- --wandb.project=my_project` |

One adapter per task: convert each task to its own dataset and train with its own `--job-name`.
For a shared multi-task adapter, write all tasks into one dataset (each episode keeps its own `task` string).

### Policies

| | π0.5 (`pi05`, default for at-run/at-skill) | SmolVLA (`smolvla`) | GR00T N1.7 (`groot`) |
|---|---|---|---|
| Base model | `lerobot/pi05_base` (~3B) | `lerobot/smolvla_base` (450M) | `nvidia/GR00T-N1.7-3B` |
| What's trained per skill | LoRA adapter (action expert attention + projections) | LoRA adapter | projector + action head (no LoRA in LeRobot) |
| Per-version size | small (adapter) | small (adapter) | whole policy (several GB) |
| Hardware | NVIDIA GPU; defaults sized for 80 GB (batch 64, bf16) | anything, incl. CPU smoke tests | NVIDIA GPU |
| Extra setup | accept the gated `google/paligemma-3b-pt-224` licence + `hf auth login` | none | `uv sync --extra groot` |
| Licence | Apache 2.0 | Apache 2.0 | NVIDIA Open Model License |

### π0.5

```bash
hf auth login            # after accepting https://huggingface.co/google/paligemma-3b-pt-224 (gated tokenizer)
uv run at-train --policy pi05 --dataset local/pick_up_water_bottle --steps 20000 --batch-size 64
uv run at-run --job job.json --train          # π0.5 is the default; "policy": "pi05" to be explicit
```

The recipe follows the [LeRobot π0.5 docs](https://huggingface.co/docs/lerobot/pi05) and LeRobot's PEFT support:
- `--policy.pretrained_path=lerobot/pi05_base`, which loads weights only, so cameras and state/action sizes come
  from your dataset (no rename map needed)
- bf16, LoRA rank 64 on LeRobot's default π0.5 targets
- learning rate 2.5e-4 → 2.5e-5 (10× π0.5's full fine-tune rate, the usual LoRA scaling)
- π0.5's default quantile normalization (the converted datasets already contain `q01`/`q99`)
- 50-step action chunks

`--pi05-gradient-checkpointing` trades speed for memory if you move to a smaller GPU. `--no-peft` does
LeRobot's full fine-tune instead (80 GB).

Verified here: LeRobot's parser accepts the generated π0.5 + LoRA config (PI05Config, bf16, quantile
normalization, LoRA r=64), the training dataset has the quantile statistics π0.5 needs, and `at-skill`/`at-run`
default to it. **Not run:** no GPU, and the base model is several GB. On Windows the base checkpoint is
downloaded and passed as a local path, the same workaround as for SmolVLA.

### GR00T N1.7

```bash
uv sync --extra groot                      # GR00T dependencies (diffusers, timm, dm-tree, decord)
uv run at-train --policy groot --dataset local/pick_up_water_bottle --steps 20000 --batch-size 64
uv run at-run --job job.json --policy groot --train      # or "policy": "groot" in the job file
```

This follows the [LeRobot GR00T recipe](https://huggingface.co/docs/lerobot/groot):
- base model `nvidia/GR00T-N1.7-3B` (LeRobot dropped N1.5)
- `embodiment_tag=new_embodiment`, the slot GR00T reserves for robots it wasn't trained on
- 16-step action chunks
- relative arm actions with the grippers kept absolute (`relative_exclude_joints=["gripper"]` matches
  `left_/right_gripper.pos`)
- bf16, image augmentation

The vision-language backbone stays frozen while the projector and flow-matching action head train.
`--groot-tune-visual` / `--groot-tune-llm` also unfreeze those (much more memory).

How it differs from SmolVLA:
- **No LoRA.** LeRobot's GR00T `lora_*` settings are deprecated and unused, so a GR00T "adapter" is the
  fine-tuned action head and projector, and checkpoints hold the whole 3B policy (several GB per version).
  Prune old versions under `outputs/train/<skill>/` on the cloud machine.
- **Needs an NVIDIA GPU.** It's 3B parameters in bf16. Simulated evaluation runs the policy too, so it
  also needs the GPU.
- **License:** NVIDIA Open Model License.

Verified here: LeRobot's own config parser accepts the generated GR00T training command (GrootConfig,
frozen backbone, trained projector/action head, relative actions with grippers excluded). The `groot`
extra installs on Windows. **Not run:** no GPU, and the base model is several GB. Versioning, the
simulated comparison and promotion work the same for both policies, and a GR00T version can be compared
against a SmolVLA version of the same skill.

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

For GR00T checkpoints, add `--policy.base_model_path=nvidia/GR00T-N1.7-3B` (as in the LeRobot docs).

**OpenArm with a policy trained on generated data:** use the overlay robot, and the camera at the
resolution and position it was calibrated at:

```bash
lerobot-rollout --strategy.type=base \
  --robot.type=bi_openarm_overlay \
  --robot.left_arm_config.port=can0 --robot.left_arm_config.side=left \
  --robot.right_arm_config.port=can1 --robot.right_arm_config.side=right \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 30}}" \
  --robot.overlay_calib=data/calib/camera.json \
  --task="Pass the bar from the left hand to the right hand." \
  --policy.path=outputs/train/<run>/checkpoints/last/pretrained_model
```

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
