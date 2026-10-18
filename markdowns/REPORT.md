# AdapterTraining: full report

Everything about this repository: what it is for, how it works, why it is built this way, what was
discussed, considered, tried, broken and fixed, what is verified and what isn't, and what to do next.

**Companion documents:** `QUICKSTART.md` (train an adapter step by step), `OVERVIEW.md` (a few-page
summary), `README.md` (every command and option).

## How to read this

| If you want… | Read |
|---|---|
| the two-minute version | §1 Executive summary |
| the ideas without the jargon | §2 Concepts primer |
| why it's built this way (research and decisions) | §3 Goals, §4 Discussion and decisions, §9 Alternatives |
| how each part works | §5 System overview, §6 Method |
| evidence | §7 Verification and results, §8 Problems found and fixed |
| what could go wrong | §10 Limitations, §11 Untested components |
| how to work on the code | §14 Engineering reference |

---

## Contents

1. [Executive summary](#1-executive-summary)
2. [Concepts primer](#2-concepts-primer)
3. [Goals and constraints](#3-goals-and-constraints)
4. [Discussion and decisions (chronological)](#4-discussion-and-decisions-chronological)
5. [System overview](#5-system-overview)
6. [Method](#6-method)
7. [Verification and results](#7-verification-and-results)
8. [Problems found and fixed](#8-problems-found-and-fixed)
9. [Alternatives considered and rejected](#9-alternatives-considered-and-rejected)
10. [Limitations and risks](#10-limitations-and-risks)
11. [Untested components](#11-untested-components)
12. [The example videos in `examples/`](#12-the-example-videos-in-examples)
13. [Recommended next steps](#13-recommended-next-steps)
14. [Engineering reference](#14-engineering-reference)
15. [Licences of components](#15-licences-of-components)
16. [Commit history](#16-commit-history)
17. [Glossary](#17-glossary)
18. [References](#18-references)

---

## 1. Executive summary

**Goal.** Teach a two-arm OpenArm robot new manipulation skills from **ordinary videos of a person
doing the task**, with no teleoperation, no markers on objects and no per-skill code. A user (through an
app) supplies a few clips, a skill name and a description. The system returns a trained skill model for
a vision-language-action (VLA) policy.

**What was built.** A Python/uv repository (`AdapterTraining/`, ~5,700 lines of Python across 42 files) with
one command, `at-run`, that chains:

1. **Camera calibration** from an AprilTag on the table, plus a photo of the empty workspace.
2. **Human video → robot motion:** 3D hand tracking, thumb/index → two-finger gripper mapping, and
   inverse kinematics on the OpenArm URDF.
3. **Object perception without markers:** Claude names the objects from the description, SAM 3 finds
   them, SAM 3D Objects reconstructs their 3D shape, the shape is fitted to the calibrated table, and a
   texture is baked from the real video.
4. **Simulation (MuJoCo):** each clip is replayed in physics with grasp correction, and the task
   (phases, reference objects, success check) is **inferred automatically** from the replay.
5. **Data generation:** a reimplementation of the DexMimicGen method multiplies the clips into
   hundreds of demos with objects moved to new positions, keeping only physically successful ones.
6. **Realistic training images:** robot and objects composited onto the real background photo with the
   real lens model and appearance randomisation.
7. **Training:** a LoRA adapter on **π0.5** (default), SmolVLA, or a fine-tuned GR00T N1.7, with
   versioning.
8. **Evaluation:** closed-loop success rate in simulation against the previous version, with a
   promotion gate.
9. **Deployment:** a LeRobot robot plugin that draws the simulated robot over the live camera image
   (the "Phantom" trick), so the policy sees the same robot as in training.

Every stage has a **quality gate**. A job stops with a plain-language reason in `report.json` rather
than training on bad data.

**Status.** All stages run end to end on synthetic test scenes on a development machine without an
NVIDIA GPU:
- A synthetic two-arm bar handover and a synthetic water-bottle pickup are both inferred correctly with
  no task-specific code.
- 68–78% of generation attempts succeed, and 100% of saved demos replay successfully from their actions.
- Object reconstruction is fitted to within millimetres, and the textured render matches the real frame
  to 4.3/255.

**Not yet run:** the GPU components (WiLoR, SAM 3, SAM 3D Objects, π0.5, GR00T), real videos and the
real robot.

**Biggest open questions:**
- how accurate monocular hand tracking is on real clips
- how many real clips survive simulation replay
- how well policies trained on composited simulated data transfer to the real robot
- whether the robot's joint-zero and gripper-direction conventions match the URDF

---

## 2. Concepts primer

For readers without a robotics or ML background. Engineers can skip to §3.

| Term | Meaning here |
|---|---|
| **VLA (vision-language-action model)** | A neural network that looks at camera images, reads an instruction ("pick up the bottle") and outputs robot motor commands. Examples: SmolVLA, π0.5, GR00T. |
| **Base model / fine-tuning** | A VLA pretrained on lots of robot data (the base) is specialised to a robot and skill by training it further on a small dataset (fine-tuning). |
| **LoRA adapter** | A cheap fine-tuning method: the base model is frozen and small extra matrices are trained. The "adapter" is just those matrices, typically megabytes, and swappable per skill. |
| **Teleoperation** | A person drives the robot directly to record demonstrations. This project avoids it. |
| **Imitation learning** | Training a policy to copy demonstrations (observation → action pairs). |
| **Retargeting** | Converting human hand motion into robot gripper motion. |
| **Inverse kinematics (IK)** | Computing joint angles that put the gripper at a desired position and orientation. |
| **URDF** | A file describing the robot's links, joints and shapes. OpenArm's comes from `enactic/openarm_description`. |
| **AprilTag** | A printed square marker whose position the camera can measure precisely. Used to locate the table. |
| **MuJoCo** | A physics simulator. Used here to replay demonstrations and generate new ones. |
| **MimicGen / DexMimicGen** | Methods that turn a few demonstrations into many by moving objects and adapting the recorded motion, keeping only successful attempts in simulation. DexMimicGen extends this to two arms. |
| **Sim-to-real gap** | Differences between simulated and real data (physics, appearance) that make policies trained in simulation work worse on real robots. |
| **Phantom overlay** | Drawing the simulated robot over the real camera image when running the policy, so the robot looks identical in training and deployment. |
| **LeRobot** | Hugging Face's robotics library: dataset format, training, robot drivers. This repo builds on it. |
| **Quality gate** | An automatic check after a stage that stops the job with a reason if the data isn't good enough. |
| **Success check (inferred)** | A test of whether a simulated episode achieved the skill, e.g. "bottle held by the right gripper and lifted", derived from how the demonstration ended. |

---

## 3. Goals and constraints

The goals evolved through the discussion. In their final form:

| Goal / constraint | Source | Consequence |
|---|---|---|
| Easily add skills to a robot | initial question | skill = trained adapter + description, produced by one command |
| Robot: **OpenArm v1**, bimanual, 7-DoF arms, parallel grippers | user | URDF from `enactic/openarm_description`; LeRobot `bi_openarm_follower` names and units |
| Cross-embodiment (other two-arm robots) | user, later **deferred** | robot-specific code isolated in `video/robot.py`; canonical names; not implemented |
| **Learn from human video**, no teleop | user | Part 1 (video → robot), Part 2 (simulation multiplication) |
| **Normal RGB video**, fixed camera at head height looking down | user | AprilTag calibration; monocular depth via hand model + PnP |
| AprilTag on the table acceptable | user | table frame from the tag |
| **No coding per skill**: clip + name + description only | user | automatic task inference, object naming, perception |
| **No labelling of objects** ("the water bottle should not need labelling") | user | SAM 3 + SAM 3D Objects instead of object tags |
| Use DexMimicGen | user | reimplemented on MuJoCo (public repo lacks generation code) |
| An app uploads clips | user | `at-run --job job.json`, `report.json`, exit codes, quality gates |
| Cloud GPU (80 GB) | user | defaults sized for 80 GB; π0.5 LoRA default |
| Licences not a concern | user | research-licensed components allowed (noted in §15) |
| uv as package manager | user | `pyproject.toml`, `uv.lock` |

---

## 4. Discussion and decisions (chronological)

This section records the research discussion that shaped the repository, including options that were
explained and not built. Claims marked *(literature search)* come from a dedicated search that checked
each paper or repo page. Others reflect the assistant's knowledge at the time, and the discussion
flagged uncertainty where relevant.

### 4.1 Is adding skills to a robot a solved problem?

- **Engineering side (solved):** ROS 2 actions as skills, behaviour trees (BehaviorTree.CPP,
  py_trees, Nav2's plugin system), skill frameworks (SkiROS2, FlexBE, RAFCON), and a plugin registry
  with a fixed skill contract (inputs, preconditions, feedback, cancel, result).
- **Sequencing with LLMs (mostly solved):** skills exposed as tools (function calling, MCP), with
  SayCan and Code-as-Policies as the classic references.
- **Learning new physical skills from demonstration (not solved):** LeRobot (ACT, Diffusion
  Policy), VLAs (OpenVLA, π0, GR00T). Works with tens to hundreds of demos, but is brittle to changes.
- **Recommendation:** hybrid. A skill registry and interface, an optional LLM planner, and skills that
  are either coded or learned behind the same interface.

### 4.2 Cross-embodiment

- **Coded skills:** write against capabilities, not hardware. URDF + ros2_control + MoveIt 2 / Nav2,
  with a capability manifest per robot.
- **Learned skills:** Open X-Embodiment / RT-X, Octo, OpenVLA, π0, CrossFormer, HPT, GR00T. Zero-shot
  transfer to a new body mostly fails; fine-tuning with 50–200 demos is typical.
- **Practical advice:** use end-effector action spaces, normalise per robot, use robot-specific
  heads, and keep data in a standard format (LeRobot/RLDS).

### 4.3 Per-robot normalisation, canonical action space, LoRA for skills

- **Normalisation** explained: per-robot q01/q99 statistics map values to [-1, 1]. This fixes scale,
  not meaning.
- **Canonical bimanual action space proposed:** per arm, end-effector position + 6D rotation + gripper,
  in a shared frame at the shoulders' midpoint, with a fixed control rate and chunked actions.
- **How models handle different robots:** padding (π0), a fixed slot per meaning (RDT-1B), or
  robot-specific encoders and decoders (GR00T).
- **LoRA for skills:** OpenVLA, openpi (π0) and GR00T support LoRA-style fine-tuning. The recommended
  structure was **one LoRA per robot, with skills selected by language**, plus per-skill LoRAs only
  where needed. Composing adapters for robot × skill is unreliable.

### 4.4 Mixtures of LoRA experts, planners, residual policies, stacking

- **Mixture of LoRA experts** (MoLE, LoraHub; a router weights experts per input or layer):
  established for language models, research-stage for robot policies, and needs smooth switching to
  avoid jerky motion.
- **Hierarchy:** a VLM planner (System 2) over a fast VLA (System 1), as in Hi Robot, GR00T N1, Helix,
  π0.5 and RT-H.
- **Residual "patch" policies:** small learned corrections on top of a base policy, a good fit for
  per-robot quirks.
- **LLM planners:** SayCan, Inner Monologue, Code as Policies / ProgPrompt, LLM+P, VoxPoser,
  embodied-reasoning VLMs. The recipe is plan → act → verify → replan, with success detection as the
  weak link.
- **Adapter stacking**, ordered by reliability: sequential merge-then-train, adapters on disjoint
  layers, weighted merges (PEFT linear/TIES/DARE), learned routing. Portable robot × skill composition
  remains unsolved.

### 4.5 Per-layer routing (robot adapter early, skill adapter late)

- **Correction:** robot-specific information sits at *both ends* of a VLA (inputs: cameras and state;
  outputs: motor commands). The right layout is a **sandwich**: robot adapters at the input and output,
  skill adapters in the middle. This matches HPT (robot-specific stems and heads) and GR00T
  (robot-specific encoders and decoders).
- When robot and skill identities are known, **fixed adapter selection** beats a learned router.
- **To make adapters swappable:** train them jointly on a robot × skill grid and hold out a pair to
  measure transfer.

### 4.6 Use HPT/GR00T robot modules for embodiment and LoRAs for skills?

- **Yes in principle**, with caveats: the robot modules are small and handle *format*, not dynamics.
  GR00T's standard recipe also fine-tunes the action head. Skill LoRAs still need multi-robot training.
- **Model comparison:** GR00T best fits the plan, π0/π0.5 is a strong alternative, HPT is research
  code with weak language conditioning, and SmolVLA is a cheap prototype.
- **LoRA training time** estimates were given (consumer vs datacentre GPUs). Data collection dominates
  the effort.

### 4.7 Focus on skills; human video; video generators vs retarget-and-render

- Cross-embodiment was **deferred**.
- **Question:** should a video generator render the robot doing the task, from the human video?
- **Answer:** generated video gives pixels but no actions, and it doesn't respect physics or kinematics.
  Better: **retarget, then render**. Actions come from IK, and the robot is rendered or composited with
  the human removed. Video models serve only for realism and augmentation.

### 4.8 Literature search *(literature search)*

A dedicated search verified claims against papers and repos. Key findings:

- **Phantom** (Stanford, CoRL 2025; MIT code) is the only open, complete retarget-and-render
  pipeline. Its repo includes a bimanual mode from **Masquerade**.
  - Pipeline: HaMeR hand pose + depth ICP, SAM 2 masks, E2FGVI inpainting, MuJoCo rendering.
  - Gripper mapping: position from the thumb–index midpoint, orientation from that plane, aperture
    with the bottom 20% snapped to closed.
- **The overlay is critical:** Phantom reports 0% success without the rendered-robot overlay at both
  training and deployment.
- **Co-training beats human-only:**
  - Masquerade: 5–6× improvement, with 50 robot demos per task.
  - Physical Intelligence (π0.5 + human video): about 2× with an even human/robot mix, but only once
    robot pretraining is diverse enough.
  - EgoMimic: 1 h of human data was worth more than 1 h of robot data at the margin.
  - GR00T N1.7 is pretrained on 20k h of egocentric human video (EgoScale).
- **Video-generation realism** for human-to-robot translation was unproven (no public code, video
  metrics only).
- **Licensing traps:** MANO, WiLoR, HaWoR, ProPainter and EgoDex are non-commercial. Licences were
  later declared not a concern.
- Relative end-effector actions plus a gripper per arm is the emerging standard representation.

### 4.9 Mixture of LoRAs worth it? Which base model?

- **Mixture of LoRAs: not yet.** Start with one language-conditioned policy, move to per-skill LoRAs
  if skills interfere, and only then consider learned routing.
- **Base model:** GR00T for the long-term cross-embodiment plan, π0.5 as the strongest alternative,
  SmolVLA as a cheap prototype, HPT not recommended. Model choice is reversible if the data is in
  LeRobot format.

### 4.10 No teleop at all

- **Human-only results exist:** Phantom (up to 92%, single arm, simple tasks) and EgoZero. Bimanual
  human-only learning is largely untested.
- **Ways to get robot data without teleop:**
  - handheld gripper copies (UMI)
  - replay-and-filter on the real robot
  - autonomous self-improvement with success detection
  - simulation filtering
  - kinesthetic teaching

### 4.11 Video-only + Phantom + LoRA: valid?

Valid, with requirements:
- overlay at deployment
- new action input/output layers alongside the LoRA
- synthetic robot state from IK
- head camera only (no wrist view in human video)
- action cleaning
- language labels per episode
- staged validation: look at the data, replay open-loop on the robot, train a cheap baseline, then the VLA

### 4.12 Is Phantom the best? EgoDex and bottle-cap unscrewing

- **Phantom** is the best *open* option. Alternatives: HumanEgo (bimanual Trossen; found the overlay
  didn't help its token-based policy), and newer systems without code (HuRo, Ego2Robot).
- **EgoDex** (Apple Vision Pro, full hand-joint poses) helps avoid noisy hand pose estimation, but its
  head-mounted, moving viewpoint differs from the robot's, and its licence is CC-BY-NC-ND.
- **Unscrewing caps** is a poor fit for parallel grippers: humans roll the cap with their fingers, so a
  robot needs grip–rotate–release–regrip cycles.
  - Option A: start with an easier task.
  - Option B: a learned approach plus a scripted twist primitive.

### 4.13 Teleop + MimicGen + LeRobot + LoRA + SmolVLA

- **MimicGen needs simulation**: object poses per frame, subtask segmentation and replay to keep
  successes. That implies a digital twin and a sim-to-real gap.
- Use DexMimicGen or Isaac Lab Mimic for two arms.
- **SmolVLA** was pretrained on single-arm SO-100 data only.
- Screw threads are hard to simulate.
- **Revised plan:** sim twin → pick-and-place pilot → real teleop co-training → compare real-only,
  sim-only and mixed data.

### 4.14 Training a SmolVLA adapter; LoRA vs newer methods

- LeRobot's PEFT integration was verified from its docs: `--peft.method_type=LORA`, `--peft.r`,
  `--peft.lora_alpha`, default targets (the action expert's q/v plus projections), and a learning rate
  about 10× the full fine-tune rate.
- **Alternatives** (DoRA, rsLoRA, PiSSA, LoRA+, QLoRA, VeRA/IA³): small gains at best; LeRobot only
  exposes LoRA. For a 450M model, full fine-tuning the action expert per skill is also viable.
- **Advice:** compare default fine-tune, LoRA r=64 and LoRA r=16; spend effort on data.

### 4.15 Build: training pipeline (commit `9447521`)

MimicGen data → LeRobot → SmolVLA LoRA → offline evaluation. Details in §6.13–6.15 and §8.

### 4.16 Part 1: replace Phantom? Part 2: does Part 1 feed DexMimicGen?

- **Problem identified with "video → Phantom → DexMimicGen":** DexMimicGen needs simulated source demos
  (a sim environment, full simulator states, per-frame object poses, subtask annotations and successful
  replays). Video gives none of these directly, so a **real-to-sim stage** is required.
- Phantom needs RGB-D and supports only five built-in robots. The user has normal RGB video, a cloud
  GPU and the OpenArm URDF.
- **Decision:** build a Phantom-style RGB pipeline from components:
  - WiLoR/HaMeR (GPU) or MediaPipe (CPU) hand tracking
  - PnP lifting with calibrated intrinsics
  - AprilTag table frame
  - Phantom's gripper mapping
  - IK with mink on the URDF
  - export to robomimic-style HDF5 so both LeRobot and DexMimicGen-style generation can consume it
- **Explained simply:** Part 1 = video → robot movements. Part 2 = simulation multiplication (sim scene,
  object poses, replay with correction, generation).
- **Task for the pilot:** passing an object from one hand to the other.

### 4.17 Part 2 implementation route

- The **public DexMimicGen repository contains environments and datasets but no generation code**.
  The DexMimicGen-style generator lives in **Isaac Lab Mimic** (Linux, Isaac Sim, RTX GPU).
- **User's choice:** reimplement the method in MuJoCo (alternatives: Isaac Lab Mimic, or
  robosuite + the original single-arm MimicGen).

### 4.18 Fixing the sim-to-real image gap

Three measures:
- composite onto a real background
- the Phantom overlay at deployment via a LeRobot plugin
- appearance randomisation

Plus an aspect-ratio fix.

### 4.19 "Can it do everything now?"

Gaps identified at the time:
- egocentric video
- one hard-coded task
- the base VLA has no prior for OpenArm (the adapter teaches, rather than improves, the skill)
- nothing tested on real data
- no closed-loop evaluation
- no single command

Data per clip was explained: configurable; 244 frames per bottle demo; about 2.7 MB in HDF5 and about
0.6 MB in LeRobot.

### 4.20 No coding, no labelling, any reasonable skill

The user's camera is fixed at head height. They required clip + descriptor + name only, and no labels
(e.g. a water bottle).

- **Automatic task inference** was designed: phases from grasp/release events, references from the
  nearest object, success from the final arrangement, and an LLM cross-check of the description.
- **"Why not ignore the objects?"** Then there's nothing to adapt to, no way to filter failures, and
  images contradict actions.
- **Options:**
  - A: Phantom overlay on real video, one demo per clip, no object model.
  - B: simulation with simple shapes.
  - C: both mixed.
- **SAM 3D Objects** was introduced as the better B: a textured reconstructed shape.
- **The user chose B with SAM 3D Objects.**

### 4.21 Follow-ups

- **Single colour:** MuJoCo can't show vertex colours, and SAM's wrapper disables texture baking.
- **Skills not handled:** non-grasp, identical objects, hidden objects, transparent objects, too
  big/heavy/precise, articulated, deformable, liquids.
- **Is texture baking worth it?** Yes, layered, with real-frame projection first.
- **One-command wrapper:** built (`at-run`).
- **Before/after comparison?** Not meaningful for a new embodiment (the base can't drive OpenArm).
  Compare versions, in closed-loop simulation. Judged worth building, and built.
- **GR00T added:** LeRobot's GR00T has no wired LoRA, so the projector and action head are fine-tuned.
- **π0.5 added**, after asking: LoRA default, 80 GB GPU, π0.5 as the default policy.

### 4.22 Plan evolution at a glance

| Stage of discussion | Plan | Outcome |
|---|---|---|
| Early | Skill registry + optional LLM planner + learned skills | Context only |
| Cross-embodiment | Robot adapters at both ends, skill LoRAs in the middle | Deferred |
| Human data | Phantom → LoRA, co-train with teleop | Replaced: user wants no teleop |
| Teleop + MimicGen | Sim twin, DexMimicGen/Isaac Lab Mimic, SmolVLA | Training stage built first (MimicGen example data) |
| Video → Phantom → DexMimicGen | Real-to-sim stage needed | **Built** as Part 1 (Phantom-style RGB) + Part 2 (MuJoCo DexMimicGen) |
| Sim-to-real | Real background + overlay + randomisation | **Built** |
| No code / no labels | Task inference + SAM 3 + SAM 3D Objects | **Built** (`at-skill`) |
| Appearance | Layered texture baking | **Built** |
| Operations | One command, closed-loop evaluation, versioning | **Built** (`at-run`, `at-sim-eval`) |
| Policies | SmolVLA → + GR00T → + π0.5 (default) | **Built** |

---

## 5. System overview

```
                        ┌──────────── setup folder (once per camera placement) ────────────┐
                        │ tag.mp4 → camera.json (intrinsics + table pose)                    │
                        │ empty.mp4 → background.png        checkerboard.mp4 → intrinsics    │
                        └───────────────────────────────────────────────────────────────────┘
clips + name + description
   │
   ├─ 1. video        hands (WiLoR/MediaPipe) → PnP → table frame → gripper pose/opening → OpenArm IK
   │                  → demos.hdf5 (joint trajectories, gripper poses, frames) + check videos
   ├─ 2. objects      Claude (name, shape hint, mass) | heuristic fallback
   ├─ 3. perception   SAM 3 mask → SAM 3D Objects mesh → fit on table → texture bake → CoACD → scene JSON
   ├─ 4. replay       MuJoCo physics replay, grasp correction → keep successes → infer task spec
   ├─ 5. generation   DexMimicGen-style: move objects, transform phases, execute, keep successes
   ├─ 6. validation   replay every saved demo from its actions
   ├─ 7. dataset      composite images (real background) → LeRobot dataset local/<skill>
   ├─ 8. training     π0.5 LoRA (default) | SmolVLA LoRA | GR00T → outputs/train/<skill>/vN
   ├─ 9. sim eval     closed-loop success vs previous version → promote (latest.txt) or not
   └─ 10. offline eval held-out action error
deployment: lerobot-rollout --robot.type=bi_openarm_overlay (simulated robot drawn over the real one)
```

**Commands.** `at-run` (everything) → `at-skill` (stages 1–10) → `at-video2robot`, `at-datagen`,
`at-convert`, `at-train`, `at-sim-eval`, `at-eval`. Helpers: `at-calibrate`, `at-fetch-robot`,
`at-synthetic-demo`, `at-download`.

---

## 6. Method

### 6.1 Calibration and frames (`video/camera.py`, `video/calibrate.py`)

- **Intrinsics:** OpenCV checkerboard calibration, or an approximate pinhole from a horizontal field
  of view.
- **Table pose:** 36h11 AprilTag detected with OpenCV ArUco. Pose from `solvePnP` (IPPE_SQUARE),
  averaged over frames (chordal mean for rotation, median for translation).
- **Table frame:** origin at the tag centre, z up, x = the tag's up direction (pointing away from the
  robot), y = robot's left. The tag must lie with its top edge away from the robot.
- **MuJoCo camera:** derived from the calibration (OpenCV axes → MuJoCo's −z forward, y up).
  `Calibration.scaled()` handles different resolutions.

### 6.2 Hand tracking and 3D lifting (`video/hands.py`)

- **Backends:**
  - MediaPipe Hand Landmarker (CPU; tasks API, VIDEO mode; handedness flipped for non-mirrored
    images)
  - WiLoR via `wilor-mini` (GPU; installed by `scripts/install_wilor.sh` because its `setup.py`
    imports pip). The left-hand mirroring convention is resolved by choosing the variant whose
    projection falls inside the detection box.
- **Lifting:** each backend gives a metric hand-centred 3D shape and 2D keypoints. `solvePnP` (SQPNP +
  LM refine) with the **calibrated** intrinsics places the hand in the camera frame. This avoids WiLoR's
  assumed focal length (5000 px scaled).
- **Rejection rule:** keep the most confident detection per side per frame; reject detections with
  reprojection error above 15 px.

### 6.3 Retargeting (`video/retarget.py`)

Phantom-style mapping per frame:

| Quantity | Definition |
|---|---|
| grasp point | thumb-tip / index-tip midpoint |
| closing axis (TCP y) | thumb → index |
| approach axis (TCP z) | knuckles → grasp point, made orthogonal to the closing axis |
| opening | thumb–index distance normalised between its 20th and 95th percentiles |

Post-processing:
- gap filling (≤ 0.3 s), Savitzky–Golay smoothing, table clearance
- sign continuity of the closing axis
- **hands whose pinch width varies by less than 1.5 cm are treated as never grasping** (§8)

### 6.4 OpenArm model and IK (`video/robot.py`, `video/ik.py`)

- **Model:** `openarm_description` v1.0 example URDF, patched for MuJoCo (STL collision meshes used as
  visuals because `.dae` is unsupported; `fusestatic=false` keeps the TCP frames).
- **Placement:** the robot root is placed in the table frame (`base_xyz` default (−0.30, 0, −0.45);
  measure yours).
- **Joint limits:** intersection of the URDF and LeRobot's OpenArm safe limits.
- **Home pose:** forward-reaching. The right arm mirrors the left with signs (−,−,−,+,−,−,−).
- **Workspace analysis** (sampling within LeRobot limits):
  - reach about 0.6 m from the shoulder
  - downward-pointing gripper rare (0.5% of samples), forward-pointing common (11%)
- **IK:** mink differential IK with frame tasks for both TCPs (position cost 1, orientation 0.3), a
  posture task, configuration limits, and **collision avoidance keeping links 3–7 and grippers ≥ 1 cm
  above the table**. Each target picks the closer of the two equivalent parallel-gripper orientations
  (rotated 180° about the approach axis).

### 6.5 Part 1 output (`video/pipeline.py`)

`demos.hdf5` per clip:
- `obs/front_image` (320 px wide, camera aspect ratio)
- `obs/joint_pos` (16 values: LeRobot `bi_openarm_follower` names, degrees, gripper motor degrees)
- end-effector poses and gripper openings
- optional object tag poses
- `actions` (next-step joint targets)

Also written: `<clip>_check.mp4` (original with hands and targets next to the MuJoCo render from the
calibrated viewpoint, with IK error), `<clip>_tracks.npz` and `summary.json`.

### 6.6 Object perception (`perception/`)

1. **Naming** (`objects.py`): Claude Opus 5.5 with structured JSON output (name, role, shape hint, mass)
   from the skill name, description and first frame, using server-side refusal fallback. The offline
   fallback takes noun phrases after "the/a/an".
2. **Segmentation** (`segment.py`): SAM 3 via `transformers` (`facebook/sam3`, gated). Highest-scoring
   instance per name. `MaskDirSegmenter` for tests and manual fixes.
3. **Reconstruction** (`reconstruct.py`, `scripts/sam3d_worker.py`): SAM 3D Objects in its own conda
   environment, run as a subprocess. Image + mask in, GLB mesh + model pose out. It tries the pipeline's
   own texture baking, then falls back to vertex colours. **Only the shape is used.**
4. **Fitting** (`fit.py`):
   - candidate resting orientations from stable poses of the convex hull, **one per distinct "up"
     direction**
   - silhouette rendered from ~5,000 surface samples with morphological closing
   - Nelder–Mead over x, y, yaw and log-scale, after a coarse grid
   - IoU against the mask, with the exact triangle silhouette used for QA
   - **fallbacks:** box (x, y, z sizes) and round cylinder (diameter, height), preferring the
     description's shape hint when IoUs are within 0.05
5. **QA:** mesh accepted at IoU ≥ 0.80, shapes at ≥ 0.65; sizes must be 1–60 cm.
6. **Texture** (`texture.py`): xatlas UV unwrap. Each texel takes:
   - the real frame pixel, if visible (depth test against the object itself, within 1 cm, and inside
     the eroded mask)
   - else the reconstruction's colours
   - else the median colour

   Seams are bled outward.
7. **Assets** (`assets.py`): textured OBJ + PNG, CoACD convex collision parts, mass from Claude or
   400 kg/m³ × hull volume.

Assets are built once per object per skill and reused across clips. Per clip, a scene file is written
(`scenes/<clip>.json`) together with an overlay image.

### 6.7 Simulation (`datagen/env.py`)

- **`SceneEnv`:** OpenArm with gravity compensation, joint position actuators (kp 400 / kv 40) and
  finger position actuators (kp 1500, ±60 N). The finger mimic is rebuilt as an equality constraint.
- **World:** collidable table; objects as box/cylinder/mesh, with textured materials that are partly
  self-lit (emission 0.6).
- **Contacts:** robot links collide with the table and objects but not with each other.
- **Timing:** 30 Hz control × 16 substeps.
- **State:** `mjSTATE_INTEGRATION`, for exact action replay.
- **Grasp detection:** both fingers of an arm contacting an object.

### 6.8 Replay with grasp correction (`datagen/source.py`)

- Objects are placed from the scene file (or from the bar's tag, for older data). The arms start where
  the clip starts.
- Gripper commands are binarised (open until close, closed until open).
- **Grasp correction:** over the second before each close, the target is shifted so the fingers
  straddle the nearest object's centre along the closing axis and sit ≥ 1.5 cm inside its bounding
  box. The shift is held until release, then faded out.
- **Keep criteria:** every grasp holds 0.4 s after closing, every release lets go (not grasped
  0.4 s after opening), and objects stay on the table.

### 6.9 Task inference (`datagen/task.py`)

From each kept replay:
- **Phases** end at each gripper event (either arm), plus 0.3 s settle.
- **Closing arm:** reference = the object nearest its gripper.
- **Opening arm:**
  - handover (the other hand still holds the object) → the released object
  - else the object the released one lands on/in within 15 cm
  - else none (replay as recorded)
- **Coordination:** an arm holding an object that another arm references also references it (shared
  transform).
- **Success:** final holders (+ lifted); placed objects within 4 cm of their final position relative to
  what they were placed on; untouched objects stay upright (tilt < 25°).
- **Structure signature:** event order + references. All kept clips must match the first.

### 6.10 Generation (`datagen/generate.py`)

- Pick a source clip. Sample each object's pose around its source pose (±4 cm, ±5 cm, ±20° yaw;
  footprints non-overlapping).
- **Per phase:** transform the source's commanded gripper poses by (object now) · (source object at phase
  start)⁻¹ for referenced arms, or replay for unreferenced ones. **An arm that keeps the same reference
  keeps its transform.** Interpolate 8 steps between phases, hold 10 steps at the end.
- Execute through IK + physics. Keep the episode if the inferred success check passes.
- **Parallel workers** write separate files, which are merged.

### 6.11 Validation

Replays each saved demo's recorded joint actions from its first state and checks success again
(deterministic MuJoCo with the full integration state).

### 6.12 Images (`render/composite.py`, `datagen/appearance.py`, `render/overlay.py`)

- **`CameraWarp`:** MuJoCo renders a slightly wider pinhole view. Each output pixel is undistorted with
  the real intrinsics and distortion and resampled from the render. This handles the principal point
  and lens distortion.
- **`Compositor`:** renders RGB + segmentation and pastes robot/object pixels onto the background photo.
  Optional shadow factor (render with vs without robot and objects, shadows on).
- **Randomisation per episode:**
  - light direction and intensity
  - object colour jitter (textured objects get a mild ±12% tint)
  - image brightness and contrast ±15%
  - camera jitter (~0.5 cm, 0.5°)
- **Aspect ratio:** images keep the camera's aspect ratio (default width 320 in tools, 512 in
  `at-skill`).
- **Deployment overlay:** the same compositor renders the robot at measured joint angles over the live
  frame (objects excluded).

### 6.13 Dataset conversion (`convert.py`)

robomimic/MimicGen HDF5 → LeRobot v3 dataset:
- camera and state key mapping
- names from HDF5 attributes (LeRobot motor names)
- task text from attributes
- streaming video encoding (about 30× faster than per-episode encoding on Windows)
- optional image storage instead of video
- quantile statistics (`q01`/`q99`, needed by π0.5) are computed by LeRobot

### 6.14 Training (`train.py`) and versions

| | π0.5 (default in at-run) | SmolVLA | GR00T N1.7 |
|---|---|---|---|
| Base | `lerobot/pi05_base` | `lerobot/smolvla_base` | `nvidia/GR00T-N1.7-3B` |
| Method | LoRA r=64 (LeRobot π0.5 targets), bf16, lr 2.5e-4→2.5e-5 | LoRA r=64, lr 1e-3→1e-4, features from dataset | projector + action head, frozen backbone, `new_embodiment`, chunk 16, relative arm actions, bf16, image augmentation |
| Batch default | 64 | 32 | 64 |
| Setup | gated PaliGemma tokenizer | — | `uv sync --extra groot` |

- **Versions:** `outputs/train/<skill>/vN`, with `latest.txt` naming the promoted version.
- **Length and checkpoints:** a fixed number of steps (`--steps`, default 20,000); no early stopping.
  One checkpoint at the end unless `--save-freq N` (passed from `at-run`/`at-skill` to `at-train`),
  which also saves every N steps and computes the validation loss at each. The best of these is chosen
  in simulation (§6.15) and its step written to `vN/best.txt`; every tool that takes a version folder
  (`at-sim-eval`, `at-eval`) uses that checkpoint, else `checkpoints/last`. Rule of thumb for steps:
  about 5–10 passes over the data, i.e. steps ≈ passes × n × frames per demo ÷ batch size (frames ≈
  clip length × 30 Hz). E.g. n = 1000: ~30k steps for the 7.5 s stacking clip, ~50k for the 13.4 s
  scrubbing clip. More passes mostly memorise the few source clips.
- **Logging:** `--wandb` (`at-train`, `at-skill`, `at-run`; `"wandb": true` in a job) turns on LeRobot's
  Weights & Biases logging: project `lerobot`, run `<skill>_v<N>`, training loss / grad norm / lr every
  `--log-freq` (100) steps and validation loss at each checkpoint. One-time setup: `uv run wandb login`
  (or `WANDB_API_KEY`). Simulated success is not logged to W&B. Off by default; behind TLS interception
  the upload may fail with certificate errors.
- **Platform workaround:** on Windows, Hub ids for path-typed fields are downloaded first and passed as
  local paths.

### 6.15 Evaluation (`evaluate.py`, `sim_eval.py`)

- **Offline:** held-out episodes through the deployment inference path. Action MSE/MAE per dimension
  vs an always-predict-the-mean baseline.
- **Closed-loop simulation:**
  - 50 episodes on unseen object layouts (×1.25 wider placement, different seeds), identical for every
    checkpoint compared
  - the composited image and joint state go to the policy, and its actions step the physics
  - success = the inferred check holds for 5 steps
  - limit: 1.5 × the longest source demo
- **Promotion:** new success ≥ 20% (`--min-sim-success`) and no more than 5 points below the previous
  version (`--max-regression`).
- **Best checkpoint:** with `--save-freq`, `at-skill` evaluates every saved checkpoint of the new version
  (found automatically, `sim_eval.list_checkpoints`) on the same layouts, and picks the highest success,
  earliest on a tie (`pick_best`). The winner goes to `vN/best.txt`, becomes the adapter in
  `report.json`, and its rate is the one used for promotion. Simulated success is preferred over
  validation loss, which keeps falling after success plateaus. Costs: `sim_episodes` per checkpoint;
  the winner's rate is slightly optimistic (selected on the same episodes). `at-sim-eval
  --all-checkpoints <vN>` re-ranks them on demand (report only).

### 6.16 Deployment (`plugins/lerobot_robot_openarm_overlay`)

- A LeRobot plugin package (discovered by name prefix) registering `bi_openarm_overlay`: the
  `bi_openarm_follower` driver plus `RobotOverlay` on one camera.
- Config: `overlay_calib`, `overlay_camera`, `overlay_base_xyz`, `overlay_render_width`,
  `overlay_shadows`.
- Requires LeRobot's `damiao` extra (`python-can`).

### 6.17 Orchestration (`skill.py`, `run.py`)

- **`at-skill`:** runs stages with gates and writes `report.json` (stage status, numbers, inferred task,
  reasons). Exit 0 = done, 1 = gate, 2 = error.
- **`at-run`:**
  - prepares the setup folder (cached; redone if clips or `setup.json` change)
  - accepts a job JSON or CLI arguments, passing unknown flags through to `at-skill`
  - appends offline evaluation and prints a summary

### 6.18 Cloud runs on Modal (`modal_app.py`)

- **Image** (`nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04` + Python 3.12), built in cached layers:
  1. apt (FFmpeg for torchcodec; EGL/OSMesa/GL for MuJoCo and OpenCV), rclone
  2. Miniforge + SAM 3D Objects' `environments/default.yml` (inline, so editing the script below
     doesn't redo it)
  3. `scripts/modal_image/install_sam3d.sh code | packages | finish`: its pip extras; PyTorch3D,
     flash-attn and gsplat compiled for sm_90 without a GPU (`FORCE_CUDA=1`,
     `TORCH_CUDA_ARCH_LIST=9.0`, `--no-build-isolation`); hydra patch; `checkpoints/hf` symlinked to
     the cache volume
  4. a step deleting the global pip config that points at `pypi.ngc.nvidia.com`
  5. `uv sync --frozen --no-install-project` from `pyproject.toml`/`uv.lock` (the project runs from
     `PYTHONPATH=src`), then `scripts/install_wilor.sh` with `NO_SYNC=1`
  6. the code; `at-fetch-robot`; the MediaPipe model; an import check; WiLoR built once on the CPU
     (bakes its weights in, checks its MANO model loads)
- **Volumes:** `adapter-training-cache` at `/cache` (`HF_HOME`, `TORCH_HOME`, SAM 3D checkpoints;
  set at run time only, because a volume can't mount over a folder a build step wrote into) and
  `adapter-training-work` at `/work` (`inputs/`, plus `outputs/` and `lerobot/` via
  `AT_OUTPUTS_DIR` / `AT_LEROBOT_DIR`, new overrides in `common.py`).
- **`run_job`:** H200, 16 CPUs, 128 GB RAM, 24 h. Downloads SAM 3D's checkpoints on first use (only
  if the job uses SAM 3D), chooses EGL rendering if it works else OSMesa, runs `at-run` with the job
  (cwd `/work/inputs`), commits the work volume every 10 minutes and at the end, and optionally copies
  `outputs/train/**` plus reports to Google Drive with rclone (`RCLONE_CONFIG_GDRIVE_*` in the secret).
- **Local entrypoint:** uploads the job, its clips, its setup folder and any `masks:`/`mesh:` paths it
  names (same relative paths), calls `run_job`, prints the result; `--download` copies
  `outputs/train/<skill>/` back.
- **`check`:** `nvidia-smi`, rendering backend, a CUDA matmul in the uv environment, a PyTorch3D CUDA
  op in the SAM 3D environment, and SAM 3D discovery.
- **Secret** `adapter-training`: `HF_TOKEN` (required for the gated models); optional
  `ANTHROPIC_API_KEY`, `WANDB_API_KEY`, rclone values. `AT_MODAL_NO_SECRET=1` runs without it.
- **Local client on this machine:** `scripts/modal_cli.py` (truststore for TLS interception; UTF-8 and
  no legacy Windows console rendering, without which the CLI printed nothing when redirected).

---

## 7. Verification and results

All numbers come from the development machine (CPU only), on synthetic scenes unless stated
otherwise.

### 7.1 Training stage (MimicGen example data, `core/stack_d1`)

| Check | Result |
|---|---|
| Conversion, 50 demos | frames match the HDF5 (mean abs error ~1.8/255 from video compression), actions exact |
| Streaming vs per-episode video encoding | 0.5 s vs ~15 s per episode |
| SmolVLA LoRA smoke training | ~17 s/step on CPU; 2.97M trainable of 453M parameters; adapter saved |
| Offline evaluation (after 8 training samples) | MSE 0.27 vs mean baseline 0.22 (as expected for an untrained model) |

### 7.2 Part 1 (synthetic videos with known hand poses)

| Check | Result |
|---|---|
| Tag calibration, 6 cm tag | 25 mm / 1.3° camera pose error |
| Tag calibration, 16 cm tag | 8 mm / 0.4° |
| MediaPipe on a sample image | both hands found, correct handedness, plausible hand size and depth |
| IK tracking, handover | ~0.7–0.9 mm median, 1.1–1.2 mm p95, after the orientation-symmetry fix (3 cm before) |
| Joint limits | all within LeRobot's OpenArm limits |
| LeRobot conversion | driver motor names, 30 fps, task text |

### 7.3 Part 2: handover with a bar (object tag)

| Check | Result |
|---|---|
| Scripted grasp + handover in physics | succeeded on the first try |
| Source replay (3 clips) | 3/3 after the fixes in §8 |
| Generation success, hand-written phases | 77% (10/13), 80% (8/10), 68% (27/40) |
| Generation success, inferred phases + transform reuse | 67.5% (27/40) |
| Same with table-aware IK | **77.5% (31/40)** |
| Validation | 100% of saved demos (10/10, 8/8, 12/12, 18/18 across runs) |
| Rendering | 202 ms/frame with shadows → 59 ms without (integrated graphics) |
| Parallel generation | 8 demos in 104 s with 4 workers |

### 7.4 Images and deployment overlay

| Check | Result |
|---|---|
| Camera warp vs OpenCV projection (distorted, off-centre camera) | 0.32 px median, 0.69 px max |
| Pasted tag marker vs real tag (640 px) | 4.6 px (consistent with the synthetic calibration error) |
| Overlay vs training composite, same joints | < 0.25% of pixels differ |
| Overlay speed (320 px, CPU) | ~128 ms/frame |
| LeRobot plugin | discovered and constructs `BiOpenArmOverlay` |

### 7.5 Perception and textures (synthetic water bottle, 7 × 7 × 22 cm, labelled)

The stand-in "reconstruction" is the bottle mesh lying on its side, at 3.7× scale, offset.

| Check | Result |
|---|---|
| Mesh fit (3 clips) | IoU 0.978–0.98; size 6.8–6.9 × 21.4–21.6 cm; position within ~3 mm; ~3 s per fit |
| Box vs cylinder fallback from one view | 0.85 vs 0.88 IoU (box nearly as good, hence the shape hint) |
| Physics | stands stably; 2 convex parts; 0.29 kg estimated |
| Texture | render vs real frame: mean colour error **4.3/255** (53/255 before the fixes); 44% of texels from real pixels |

### 7.6 `at-skill` / `at-run` end to end (bottle, stand-ins for GPU models)

| Check | Result |
|---|---|
| Inferred task | right grasps water bottle → retreat; success: held by right, lifted |
| Replay | 3/3 clips (after two fixes) |
| Generation | 12 demos, 75% (untextured) / 71% (textured, 512 px) |
| Validation | 12/12 |
| `at-run` with training + evaluation | every stage `done`; setup cached and invalidated correctly |
| Versioning + simulated comparison | v1 promoted; v2 trained, evaluated against v1 on identical layouts, promoted (0% each, after 1 training step) |
| Closed-loop evaluation speed (CPU) | ~55–70 s per adapter for 2 episodes, plus model loading |

### 7.7 Policies (configuration only)

- The GR00T and π0.5 commands are accepted by LeRobot's own config parser:
  - GrootConfig: frozen backbone, trained projector and head, relative actions with grippers excluded
  - PI05Config: bf16, quantile normalisation, LoRA r=64
- The `groot` extra installs on Windows.

---

## 8. Problems found and fixed

In order of discovery. Each was found by running the code, not by inspection.

| # | Problem | Cause | Fix |
|---|---|---|---|
| 1 | HTTPS failures for downloads (uv, Python, git) | TLS interception on the dev machine | `native-tls` for uv, `truststore` injected at import, schannel for git clones |
| 2 | LeRobot "repo id" error on Windows | `--policy.path` parsed as a Path: `lerobot\smolvla_base` | download the base model and pass a local path |
| 3 | Training validation pass would take ~1 h on CPU | LeRobot evaluates all held-out frames | `--max-eval-samples` cap |
| 4 | Images flipped in an assumed fix | MimicGen release images are already upright | flip made opt-in (`--flip`) |
| 5 | Video encoding 15 s/episode; batched encoding crashed | per-episode process spawning; LeRobot bug | streaming encoding by default |
| 6 | IK 10–36 cm errors | robot too far; LeRobot limits; poor seed | workspace analysis → base 30 cm behind the tag, forward home pose |
| 7 | Left arm 3 cm IK error | arbitrary choice between two equivalent gripper orientations | IK picks the closer of the two |
| 8 | Calibration test showed a 180° error | the test rotated the synthetic tag the wrong way | test fixed (library correct) |
| 9 | MuJoCo offscreen width error | 640 px default framebuffer | larger offscreen buffer |
| 10 | Training segfault | out of RAM while loading the model | documented: free RAM |
| 11 | Handover success check counted the wrong finger | `openarm_left_right_finger` matched "right" | finger geoms found via finger joints |
| 12 | Right gripper pinched the bar's edge | correction only kept the grasp point inside the object | centre along the closing axis, ≥ 1.5 cm margin |
| 13 | Fingers clipped the bar on approach | the human's pinch maps to ~85% open, < 1 cm clearance | binary open/closed commands in simulation |
| 14 | Rendering bottleneck | shadows on integrated graphics | shadows off by default, `--shadows` flag |
| 15 | `fillPoly`-based silhouettes had holes | even-odd filling of overlapping polygons | fill triangles individually; later point-splat silhouettes |
| 16 | `at-run` mistook `--n` for `--name` | argparse abbreviation matching | `allow_abbrev=False` |
| 17 | Idle hand made phantom grasps | per-hand normalisation stretched tracking noise | treat as open if pinch range < 1.5 cm |
| 18 | Elbow through the table | gripper-pose-only IK | collision avoidance with the table |
| 19 | Arms twitched at phase boundaries | re-anchoring on an object held by both hands | keep the transform while the reference is unchanged |
| 20 | Upright bottle fitted lying down | the most probable stable poses were all side-lying | one candidate per distinct "up" direction |
| 21 | Mesh simplification erased thin parts | quadric decimation on non-watertight meshes | use only if the bounding box is preserved |
| 22 | Fitting took 87 s | triangle rasterisation of a 30k-face mesh × 8 candidates | point-splat silhouettes (3 s) |
| 23 | Point silhouettes biased sizes by ~1 px | dilation without erosion | morphological closing |
| 24 | Texture: few real pixels, smeared | strict face-id visibility; decimation | depth-test visibility, no lossy decimation |
| 25 | Textured objects too dark | real shading lit twice | partly self-lit material |
| 26 | Mesh file not found | URDF `strippath` applied to object meshes | `spec.strippath = False` |
| 27 | `.dae` meshes unsupported in MuJoCo | format | STL collision meshes as visuals |
| 28 | `wilor-mini` won't build under uv | its `setup.py` imports pip | install script without build isolation |
| 29 | `pipeline.run` returned an undefined name | refactor patch didn't apply | fixed |
| 30 | Noun-phrase heuristic missed "bowl" | regex consumed the rest of the sentence | word-by-word scan |
| 31 | Validation could diverge from the recording | state lacked warm-start/controls | `mjSTATE_INTEGRATION` |
| 32 | Squeezed 4:3 images vs a 16:9 live camera | fixed 320×240 export | camera-aspect output size |
| 33 | Modal CLI: "Could not connect" / no output | TLS interception; rich's legacy Windows console writer | `scripts/modal_cli.py` |
| 34 | SAM 3D env build aborted after creating the env | `set -u` in the install script; conda activation reads unset variables | no `-u` |
| 35 | SAM 3D build crawled (5 retries per package) | its docs' extra index `pypi.ngc.nvidia.com` no longer resolves; a dependency also writes it into a global pip config | index dropped (script and QUICKSTART); config deleted |
| 36 | torchcodec failed to load in the image | `wilor-mini` caps torch at 2.5 and downgraded torch 2.11 → 2.5 (also adds a second OpenCV) | `install_wilor.sh` installs it with uv overrides keeping torch/torchvision, dropping opencv-python |
| 37 | WiLoR failed to load its detector | ultralytics 8.1 checkpoint vs torch ≥ 2.6 `weights_only=True` | `weights_only=False` only while WiLoR loads its own weights (`video/hands.py`) |
| 38 | Modal container wouldn't start | build steps wrote into `/cache` (`HF_HOME`), so the volume couldn't mount | cache paths set at run time only |

---

## 9. Alternatives considered and rejected

| Alternative | Why not (or not yet) |
|---|---|
| Training directly on human video without an overlay | Phantom: 0% without an overlay; appearance gap |
| Free video generation of the robot from human video | no actions, no physics or kinematics guarantees, unproven |
| Phantom as-is | needs RGB-D, five built-in robots, undocumented outputs; replaced by an RGB Phantom-style pipeline |
| EgoDex for the first skill | moving head-mounted viewpoint, CC-BY-NC-ND, unclear task availability |
| Unscrewing caps first | finger rolling vs parallel gripper, thread simulation |
| Teleop + MimicGen | user requires no teleop |
| Isaac Lab Mimic | Linux + Isaac Sim, untestable here; the MuJoCo reimplementation chosen instead |
| robosuite + original MimicGen | most work, single-arm, least maintained |
| Object tags | user requires no labelling (tags remain supported for older data) |
| Ignoring objects in simulation | nothing to adapt to, no failure filtering, inconsistent images |
| Simple shapes only (option B without SAM 3D) | kept as the automatic fallback |
| Phantom overlay on real clips (option A) | not built; one demo per clip; a possible future co-training source |
| Mixture of LoRA experts / learned routing | research-stage; fixed selection or a single adapter suffices for now |
| LoRA variants (DoRA, PiSSA, …) | small gains, not exposed by LeRobot |
| "Base model vs adapter" comparison | the base can't control OpenArm; compare versions instead |
| GR00T LoRA | LeRobot's GR00T LoRA fields are deprecated and unused |
| Egocentric (worn) camera | the user's camera is fixed |

---

## 10. Limitations and risks

### Data and perception

- **Monocular hand depth** is approximate (Phantom's RGB mode reports 3–4 cm). Mitigated by calibration
  and the hand-size prior; real accuracy is unmeasured.
- **Top-down views** show mostly the backs of the hands, so fingertip occlusion is likely.
- **One perception frame** (frame 0): objects must be fully visible at the start.
- **Transparent or shiny objects** are hard to segment and reconstruct.
- **Several identical objects:** only the best-scoring instance is kept.
- **Hidden sides** of objects are the reconstruction's guess.

### Skills

- **Unsupported skill types:** no grasp (push, wipe, press); articulated objects (risk of *silent*
  failure: a drawer modelled as a rigid box); deformables; liquids; precise insertion or screwing.
- **Success checks are end-state only.** Process skills (e.g. scrubbing) aren't captured.
- **Gripper:** OpenArm opens to about 8.8 cm; the right arm reaches the handover point only with a
  mostly forward-pointing gripper.

### Simulation and transfer

- **Approximate physics:** no gripper-gripper collisions; friction and mass are estimates.
- **Sim-to-real:** the object is rendered in training but real at deployment; cables and mounts are
  missing from the overlay; shadows only with `--shadows`.
- **Simulated success is optimistic** (trained and tested in the same simulator).

### Robot

- **Unverified conventions:** motor zero and direction vs URDF; gripper 0° = closed, −65° = open;
  `base_xyz`.

### Operations

- **GR00T checkpoints** are several GB per version.
- **CoACD** takes ~40 s per object on CPU.
- **Overlay latency:** ~128 ms/frame on CPU.

---

## 11. Untested components

| Component | Status |
|---|---|
| WiLoR tracker | written to the package source; untested (no GPU) |
| SAM 3 segmentation | written to the transformers API; untested (gated, GPU) |
| SAM 3D Objects worker (incl. texture baking) | written from the repo's code; untested (Linux, ≥ 32 GB GPU) |
| Claude object naming | one successful test call; normally run with `--no-claude` in tests |
| π0.5 training/inference | config accepted by LeRobot; not run |
| GR00T training/inference | config accepted by LeRobot; not run |
| Real videos | not processed |
| Real robot, `bi_openarm_overlay` with hardware | plugin constructs; never connected |
| "Not promoted" branch of the promotion gate | simple logic; not exercised end to end |
| Shadows in composites at scale | implemented; lightly tested |

---

## 12. The example videos in `examples/`

The user added two clips (commit `a056d6b`):

| File | Content | Format |
|---|---|---|
| `Task1_Stacking.mp4` | stacking blue blocks | 1920×1080, ~20 fps, 7.5 s |
| `Task2_Scrubbing.mp4` | scrubbing the table with a held block | 1920×1080, ~20 fps, 13.4 s |

Observations relevant to this pipeline:

- **Camera:** top-down and wide-angle with visible lens distortion. Use a **checkerboard calibration**
  rather than the field-of-view approximation.
- **No table AprilTag** is visible, so a separate `tag.mp4` with the tag on the table is needed (camera
  unchanged).
- **The workspace is marked with tape;** other people, laptops and clutter are in view. The background
  photo should come from an empty-workspace clip. The clutter itself is fine (only the robot and objects
  are composited).
- The demonstrator's arms enter from the top edge (the robot side), matching the recording guide.
- **Stacking** fits the supported skill types: grasp → place on another object → release. Its success
  check would be "block placed relative to the other block".
- **Scrubbing** involves holding the block, but its goal is the back-and-forth *motion*. The current
  end-state success check would reduce it to "block held/moved". Supporting it needs a process-based
  success definition (§13).
- The clips haven't been run through the pipeline.

---

## 13. Recommended next steps

1. **GPU machine bring-up:** install WiLoR, gated model access, the SAM 3D Objects environment. Run the
   synthetic CPU job, then one real clip with `--stop-after perception` and inspect the overlays.
2. **First real skill (stacking):**
   - record `tag.mp4`, `empty.mp4` and `checkerboard.mp4`
   - record 10–20 clips
   - run `at-run`; inspect `report.json`, check videos, `source_*.mp4`
3. **Robot conventions:** verify joint zero and direction, the gripper range and `base_xyz` on the real
   OpenArm before running any policy.
4. **Real-robot evaluation** of the first adapter against simulated success, to calibrate how
   optimistic the simulation is.
5. **Cheap extensions:**
   - best-visibility frame selection for perception
   - a safeguard against articulated-object silent failures
6. **Next skill types:**
   - contact-based phases for pushing/wiping
   - process-based success checks (e.g. scrubbing coverage)
   - multiple identical objects
7. **Data:** Phantom-style overlay of real clips (option A) as a co-training source alongside generated
   data.
8. **Later:** cross-embodiment (robot adapters at both ends), egocentric video, articulated objects.

---

## 14. Engineering reference

### 14.1 Repository layout

| Path | Lines | Purpose |
|---|---:|---|
| `src/adapter_training/__init__.py` | 7 | injects `truststore` (OS certificate store) |
| `common.py` | 30 | paths, device, video backend, dataset roots |
| `download.py` | 43 | MimicGen dataset download (`at-download`) |
| `convert.py` | 172 | robomimic HDF5 → LeRobot (`at-convert`) |
| `tasks.py` | 33 | language instructions for MimicGen/robomimic envs |
| `train.py` | 230 | lerobot-train wrapper: SmolVLA / π0.5 LoRA, GR00T (`at-train`) |
| `evaluate.py` | 157 | offline action-error evaluation (`at-eval`) |
| `sim_eval.py` | 182 | closed-loop simulated evaluation (`at-sim-eval`) |
| `skill.py` | 263 | staged pipeline with gates and reports (`at-skill`) |
| `run.py` | 162 | one command incl. setup caching (`at-run`) |
| `video/robot.py` | 248 | OpenArm config, URDF patching, MuJoCo model (`at-fetch-robot`) |
| `video/ik.py` | 111 | bimanual mink IK with table avoidance |
| `video/camera.py` | 213 | calibration, AprilTag, table frame, object tags |
| `video/calibrate.py` | 94 | `at-calibrate intrinsics/table` |
| `video/hands.py` | 190 | MediaPipe / WiLoR trackers, PnP lifting, tracks I/O |
| `video/retarget.py` | 108 | hands → gripper targets |
| `video/pipeline.py` | 345 | Part 1 (`at-video2robot`) |
| `video/synthetic.py` | 320 | synthetic handover/bottle scenes (`at-synthetic-demo`) |
| `datagen/env.py` | 375 | `SceneEnv` physics |
| `datagen/dataset.py` | 186 | episode recording, HDF5 I/O, rendering |
| `datagen/source.py` | 176 | replay with grasp correction |
| `datagen/task.py` | 195 | task inference, success checks |
| `datagen/generate.py` | 149 | generation |
| `datagen/appearance.py` | 70 | background capture, randomisation |
| `datagen/cli.py` | 381 | `at-datagen background/prepare/generate/validate` |
| `perception/objects.py` | 130 | Claude / heuristic object naming |
| `perception/segment.py` | 61 | SAM 3, mask directory |
| `perception/reconstruct.py` | 59 | SAM 3D Objects subprocess, mesh file |
| `perception/fit.py` | 239 | silhouette fitting, primitives |
| `perception/texture.py` | 153 | layered texture baking |
| `perception/assets.py` | 90 | MuJoCo assets (texture, CoACD, mass) |
| `perception/perceive.py` | 119 | per-clip perception orchestration |
| `render/composite.py` | 159 | camera warp, compositor, appearance |
| `render/overlay.py` | 77 | deployment overlay |
| `plugins/lerobot_robot_openarm_overlay/` | 59 | LeRobot plugin `bi_openarm_overlay` |
| `scripts/sam3d_worker.py` | 73 | runs inside the SAM 3D environment |
| `scripts/install_wilor.sh` | 9 | WiLoR install |
| `jobs/*.json` | — | example job files |

### 14.2 Commands

| Command | Does |
|---|---|
| `at-run` | everything; `--job job.json` or `--setup/--name/--description/--videos`; extra flags go to `at-skill` |
| `at-skill` | stages 1–10 with gates; `--stop-after perception/prepare` |
| `at-calibrate intrinsics/table` | camera calibration |
| `at-video2robot` | Part 1 |
| `at-datagen background/prepare/generate/validate` | Part 2 |
| `at-convert` | HDF5 → LeRobot |
| `at-train` | training (`--policy smolvla/pi05/groot`) |
| `at-sim-eval` | closed-loop comparison of checkpoints |
| `at-eval` | offline action error |
| `at-synthetic-demo` | synthetic test scenes (`--scenario handover/bottle`) |
| `at-fetch-robot` | OpenArm description + MuJoCo model |
| `at-download` | MimicGen example data |

### 14.3 Key defaults

| Setting | Default |
|---|---|
| Control rate | 30 Hz (16 physics substeps) |
| Robot base in table frame | (−0.30, 0, −0.45) m |
| Hand-tracking reprojection limit | 15 px |
| Gap filling / smoothing | 0.3 s / Savitzky–Golay window 9 |
| Grasp "closed" / "open" thresholds | opening < 0.3 / > 0.6 |
| Never-grasping hand | pinch range < 1.5 cm |
| Grasp correction | 1 s ramp, 1.5 cm margin |
| Perception gates | mesh IoU ≥ 0.80, shape IoU ≥ 0.65, size 1–60 cm |
| Generation randomisation | ±4 cm x, ±5 cm y, ±20° yaw |
| Generation gates | ≥ 10% success, ≥ n/4 demos |
| Validation gate | ≥ 90% |
| Simulated evaluation | 50 episodes, ×1.25 wider placement, success held 5 steps |
| Promotion | ≥ 20% success, ≤ 5 points below previous |
| Training | 20,000 steps; checkpoint at the end only (`--save-freq`: every N steps, best chosen in simulation) |
| Images | camera aspect; 512 px wide in `at-skill` |

### 14.4 File formats

**`demos.hdf5` (Part 1) and `sources/generated.hdf5` (Part 2):** robomimic-style.
- `data/` attributes: `env_args` (JSON with `control_freq`), `task`, `robot`; for Part 2 also `scene`
  (JSON list of objects) and `task_spec` (JSON).
- Each `demo_i` has `actions` (T, 16), `obs/joint_pos` (T, 16) with `names`, `obs/front_image`,
  end-effector and object observations.
- Part 2 additionally has `states` and `datagen_info/{left,right}_{eef,target}_pose`,
  `datagen_info/object_pose/<name>` and `datagen_info/gripper_opening`.

**Scene file** (`part1/scenes/<clip>.json`):

```json
{"objects": [{"name": "water_bottle", "shape": "mesh", "size": [0.069, 0.069, 0.216], "mass": 0.29,
  "visual_mesh": ".../visual.obj", "texture": ".../texture.png", "collision_meshes": ["..."],
  "pose": [[...4x4...]]}],
 "qa": {"water_bottle": {"kind": "mesh", "iou": 0.978, "size_cm": [6.9, 6.9, 21.6], "ok": true}}}
```

**Task spec** (`datagen/task_spec.json`):

```json
{"phases": [{"name": "p0_right_close", "end_event": "right_close",
             "refs": {"left": null, "right": "water_bottle"}, "settle_s": 0.3},
            {"name": "p1_end", "end_event": null, "refs": {"left": null, "right": null}}],
 "success": {"water_bottle": {"holders": ["right"], "lifted": true}},
 "events": [[68, "right", "close"]]}
```

**`report.json`:** `stages.<name>` (status, numbers, reason), `result` (status, dataset, adapter,
promoted), `setup` (paths).

### 14.5 Platform notes

- **TLS-intercepting networks:** uv `native-tls = true`; `truststore.inject_into_ssl()` at import; git
  clones with `http.sslBackend=schannel` on Windows.
- **Windows paths:** Hub ids in path-typed LeRobot fields are mangled; base models are downloaded and
  passed as local paths.
- **Video decoding:** torchcodec needs FFmpeg shared libraries (often missing on Windows); PyAV is used
  there. The torchcodec traceback in logs is harmless.
- **Dependencies:** `opencv-contrib-python` (from MediaPipe) coexists with LeRobot's headless OpenCV.
- **Extras:** GR00T is an optional extra; a plain `uv sync` removes it again.
- **Memory:** training needs several GB of free RAM.

---

## 15. Licences of components

The user declared licences not a concern. Recorded for completeness:

| Component | Licence |
|---|---|
| This repo's dependencies: LeRobot, MuJoCo, mink, trimesh, xatlas | open source (Apache/MIT/BSD-style) |
| OpenArm description | Apache 2.0 |
| SmolVLA, π0.5 (openpi/LeRobot) | Apache 2.0 |
| π0.5 tokenizer (PaliGemma) | Google's gated licence |
| GR00T N1.7 weights | NVIDIA Open Model License |
| SAM 3, SAM 3D Objects | SAM License (gated) |
| WiLoR / HaMeR / MANO | research / non-commercial terms (MANO) |
| MediaPipe | Apache 2.0 |
| MimicGen / DexMimicGen code (reference only) | NVIDIA Source Code License; datasets CC-BY 4.0 |
| CoACD | MIT |

---

## 16. Commit history

| Commit | Summary |
|---|---|
| `9447521` | LoRA adapter training (MimicGen → LeRobot → SmolVLA) |
| `063f97d` | Human RGB video → OpenArm joint trajectories |
| `7450b71` | DexMimicGen-style generation for the OpenArm handover in MuJoCo |
| `9b6cd20` | Real-background composites and deployment overlay |
| `e1b1206` | `at-skill`: SAM 3D Objects perception and inferred tasks |
| `156c021` | Texture baking from the clip |
| `d518b04` | `at-run`: one command |
| `ab2b7b6` | Closed-loop simulated evaluation, versioning, promotion |
| `b3efde4` | GR00T N1.7 option |
| `2351099` | π0.5 LoRA option, default policy |
| `a056d6b` | Example videos (user) |

---

## 17. Glossary

| Term | Meaning |
|---|---|
| Action chunk | several future actions predicted at once (e.g. 50 steps) |
| Approach axis | the gripper's pointing direction (TCP z) |
| Closing axis | the direction the fingers close along (TCP y) |
| CoACD | approximate convex decomposition for collision shapes |
| Coordination subtask | a phase where two arms share one transform (DexMimicGen) |
| Embodiment | a specific robot body |
| Gripper opening | 0 = closed, 1 = fully open |
| IoU | intersection over union of two masks |
| Phase | a segment of a demo between gripper events |
| PnP | perspective-n-point: pose from 3D–2D correspondences |
| Promotion | marking a trained version as the one to deploy (`latest.txt`) |
| q01/q99 | 1st/99th percentile statistics used for normalisation |
| Reference object | the object an arm's motion is expressed relative to in a phase |
| Structure signature | a demo's event order + references; must match across clips |
| TCP | tool centre point, between the fingertips |

---

## 18. References

Verified during the literature search or used directly:

- LeRobot: https://github.com/huggingface/lerobot. Docs: SmolVLA
  https://huggingface.co/docs/lerobot/smolvla, PEFT https://huggingface.co/docs/lerobot/peft_training,
  π0.5 https://huggingface.co/docs/lerobot/pi05, GR00T https://huggingface.co/docs/lerobot/groot
- OpenArm description: https://github.com/enactic/openarm_description
- Phantom / Masquerade: https://github.com/MarionLepert/phantom, https://phantom-human-videos.github.io/,
  https://masquerade-robot.github.io/
- MimicGen: https://github.com/NVlabs/mimicgen. Datasets:
  https://huggingface.co/datasets/amandlek/mimicgen_datasets
- DexMimicGen: https://github.com/NVlabs/dexmimicgen
- Isaac Lab Mimic:
  https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html
- SAM 3D Objects: https://github.com/facebookresearch/sam-3d-objects
- SAM 3: https://huggingface.co/facebook/sam3 (via `transformers`)
- WiLoR-mini: https://github.com/warmshao/WiLoR-mini
- mink: https://github.com/kevinzakka/mink
- EgoMimic: https://egomimic.github.io/. EgoZero: https://egozero-robot.github.io/. HumanEgo:
  https://humanego-ai.github.io
- Physical Intelligence human-to-robot transfer: https://pi.website/research/human_to_robot
- EgoDex: https://github.com/apple/ml-egodex
- DreamGen: https://research.nvidia.com/labs/gear/dreamgen/
- Cosmos-Transfer2.5: https://github.com/nvidia-cosmos/cosmos-transfer2.5
