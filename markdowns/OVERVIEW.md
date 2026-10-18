# AdapterTraining: overview

A few-page summary. For step-by-step use see `QUICKSTART.md`; for everything (history, method details,
every result and fix) see `REPORT.md`.

## What it is

A pipeline that teaches a **two-arm OpenArm robot** new manipulation skills from **ordinary videos of
a person doing the task**. A user, for example through an app, provides:

- a few clips from a fixed camera
- a skill name
- a description (also used as the robot's instruction)

…and gets back a trained skill model for a vision-language-action (VLA) policy: a **π0.5 LoRA adapter**
by default, with SmolVLA and GR00T N1.7 as alternatives.

There's no teleoperation, no markers on objects and no per-skill code. One command does everything:

```bash
uv run at-run --job job.json
```

## How it works

```
setup clips ─► camera calibration (AprilTag) + empty-workspace photo         [once per camera]

skill clips ─► 1 video        hands → gripper motion → OpenArm joint angles
             ─► 2 objects      names from the description (Claude)
             ─► 3 perception   SAM 3 masks → SAM 3D Objects shapes → fitted on the table → textured
             ─► 4 replay       each clip replayed in physics (MuJoCo); task inferred automatically
             ─► 5 generation   hundreds of variations with objects moved (DexMimicGen method)
             ─► 6 validation   every demo re-checked from its recorded actions
             ─► 7 dataset      realistic images (robot + objects on the real background) → LeRobot
             ─► 8 training     π0.5 LoRA (default) | SmolVLA LoRA | GR00T
             ─► 9 evaluation   closed-loop success in simulation vs the previous version → promote
deployment  ─► lerobot-rollout with the simulated robot drawn over the real one
```

1. **Video → robot motion.** The camera's pose is measured once from an AprilTag on the table. Hands
   are tracked in 3D (WiLoR on GPU, MediaPipe on CPU) and placed in metres using the calibrated
   camera. Thumb and index finger map to the robot's two-finger gripper: midpoint → position, their
   plane → orientation, distance → opening. Inverse kinematics on the OpenArm URDF turns that into
   joint angles, keeping the arms above the table.

2. **Objects without labels.** Claude reads the description and first frame and lists the objects,
   with a rough shape and mass. SAM 3 finds each one in the image. SAM 3D Objects reconstructs its 3D
   shape, which is then fitted to the calibrated table: how it rests, where, which way it faces, how
   big. If the fit is poor, a box or cylinder is used instead. A texture is baked from the real video
   for the sides the camera saw, with the reconstruction's colours for the rest.

3. **The task, inferred.** Each clip is replayed in physics, with each grasp nudged onto the object
   (hand tracking is off by centimetres). From the replay:
   - **phases:** split at every grasp and release
   - **reference objects:** which object each arm's motion is relative to
   - **success check:** taken from how the demo ends, e.g. "bottle held by the right gripper and lifted"

   No code is written per skill.

4. **Many demos from few.** A reimplementation of DexMimicGen moves the objects to new positions,
   adapts each phase's motion to them (both arms together in handovers), runs it in physics, and keeps
   only successes. Every kept demo is replayed again from its actions to confirm it.

5. **Realistic images.** Training images show the simulated robot and objects composited onto a real
   photo of the workspace, through the real camera's lens model, with randomised lighting and colours.
   At deployment, a LeRobot plugin draws the same simulated robot over the live camera image (the
   "Phantom" trick), so the policy sees the same robot in training and on the real robot.

6. **Training and promotion.** Each training run is a new version. It's tested closed-loop in
   simulation on unseen object layouts against the previous version, and is promoted only if it's good
   enough and not worse. Training runs for a fixed number of steps; with `--save-freq` it also keeps
   intermediate checkpoints, all of which are tested in simulation, and the best one is used.

Every stage has a quality check. If one fails, the job stops with a plain-language reason in
`report.json` instead of training on bad data.

## Key design decisions

| Decision | Why |
|---|---|
| Learn from human video, not teleop | the user's requirement; human clips are cheap to collect |
| Retarget then render, not generate robot video | generated video has no actions and no physics guarantees |
| Phantom-style pipeline built from parts | Phantom needs RGB-D and supports only five robots; the user has RGB and OpenArm |
| Reimplement DexMimicGen on MuJoCo | the public DexMimicGen repo has no generation code; Isaac Lab Mimic needs Isaac Sim |
| Reconstruct objects (SAM 3D Objects), don't ignore them | generation needs objects to adapt to, check against and show in images |
| Use only the reconstructed *shape*; fit pose and size to the table | robust to the model's camera conventions; uses the calibration |
| Infer the task from the replay | no per-skill code; checked against how the demo ends |
| Real background + deployment overlay | closes most of the appearance gap between simulation and the real camera |
| Versioned adapters + simulated comparison | no base-model "before" exists for OpenArm; compare versions instead |
| π0.5 LoRA as the default | strong two-arm model, small adapters, fits the 80 GB GPU |

## What's verified, and what isn't

Everything below was verified on a development machine without an NVIDIA GPU, using synthetic
scenes with known ground truth.

**Verified:**
- **Skills inferred with no task-specific code:** a two-arm bar handover and a water-bottle pickup.
- **Generation:** 68–78% of attempts succeed; 100% of saved demos replay successfully from their
  actions.
- **Perception:** a bottle reconstruction given lying on its side at the wrong scale was fitted
  correctly (7 × 7 × 22 cm recovered as 6.9 × 6.9 × 21.6 cm, position within ~3 mm, ~3 s). The
  textured render matches the real frame to 4.3/255.
- **Images and overlay:** the lens-model warp agrees with OpenCV to < 0.7 px. The deployment overlay
  matches the training images (< 0.25% of pixels differ).
- **Operations:** `at-run` completes every stage; setup caching, versioning, the simulated comparison
  and promotion work.
- **Policy configs:** π0.5 and GR00T training configurations are accepted by LeRobot.

**Not yet run:**
- WiLoR, SAM 3, SAM 3D Objects, π0.5 and GR00T (need a GPU)
- real videos
- the real robot

The robot's motor zero, joint directions, gripper direction and base position are assumed, not
measured.

## Limitations

- **Supported skills:** picking up, placing, putting in/on, stacking, handovers, with rigid objects
  that fit the gripper (≤ ~8.8 cm) and are fully visible in the first frame.
- **Not yet supported:**
  - skills without a grasp (pushing, wiping, pressing)
  - drawers, lids, doors (risk of silent failure: modelled as rigid)
  - cloth, liquids
  - several identical objects
  - precise insertion or screwing
- **Success checks look only at the end state.** Process skills like scrubbing aren't captured.
- **Depth from one camera is approximate,** and transparent or shiny objects are hard to perceive.
- **Simulated success is optimistic.** It filters and ranks adapters; real-robot trials decide.
- **GR00T has no LoRA in LeRobot.** Each GR00T version is several GB.

## The example clips

`examples/Task1_Stacking.mp4` and `examples/Task2_Scrubbing.mp4` are top-down, wide-angle, 1080p
clips.
- **Stacking** fits the supported skills.
- **Scrubbing** needs a process-based success check that doesn't exist yet.
- **For either:** record a setup clip with the table AprilTag (none is visible), an empty-workspace clip
  and, given the visible lens distortion, a checkerboard clip.

## Next steps

1. **Bring up the GPU machine:** WiLoR, gated model access (PaliGemma, SAM 3, SAM 3D Objects), the
   SAM 3D environment. Run the synthetic CPU job, then a real clip up to perception.
2. **First real skill: stacking.** Record setup clips and 10–20 skill clips, run `at-run`, inspect the
   report and videos.
3. **Robot checks:** verify joint conventions, gripper range and base position on the real OpenArm.
   Compare real-robot success with simulated success.
4. **Extensions:** best-visibility frame for perception, a guard against articulated objects,
   contact-based phases for pushing/wiping, process-based success (scrubbing), several identical
   objects.
5. **Later:** co-train with Phantom-style overlays of the real clips; cross-embodiment; egocentric video.
