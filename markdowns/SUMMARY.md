# Teaching a robot new skills from ordinary videos: summary

**What this is.** A pipeline that adds a skill to a two-arm robot (OpenArm v1) from a few ordinary videos
of a person doing the task. Someone uploads the clips, a skill name and a one-line description. The
system produces a trained skill "adapter" for a robot foundation model, plus a report on whether it can
be trusted. There is no teleoperation, no object labelling and no task-specific code.

**Status, in one paragraph.** The whole pipeline is built (about 6,000 lines of Python) and runs end to end
on synthetic test scenes with known ground truth. The GPU software stack has been installed and checked
on H200 GPUs (Modal and RunPod). **It has not yet been run on real videos or on the real robot**, so there
is no real-world success rate yet. Everything below separates what was measured from what is a design
choice or an expectation.

---

## 1. The approach

```
fixed camera (head height, looking down) + one AprilTag for calibration
   │
   ├─ 1 Video → robot motion   track both hands in 3D, map thumb+index to a two-finger gripper,
   │                           inverse kinematics on the robot model (keeps the arms above the table)
   ├─ 2 Objects, no labels     names from the description (LLM) → masks (SAM 3) → 3D shape (SAM 3D Objects)
   │                           → fitted to the table at the right size → textured from the video
   ├─ 3 Task inferred          replay in physics; split at grasps/releases; which object each arm works
   │                           relative to; success = how the demonstration ends
   ├─ 4 Multiply               DexMimicGen method: move objects, adapt each phase, keep only successes
   │                           → hundreds of demos from a few clips
   ├─ 5 Realistic images       simulated robot + objects composited onto a real photo of the workspace
   ├─ 6 Train                  LoRA adapter on π0.5 (default), SmolVLA, or GR00T N1.7
   └─ 7 Gate                   closed-loop test in simulation vs the previous version → promote or not
deploy: LeRobot, with the simulated robot drawn over the live camera image (the "Phantom" trick),
        so the policy sees the same robot it was trained on
```

The key ideas:
- **Human video gives intent and motion. Simulation gives quantity and variation.** A person shows the
  task once or a few times. Simulation turns that into hundreds of varied robot demonstrations, keeping
  only the ones that succeed.
- **Remove the human from the images.** The model never sees human hands. It sees a rendered robot both
  in training and at deployment, which is the main bridge from human video to robot.
- **No per-skill engineering.** Phases, reference objects and the success check are inferred from the
  replayed demonstration. One command (`at-run`) runs everything, and quality gates stop the job with a
  plain-language reason instead of training on bad data.

---

## 2. Main results

All on synthetic scenes with known ground truth, on a laptop without an NVIDIA GPU unless stated.

| What was tested | Result |
|---|---|
| Skills inferred with no task-specific code | two-arm handover of a bar; one-arm water-bottle pickup |
| Video → robot motion (synthetic hands) | median gripper position error ~1–1.5 mm; worst 5% of frames ~1–2.5 cm |
| Demo multiplication (DexMimicGen-style) | 68–78% of attempts succeed (handover 77.5%, bottle 71–75%) |
| Validation of generated demos | 100% replay successfully from their recorded actions (e.g. 12/12) |
| 3D object fitting | bottle given at the wrong scale and lying down → recovered 6.9 × 6.9 × 21.6 cm (true 7 × 7 × 22), position within ~3 mm, in ~3 s |
| Textures from the video | rendered object matches the real frame to 4.3/255 per pixel |
| Camera model | lens warp agrees with OpenCV to < 0.7 px; deployment overlay matches training images (< 0.25% of pixels differ) |
| Training stack | LoRA training, offline evaluation, versioning and simulated comparison run end to end (1–2 step "smoke" runs; MimicGen public data first) |
| GPU stack (H200, Modal + RunPod) | all environments build and pass GPU checks: hand tracker, 3D reconstruction (compiled CUDA code), physics rendering on the GPU |
| **Not yet run** | real videos; SAM 3 / SAM 3D on real images; a full π0.5 or GR00T training; the real robot; any real success rate |

**Practical findings that cost the most time** (useful to anyone building something similar):
- **Physical consistency matters more than model choice.** The biggest early failures were geometric, not
  learning problems: unreachable poses, elbows through the table, grasps a few cm off, an object
  reconstructed lying down instead of upright. Each was fixed by measuring against ground truth.
- **Hand tracking is centimetres off, so grasps are corrected in simulation.** Each grasp is nudged onto
  the object before multiplying, otherwise demos fail in physics.
- **The research software stack is fragile.** One 3D model needs its own environment and compiled GPU
  code. Its official install instructions point at a package server that no longer exists. One hand
  tracker silently downgrades the deep-learning library. Containerising (Modal) and a re-runnable setup
  script (RunPod) turned a day of installing into one command.
- **Calibration is the one manual step.** A few seconds of video of a printed tag (a phone screen works,
  less accurately) and a measured robot position. Errors there shift every grasp on the real robot.

---

## 3. Answers to the key questions

**Skill training: a high success rate with minimal data?**
*Expected yes for simple manipulation, but not yet measured on the real robot.* The design spends data
where it's cheap: 1–5 human clips per skill, multiplied to 200–1000 simulated demos. Published MimicGen
results show 10 source demos becoming ~1000 with large success gains; beyond ~1000 the gains are small.
Simulated success here is a gate and a way to rank models, not a real-world success rate (it's
optimistic). *Recommendation:* 3–5 varied clips, n ≈ 500–1000 generated demos, about 5–10 passes of
training, intermediate checkpoints compared in simulation, then a fixed real-robot test protocol.

**Data efficiency: the best mix of human video, teleoperation and simulation?**
- **Human video:** what to do and roughly how. Cheapest to collect, needs no robot or expert, scales
  through an app.
- **Simulation:** variation (object positions, appearance) and volume, with automatic success filtering.
- **Teleoperation:** a small amount (tens of episodes) for what video and simulation get wrong: contact-rich
  details, recovering from mistakes, and the real-world evaluation set.

This project deliberately uses no teleoperation. For a product, the recommendation is video + simulation
for coverage, plus a small teleop budget where real-world tests show a gap.

**Skill factory: one pipeline, many skills?**
*Yes, by construction.* Adding a skill is a job file (clips, name, description) and one command. Each skill
gets versioned adapters, an automatic simulated test against the previous version, and a report an app can
show. *Limits:* it handles what the success inference can express, meaning grasping, moving, placing,
stacking and handing over rigid objects, with both arms. It does not handle deformable objects, articulated
objects (drawers, lids), or tasks judged by force or effect rather than final arrangement. Scrubbing a
surface is an example: the motion can be copied, but "clean" can't be checked.

**One-shot learning: from a single demonstration?**
*Mechanically yes, practically limited.* One clip is enough to run the whole pipeline: it is multiplied into
hundreds of variations. But every variation is the same strategy shifted by a few centimetres and degrees
(±4–5 cm, ±20°), so the robot generalises around that one demo, not beyond it. Several clips with
different positions and grasps help more than more generated copies of one.

**Skill transfer: deploy on another robot without retraining?**
*Not without retraining today; deferred by choice.* The pipeline is driven by the robot's model file
(URDF), so a new two-arm robot needs **no new human data**. The same clips are retargeted and regenerated,
and the adapter is retrained (hours on one GPU). True zero-shot transfer would rely on cross-embodiment
base models (GR00T's per-robot input/output modules, π0.5's normalised actions), which is still a research
question. *Recommendation:* treat "re-target and retrain from the same videos" as the transfer mechanism.
It's cheap and predictable.

**Skill composition: combine skills into new capabilities?**
*Not built; a clear path exists.* Each skill is language-conditioned and comes with an inferred success
check. A planner (an LLM or vision-language model) can sequence skills by instruction and use those checks
to decide when each step is done. Merging or stacking adapters (mixtures of LoRA experts) was discussed and
is riskier. *Recommendation:* sequence skills with a planner first. Merge adapters only if there is a
measured need.

**Easy deployment: usable without being a robotics engineer?**
*Partly.* Using it is easy: upload clips plus a name and description, get an adapter and a plain-language
report, and deploy with one LeRobot command. Setting it up still needs care: a fixed camera, a
one-off calibration (tag clip plus a measured robot position), and a heavy GPU stack. The GPU stack is
now packaged as a Modal image and a RunPod setup script. *Recommendation:* hide the GPU stack behind a
hosted service, and ship a setup kit (camera mount, printed tag, guided calibration with an on-screen
check).

---

## 4. Recommendations

**Technical**
1. **Measure real-world success next**, before adding features: one simple skill (pick up a bottle), fixed
   test layouts, 20–50 trials, compared with the base model and with versions trained on more clips or
   more generated data.
2. Record **3–5 varied clips per skill** with an AprilTag setup clip from the same camera position. Use
   n ≈ 500–1000, save checkpoints, and let the simulated comparison pick the best one.
3. Keep the **simulated gate** as a release check, never as the headline number.
4. Add **a small teleop or correction loop** only where real tests fail.
5. Build **composition with a planner** on top of the existing language-conditioned skills; build transfer
   as retarget-and-retrain.

**Business**
1. **Product shape:** "upload a video, get a skill" as a cloud service. The heavy models live on rented
   GPUs (H200, minutes to hours per skill), and customers only need a camera, the robot and the setup kit.
2. **Wedge:** narrow, repetitive pick-and-place and handover tasks on one robot model with a fixed camera,
   where many variants of the same skill are needed and collecting video is far cheaper than teleop.
3. **Moat:** the automation (no per-skill code, inferred success checks, quality gates, versioning) and
   the accumulated per-skill data. The base models are commodities.
4. **Risks:** no real-world numbers yet. Fixed-camera calibration is mandatory. Skills judged by force or
   effect aren't covered. Licences of the third-party models (SAM 3 / SAM 3D, PaliGemma used by π0.5,
   GR00T) must be reviewed before commercial use.

---

## 5. Where things are

- **Docs:** `QUICKSTART.md` (train a skill, including running on Modal), `OVERVIEW.md` (what and how),
  `REPORT.md` (everything, including all discussions and fixes), `README.md` (every option).
- **One command:** `uv run at-run --job jobs/<skill>.json`. In the cloud:
  `uvx modal run modal_app.py --job ...`, or on RunPod run `/workspace/at/runpod_setup.sh` once, then the
  same command.
- **Defaults:** π0.5 LoRA adapter (80 GB GPU), 200 generated demos (n ≈ 1000 recommended), 20,000
  steps, 50 simulated test episodes, promotion if ≥ 20% simulated success and not worse than the
  previous version.
