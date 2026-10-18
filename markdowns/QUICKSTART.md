# Quickstart: train a skill adapter

From a few fixed-camera clips of a person doing a task to a trained adapter for the OpenArm robot,
with one command. For how it works, see `REPORT.md`. For every option, see `README.md`.
No GPU machine? Skip section 1 and use [Modal](#run-on-modal-h200-instead).

```
setup clips (once)  +  skill clips  +  name  +  description
        └────────────── uv run at-run ──────────────┘
                              │
        dataset  +  adapter (π0.5 LoRA by default)  +  report.json
```

---

## 1. Machine

Use a **Linux machine with an NVIDIA GPU** (the defaults are sized for 80 GB, e.g. A100/H100), about
100 GB of free disk, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Thierryonre/AdapterTraining AdapterTraining && cd AdapterTraining
uv sync                              # everything except GR00T
uv run at-fetch-robot                # downloads the OpenArm description, builds the MuJoCo model
bash scripts/install_wilor.sh        # WiLoR hand tracker
```

### Model access (one time)

Three models are **gated** on Hugging Face. Request access with the account you'll use, then log in:

| Model | Used for | Page |
|---|---|---|
| `google/paligemma-3b-pt-224` | π0.5's tokenizer (default policy) | https://huggingface.co/google/paligemma-3b-pt-224 |
| `facebook/sam3` | finding objects from the description | https://huggingface.co/facebook/sam3 |
| `facebook/sam-3d-objects` | 3D shape of each object | https://huggingface.co/facebook/sam-3d-objects |

```bash
hf auth login
```

### SAM 3D Objects (own conda environment)

SAM 3D Objects reconstructs each object's 3D shape. It needs ≥ 32 GB of GPU memory and its **own conda
environment** (PyTorch 2.5, Kaolin, PyTorch3D), separate from this repo's uv environment. This repo
calls it as a separate process.

Install it in these exact locations and it's found automatically, with no environment variables:

| What | Location |
|---|---|
| conda (Miniforge) | `~/miniforge3` |
| SAM 3D Objects environment | `~/miniforge3/envs/sam3d-objects` (python: `~/miniforge3/envs/sam3d-objects/bin/python`) |
| SAM 3D Objects code + checkpoints | `AdapterTraining/third_party/sam-3d-objects` |

Run these from the `AdapterTraining` folder, after `hf auth login` above and after access to
`facebook/sam-3d-objects` has been granted.

**1. Install Miniforge** (conda + mamba). Skip this if `~/miniforge3` already exists.

```bash
curl -L -o /tmp/Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash /tmp/Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
source "$HOME/miniforge3/etc/profile.d/mamba.sh"
```

**2. Get the code and create the environment.** These are the steps from SAM 3D Objects' `doc/setup.md`.

```bash
git clone https://github.com/facebookresearch/sam-3d-objects.git third_party/sam-3d-objects
cd third_party/sam-3d-objects

mamba env create -f environments/default.yml        # creates the "sam3d-objects" environment
mamba activate sam3d-objects

export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"   # (their docs also list pypi.ngc.nvidia.com, which no longer exists)
pip install -e '.[dev]'
pip install -e '.[p3d]'                             # PyTorch3D, installed in a second step on purpose
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'
./patching/hydra                                    # patch from the SAM 3D repo
```

Build the environment on the GPU machine itself. Some parts fail with "Not compiled with GPU support"
otherwise.

**3. Download the checkpoints** (still inside `third_party/sam-3d-objects`, environment active):

```bash
pip install 'huggingface-hub[cli]<1.0'
hf download --repo-type model --local-dir checkpoints/hf-download --max-workers 1 facebook/sam-3d-objects
mv checkpoints/hf-download/checkpoints checkpoints/hf
rm -rf checkpoints/hf-download
```

This uses the Hugging Face login from above (the token is shared between environments).

**4. Go back and check it's found:**

```bash
mamba deactivate
cd ../..                                            # back to AdapterTraining
uv run python -m adapter_training.perception.reconstruct
```

Expected output, with your home folder:

```
SAM 3D python: /home/<you>/miniforge3/envs/sam3d-objects/bin/python
SAM 3D repo:   /home/<you>/AdapterTraining/third_party/sam-3d-objects
```

**Other locations.** If you already use Miniconda, Mambaforge or Anaconda in your home folder
(`~/miniconda3`, `~/mambaforge`, `~/anaconda3`), keep the environment name `sam3d-objects` and it's still
found. As a last resort `conda run -n sam3d-objects` is used. If the code or environment live somewhere
else entirely, set the paths explicitly. For example, with a cloud image that has conda in
`/opt/conda` and the repo in `/workspace`:

```bash
export SAM3D_PYTHON=/opt/conda/envs/sam3d-objects/bin/python
export SAM3D_REPO=/workspace/sam-3d-objects
```

Put those lines in `~/.bashrc` to keep them, then re-run the check in step 4.

### Claude (optional)

Claude identifies the objects from the skill description and first frame. It needs Anthropic
credentials in the environment (`ANTHROPIC_API_KEY`, or an `ant auth login` profile). Without them, or
with `"no_claude": true`, a simple noun-phrase rule is used instead.

### Weights & Biases (optional)

To watch training curves in the browser. `wandb` is already installed with the project.

```bash
cd ~/AdapterTraining
uv run wandb login          # paste the API key from https://wandb.ai/authorize
```

The key is stored in `~/.netrc`, so this is needed once per machine. On a machine where you can't log in
interactively, set it instead: `export WANDB_API_KEY=<your key>`.

Then add `--wandb` to `at-run` (or `"wandb": true` to the job file). Runs appear in the `lerobot`
project, named `<skill>_v<N>`, with:
- training loss, gradient norm and learning rate every 100 steps
- validation loss (held-out 10% of episodes) at each saved checkpoint

Simulated success rates aren't sent to W&B. They're in `report.json` and `sim_eval/`.

---

## 2. Camera setup (once per camera placement)

1. **Fix the camera** at the robot's head position, looking down at the table. It must not move from
   now on.
2. **Print a 36h11 AprilTag, id 0**, as large as fits on A4 (~16 cm black square). Measure the black
   square.
3. **Lay it flat on the table** with its **top edge pointing away from the robot side**.
4. Record into a setup folder, e.g. `data/setup/`:

| File | What to record | |
|---|---|---|
| `tag.mp4` | 3–5 s, tag visible, nobody in view | **required** |
| `empty.mp4` | 3–5 s of the empty workspace (tag removed if you like), no robot, no people | recommended |
| `checkerboard.mp4` | a printed checkerboard moved around the whole view | recommended |
| `setup.json` | `{"tag_size": 0.16, "tag_id": 0, "hfov": 70, "pattern": "9x6", "square": 0.025, "base_xyz": "-0.30,0,-0.45"}` | optional |

- `tag_size`: your measured black square, in metres.
- `pattern` / `square`: the checkerboard's inner corners (cols x rows) and square size, in metres.
- `hfov`: approximate field of view, used only without a checkerboard.
- `base_xyz`: where the robot's base is relative to the tag (x away from robot, y robot's left, z up),
  in metres. **Measure it on your setup.**
- `robot_side`: the image edge nearest the robot (`"top"`, `"bottom"`, `"left"`, `"right"`). Set it if
  the tag is visible but too small or blurred to decode (e.g. a phone screen far from a wide-angle
  camera). The tag's black square is then found by shape and its top edge taken as pointing away from
  the robot. The calibration prints "tag not decoded: used its black square"; check `camera.png`.
- `tag_corners_px`: last resort if even that fails: the black square's 4 corners in pixels, read off a
  frame of `tag.mp4`, e.g. `[[944,579],[998,582],[1000,530],[946,527]]` (any order; needs `robot_side`).

Calibration and the background photo are computed on the first run and cached in this folder.

---

## 3. Record the skill

Record **5–20 clips** of a person doing the skill, from the same fixed camera:

- Stand where the robot stands, facing the camera. Your **left hand drives the robot's left arm**.
- Grasp with **thumb and index finger** (the robot has a two-finger gripper, which opens up to ~8.8 cm).
- Keep **both hands visible**, and keep every object **fully visible in the first frame** before
  your hands reach it.
- One attempt per clip, 5–15 s. **Move the objects around between clips.**
- Prefer opaque, non-shiny objects. Clear plastic and glass are hard to segment and reconstruct.

---

## 4. Run it

Write a job file (example: `jobs/example_pick_up_bottle.json`):

```json
{
  "name": "Pick up water bottle",
  "description": "Pick up the water bottle with the right hand and lift it.",
  "videos": ["clips/bottle_01.mp4", "clips/bottle_02.mp4", "clips/bottle_03.mp4"],
  "setup": "data/setup",
  "train": true
}
```

```bash
uv run at-run --job jobs/example_stack_sponges.json --n 1000 --steps 30000 --save-freq 5000

uv run at-run --job jobs/example_scrub_surface.json --n 1000 --steps 50000 --save-freq 5000
```

Rule of thumb: steps ≈ passes × n × frames per demo ÷ batch size.

The **description is also the robot's instruction**. Write it as a short, clear command.

Common additions to the job file (any `at-skill` option works, with underscores):

| Key | Default | Meaning |
|---|---|---|
| `policy` | `"pi05"` | `"pi05"`, `"smolvla"` (small, runs anywhere) or `"groot"` (needs `uv sync --extra groot`) |
| `n` | `200` | demos to generate in simulation |
| `workers` | `4` | parallel simulation processes (use about your CPU core count) |
| `steps` | `20000` | training steps |
| `save_freq` | end only | also save a checkpoint every N steps; all are compared and the best is used (see [Choosing the best checkpoint](#choosing-the-best-checkpoint)) |
| `wandb` | `false` | log training curves to Weights & Biases (see [setup](#weights--biases-optional)) |
| `batch_size` | 64 (π0.5/GR00T), 32 (SmolVLA) | training batch size |
| `image_width` | `512` | generated image width |
| `sim_episodes` | `50` | simulated test episodes after training |
| `min_sim_success` | `0.2` | minimum simulated success to promote the new adapter |
| `shadows` | `false` | render shadows in generated images (cheap on a GPU) |

The command takes roughly an hour or more on a GPU machine, depending on `n`, `steps` and the number of
clips (not measured here).

---

## 5. Check the result

The command ends with a summary like:

```
Skill: Pick up water bottle  ->  done
  video              done
  objects            done
  perception         done
  simulation_replay  done
  generation         done
  validation         done
  dataset            done
  training           done
  sim_evaluation     done
  offline_evaluation done
  adapter: outputs/train/pick_up_water_bottle/v1/checkpoints/last/pretrained_model (promoted: v1)
  simulated success: 64% (previous —)
```

(Example output. The success rate is illustrative.)

Everything is under `outputs/skills/<skill>/`:

| Look at | To check |
|---|---|
| `report.json` | every stage's numbers, the inferred task, any stop reason |
| `part1/*_check.mp4` | the robot copies the hands (left: your video, right: simulated robot) |
| `part1/scenes/*_objects.png` | each object's fitted outline (green) sits on the real object |
| `datagen/source_*.mp4` | each clip replayed in physics |
| `datagen/generated_*.mp4` | generated demos (what the policy trains on) |
| `sim_eval/*.mp4` | the trained adapter acting in simulation |

Exit code 1 means a quality check stopped the job, and the reason is printed and stored in
`report.json`. See [Troubleshooting](#troubleshooting).

**Adapter versions:** each run adds `outputs/train/<skill>/vN`. `outputs/train/<skill>/latest.txt`
names the promoted version, which is the one to deploy. A new version is promoted only if it reaches
`min_sim_success` and isn't more than 5 points worse than the previous one.

### Choosing the best checkpoint

By default training saves one checkpoint, at the end. With `--save-freq 5000` (or `"save_freq": 5000` in
the job file) it also saves one every 5,000 steps under `outputs/train/<skill>/vN/checkpoints/` (`005000`,
`010000`, …), and `at-run` then picks the best one for you:

- it runs **every** saved checkpoint of the new version in simulation, on the same object layouts
  (and the previously promoted version, for the promotion check)
- the checkpoint with the highest simulated success wins (the earliest one on a tie); its step is
  written to `outputs/train/<skill>/vN/best.txt`
- that checkpoint is what's promoted, deployed and reported (`adapter:` in the summary;
  `checkpoints` and `best_checkpoint` under `sim_evaluation` in `report.json`)

Each checkpoint adds `sim_episodes` simulated episodes, so 10 checkpoints take about 10 times as long as
one. For a reliable choice, set `"sim_episodes": 100`: with 50, differences of a few points are noise.
The chosen checkpoint's success rate is slightly flattering, because it was picked on those same episodes.

To re-rank later without listing checkpoints (e.g. with more episodes), point `at-sim-eval` at the
version folder with `--all-checkpoints`. It finds every saved checkpoint itself and prints the best:

```bash
uv run at-sim-eval --all-checkpoints --checkpoints outputs/train/stack_sponges/v1 \
  --sources outputs/skills/stack_sponges/datagen/sources.hdf5 \
  --calib data/setup/camera.json --background data/setup/background.png \
  --episodes 100 --out outputs/sim_eval_stack_sponges
```

(Check `outputs/train/` for the exact folder name of your skill.) This only reports; it doesn't change
`best.txt`.

Simulated success is preferred over validation loss, which often keeps falling after success has
stopped improving. It is still optimistic: use it to rank checkpoints, then confirm on the real robot.

---

## 6. Run it on the robot

Use the overlay robot type. It draws the simulated robot over the camera image so it looks the same as
in training. Keep the camera exactly where it was calibrated, at the same resolution.

```bash
lerobot-rollout --strategy.type=base \
  --robot.type=bi_openarm_overlay \
  --robot.left_arm_config.port=can0 --robot.left_arm_config.side=left \
  --robot.right_arm_config.port=can1 --robot.right_arm_config.side=right \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 30}}" \
  --robot.overlay_calib=data/setup/camera.json \
  --task="Pick up the water bottle with the right hand and lift it." \
  --policy.path=outputs/train/pick_up_water_bottle/v1/checkpoints/last/pretrained_model
```

- `--policy.path` is the `adapter:` path from the summary. If you used `save_freq`, that's
  `checkpoints/<step in best.txt>/pretrained_model` instead of `checkpoints/last/...`.
- `--task` must be exactly the skill description.
- GR00T checkpoints also need `--policy.base_model_path=nvidia/GR00T-N1.7-3B`.

**Before the first real run, check these (not verified on hardware):**
- Move the robot slowly to an exported pose to confirm the motor zero positions and directions match
  the URDF.
- Gripper direction: 0° = closed and −65° = open is assumed (`gripper_motor_range` in `video/robot.py`).
- `base_xyz` in `setup.json` matches where the robot really stands.

Simulated success is optimistic, so test on the real robot with a hand near the stop button.

---

## Run on Modal (H200) instead

No GPU machine to set up: `modal_app.py` runs the same `at-run` on a Modal H200 (141 GB). The image
contains everything in section 1 (uv environment, WiLoR, SAM 3D Objects with its compiled CUDA code,
the OpenArm model). You still do sections 2–4 on your computer: setup clips, skill clips, job file.

**Once:**

1. Request access to the gated models (table in [Model access](#model-access-one-time)).
2. Log in to Modal (opens the browser; stores a token in `~/.modal.toml`):
   ```bash
   uvx modal setup
   ```
3. Create the secret `adapter-training`. Put the values in a file `modal.env` in the repo (it's in
   `.gitignore`):
   ```
   HF_TOKEN=hf_...                 # https://huggingface.co/settings/tokens (read access)
   # optional
   ANTHROPIC_API_KEY=sk-ant-...    # Claude names the objects (otherwise a simple rule)
   WANDB_API_KEY=...               # with --wandb in --args
   ```
   ```bash
   uvx modal secret create adapter-training --from-dotenv modal.env
   ```
4. Optional, but catches problems before a long job. The first time, this builds the image (tens of
   minutes, mostly compiling SAM 3D's CUDA code; cached afterwards), then tests it on an H200 for a
   minute or two:
   ```bash
   AT_MODAL_NO_SECRET=1 uvx modal run modal_app.py::check     # ends with "check passed"
   ```

**Each skill:**

```bash
uvx modal run --detach modal_app.py --job jobs/example_stack_sponges.json \
  --args "--n 1000 --steps 30000 --save-freq 5000 --workers 16"
```

- `--args` takes any `at-run` option. `--workers 16` matches the 16 CPUs the job gets.
- The job file, its clips and its setup folder are uploaded automatically. They must be inside the
  repo.
- `--detach` keeps it running if you close the terminal. Follow it with
  `uvx modal app logs adapter-training`, or on modal.com.
- The first run also downloads SAM 3D Objects' checkpoints to the `adapter-training-cache` volume.

**Where the results are:** the volume `adapter-training-work`.

| On the volume | What |
|---|---|
| `outputs/train/<skill>/vN/checkpoints/*` | every checkpoint (`best.txt` names the chosen one) |
| `outputs/skills/<skill>/` | `report.json`, `sim_eval/` videos, check videos |
| `lerobot/<skill>` | the generated LeRobot dataset |

The volume is saved every 10 minutes while the job runs, so checkpoints survive a crash or the
time limit. To copy them to your computer:

```bash
uvx modal volume ls adapter-training-work outputs/train/stack_sponges
uvx modal volume get adapter-training-work outputs/train/stack_sponges/v1 outputs/train/stack_sponges/
```

Or add `--download` to the run command (without `--detach`, terminal kept open): it copies
`outputs/train/<skill>/` and the `sim_eval/` videos into this repo when the job ends.

**Limits:**
- 24 hours per job (Modal's maximum). n = 1000 with 50k π0.5 steps plus checkpoint comparison may not
  fit: check the timings of a smaller job first. Checkpoints saved before the limit are kept.
- Modal can restart a job on a new machine (rare). It then starts again from the beginning.
- Costs: H200 time for the whole job, including generation and simulated evaluation (see
  modal.com/pricing).

**Optional: also copy checkpoints to Google Drive.** Install rclone locally, run
`rclone authorize "drive"`, sign in, and add the printed token to `modal.env`, then re-create the
secret with `--force`:
```
RCLONE_CONFIG_GDRIVE_TYPE=drive
RCLONE_CONFIG_GDRIVE_SCOPE=drive
RCLONE_CONFIG_GDRIVE_TOKEN='{"access_token":"...","token_type":"Bearer","refresh_token":"...","expiry":"..."}'
```
Checkpoints and reports then also go to the Drive folder `AdapterTraining/` (`--drive-folder` to
change it) every 10 minutes and at the end.

**Networks that intercept TLS** (antivirus, corporate proxy): if `modal` says "Could not connect to
the Modal server", use the wrapper instead of `uvx modal`, with the same arguments:
```bash
uvx --native-tls --from modal --with truststore python scripts/modal_cli.py run modal_app.py::check
```

---

## Try it without a GPU or filming

A synthetic "pick up the water bottle" scene exercises every stage on a CPU. The GPU models are
replaced by stand-ins and training runs for 1 step, so it checks the plumbing, not the model:

```bash
uv sync
uv run at-fetch-robot
uv run at-synthetic-demo --scenario bottle --out data/synthetic_bottle --demos 3
mkdir -p data/setup_test && cp data/synthetic_bottle/tag.mp4 data/setup_test/
echo '{"tag_size": 0.16, "hfov": 70}' > data/setup_test/setup.json
mkdir -p outputs/skills/pick_up_water_bottle/part1
cp data/synthetic_bottle/bottle_*_tracks.npz outputs/skills/pick_up_water_bottle/part1/
uv run at-run --job jobs/synthetic_cpu_test.json --n 100
```

`jobs/synthetic_cpu_test.json` is the section 4 job plus the keys that make it CPU-only: SmolVLA, 1
training step, 8 generated demos, 2 simulated test episodes, MediaPipe hand tracking on pre-computed
synthetic tracks, no Claude, ground-truth masks instead of SAM 3, and a stand-in mesh instead of
SAM 3D Objects.

Expect all stages to reach `done` and the simulated success to be 0%, because one training step
learns nothing. Training needs several GB of free RAM; close other applications if it crashes.

---

## Troubleshooting

| Stopped at | Reason shown | What to do |
|---|---|---|
| video | hands tracked in < 80% of frames | keep both hands in view; better lighting; check `*_check.mp4` |
| objects | no objects identified | make the description name the objects ("the water bottle") |
| perception | `'x' not found in the first frame` | object hidden at the start, or the name doesn't match what SAM 3 sees: rename it in the description |
| perception | no shape matched its outline / implausible size | clear, shiny or partly hidden objects: use an opaque object, or check `*_objects.png` |
| simulation_replay | only 0 clip(s) reproduced in simulation | see `datagen/prepare_report.json`: grasp not holding (object too wide or slippery, grasp too low), or clips with a different grasp order |
| generation | low success rate | more varied clips; check `generated_*.mp4` |
| sim_evaluation | below the minimum / worse than previous | more clips or steps; the adapter is kept as `vN` but not promoted |
| training (crash) | segmentation fault | out of RAM: close applications, lower `batch_size` |

Other things to check:
- **Calibration:** `data/setup/camera.png` should show red x pointing away from the robot and blue z up.
  If not, the tag was laid the wrong way round.
- **On Windows**, downloads failing with certificate errors are handled automatically (Windows
  certificate store). Prefer Linux for real runs.
