# AdapterTraining

Train **SmolVLA LoRA adapters** from MimicGen-style demonstration data with LeRobot.

```
teleop demos ─► MimicGen / DexMimicGen / Isaac Lab Mimic ─► robomimic HDF5
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
- `smolvla_base` loads `SmolVLM2-500M-Video-Instruct` weights on first use, so the first run downloads ~2 GB.
- Converting all 1000 demos of `stack_d1` takes ~10 min and ~100 MB.
