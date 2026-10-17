"""Fine-tune a policy (SmolVLA LoRA adapter, or GR00T N1.7) on a converted LeRobot dataset.

Thin wrapper around `lerobot-train`. Runs LeRobot in-process so the TLS fix in
`adapter_training/__init__.py` applies.

SmolVLA (default): LoRA defaults from the LeRobot PEFT docs (https://huggingface.co/docs/lerobot/peft_training).
π0.5 (--policy pi05): LoRA adapter on lerobot/pi05_base (LeRobot's default π0.5 targets: the action
expert's attention + projections), bf16. Its tokenizer is the gated google/paligemma-3b-pt-224: accept
the licence on the Hub and `hf auth login` first. Sized for an 80 GB GPU (batch 64).
GR00T N1.7 (--policy groot): the LeRobot/NVIDIA recipe (https://huggingface.co/docs/lerobot/groot) --
the 3B vision-language backbone stays frozen; the projector and the flow-matching action head are
trained for the robot ("new_embodiment"). LeRobot's GR00T has no working LoRA option (its lora_* config
fields are deprecated and unused), so checkpoints contain the full policy (several GB). Needs an
NVIDIA GPU (bf16) and `uv sync --extra groot`.

    # Smoke test on CPU
    uv run at-train --dataset local/mimicgen_stack_d1 --steps 5 --batch-size 2

    # Real run on a GPU
    uv run at-train --dataset local/mimicgen_stack_d1 --steps 20000 --batch-size 64

    # Full fine-tune of the action expert instead of LoRA (for comparison)
    uv run at-train --dataset local/mimicgen_stack_d1 --no-peft

    # π0.5 LoRA adapter (GPU)
    uv run at-train --policy pi05 --dataset local/pick_up_water_bottle --steps 20000 --batch-size 64

    # GR00T N1.7 (GPU)
    uv run at-train --policy groot --dataset local/pick_up_water_bottle --steps 20000 --batch-size 64

Anything after `--` is passed straight to lerobot-train, e.g. `-- --wandb.enable=true`.
"""

import argparse
import shlex
import sys
from datetime import datetime
from pathlib import Path

import adapter_training  # noqa: F401  (TLS setup)
from adapter_training.common import OUTPUTS_DIR, dataset_root, default_device, default_video_backend

PROJECTIONS = ["state_proj", "action_in_proj", "action_out_proj"]
BASE_MODELS = {"smolvla": "lerobot/smolvla_base", "pi05": "lerobot/pi05_base", "groot": "nvidia/GR00T-N1.7-3B"}
DEFAULT_LR = {"smolvla": (1e-4, 1e-3), "pi05": (2.5e-5, 2.5e-4)}  # (full fine-tune, LoRA ~10x)
LORA_TIME_MLPS = "(action_time_mlp_in|action_time_mlp_out)"


def resolve_base_model(base_model: str, download: bool = True) -> str:
    """Download a Hub model to a local folder and return its path.

    LeRobot 0.6.1 turns `--policy.path` into a Path, which on Windows mangles Hub ids into
    'lerobot\\smolvla_base'. A local path avoids that, and the PEFT adapter config then records a
    local base-model path that loads offline at evaluation/deployment time.
    """
    if Path(base_model).exists() or not download:
        return base_model
    from huggingface_hub import snapshot_download

    return snapshot_download(base_model)


def build_lerobot_args(args: argparse.Namespace) -> list[str]:
    kind = "groot" if args.policy == "groot" else ("full" if args.no_peft else "lora")
    job_name = args.job_name or f"{args.dataset.split('/')[-1]}_{kind}"
    output_dir = args.output_dir or OUTPUTS_DIR / "train" / f"{job_name}_{datetime.now():%Y%m%d_%H%M%S}"
    device = args.device or default_device()
    base_model = args.base_model or BASE_MODELS[args.policy]

    cli = [
        f"--dataset.repo_id={args.dataset}",
        f"--dataset.root={args.dataset_root or dataset_root(args.dataset)}",
        f"--dataset.video_backend={default_video_backend()}",
        "--policy.push_to_hub=false",
        f"--policy.device={device}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--save_freq={args.save_freq or args.steps}",
        f"--log_freq={args.log_freq}",
        f"--num_workers={args.num_workers}",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--wandb.enable={str(args.wandb).lower()}",
    ]
    if args.eval_split > 0:
        # Hold out the last episodes and report validation loss during training.
        cli += [
            f"--dataset.eval_split={args.eval_split}",
            f"--eval_steps={args.eval_steps or args.save_freq or args.steps}",
            f"--max_eval_samples={args.max_eval_samples}",
        ]

    if args.policy == "groot":
        cli += groot_args(args, base_model)
        return cli + args.extra

    if args.policy == "pi05":
        # pretrained_path loads weights only: features (cameras, state/action sizes) come from the dataset.
        cli += [
            "--policy.type=pi05",
            f"--policy.pretrained_path={resolve_base_model(base_model, download=not args.dry_run)}",
            "--policy.dtype=bfloat16",
            f"--policy.gradient_checkpointing={str(args.pi05_gradient_checkpointing).lower()}",
        ]
    else:
        # SmolVLA. Take camera/state/action shapes from the dataset instead of the SO-100 setup it was pretrained on.
        cli += [f"--policy.path={resolve_base_model(base_model, download=not args.dry_run)}",
                "--policy.input_features=null", "--policy.output_features=null"]
    full_lr, lora_lr = DEFAULT_LR[args.policy]
    if args.no_peft:
        cli += [f"--policy.optimizer_lr={args.lr or full_lr}"]
    else:
        # LoRA tolerates ~10x the full fine-tuning learning rate.
        lr = args.lr or lora_lr
        cli += [
            f"--policy.optimizer_lr={lr}",
            f"--policy.scheduler_decay_lr={lr / 10}",
            "--peft.method_type=LORA",
            f"--peft.r={args.rank}",
            f"--peft.lora_alpha={args.alpha or args.rank}",
        ]
        if args.init:
            cli.append(f"--peft.init_type={args.init}")
        if args.full_train_projections:
            # The state/action projections must learn a brand-new robot layout, so train them fully
            # and keep LoRA on the expert's attention and the timestep MLPs only.
            cli += [
                rf"--peft.target_modules=(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.{LORA_TIME_MLPS})",
                f"--peft.full_training_modules=[{','.join(PROJECTIONS)}]",
            ]

    if args.warmup_steps is not None:
        cli.append(f"--policy.scheduler_warmup_steps={args.warmup_steps}")
    if args.decay_steps is not None:
        cli.append(f"--policy.scheduler_decay_steps={args.decay_steps}")

    return cli + args.extra


def groot_args(args: argparse.Namespace, base_model: str) -> list[str]:
    """GR00T N1.7 fine-tuning flags, following the LeRobot GR00T docs."""
    if (args.device or default_device()) != "cuda":
        print("WARNING: GR00T N1.7 (3B, bf16) is meant for an NVIDIA GPU; this will be extremely slow or fail.")
    cli = [
        "--policy.type=groot",
        f"--policy.base_model_path={base_model}",  # a plain string field: Hub ids are fine here
        "--policy.embodiment_tag=new_embodiment",  # the slot GR00T reserves for robots it wasn't trained on
        f"--policy.chunk_size={args.groot_chunk}",
        f"--policy.n_action_steps={args.groot_chunk}",
        # Arm joints as relative actions, grippers absolute (matches left_gripper.pos / right_gripper.pos).
        "--policy.use_relative_actions=true",
        '--policy.relative_exclude_joints=["gripper"]',
        "--policy.use_bf16=true",
        f"--policy.tune_llm={str(args.groot_tune_llm).lower()}",
        f"--policy.tune_visual={str(args.groot_tune_visual).lower()}",
        "--dataset.image_transforms.enable=true",
        "--use_policy_training_preset=true",
    ]
    if args.lr:
        cli.append(f"--policy.optimizer_lr={args.lr}")
    return cli


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="LeRobot dataset id, e.g. local/mimicgen_stack_d1")
    parser.add_argument("--dataset-root", help="Dataset folder (default: data/lerobot/<name>)")
    parser.add_argument("--policy", default="smolvla", choices=["smolvla", "pi05", "groot"])
    parser.add_argument("--base-model", help="Default: lerobot/smolvla_base, lerobot/pi05_base or nvidia/GR00T-N1.7-3B")
    parser.add_argument("--job-name")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", help="cuda / mps / cpu (default: auto)")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-freq", type=int, help="Checkpoint every N steps (default: only at the end)")
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, help="Default 1e-3 for LoRA, 1e-4 for full fine-tuning / GR00T preset")
    parser.add_argument("--warmup-steps", type=int, help="LR warmup (SmolVLA default 1000)")
    parser.add_argument("--decay-steps", type=int, help="Cosine decay length (SmolVLA default 30000)")
    parser.add_argument("--eval-split", type=float, default=0.1, help="Fraction of episodes held out (0 disables)")
    parser.add_argument("--eval-steps", type=int, help="Validation loss every N steps (default: at each save)")
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=1000,
        help="Cap on held-out frames per validation pass (0 = all; slow on CPU)",
    )
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")

    lora = parser.add_argument_group("LoRA")
    lora.add_argument("--no-peft", action="store_true", help="Full fine-tune instead of LoRA (SmolVLA: action expert; π0.5: everything)")
    lora.add_argument("--rank", type=int, default=64)
    lora.add_argument("--alpha", type=int, help="LoRA alpha (default: = rank)")
    lora.add_argument("--init", help="LoRA init, e.g. 'pissa', 'olora', 'gaussian' (PEFT init_lora_weights)")
    lora.add_argument(
        "--full-train-projections",
        action="store_true",
        help="Fully train state/action projections instead of adapting them with LoRA",
    )
    pi05 = parser.add_argument_group("π0.5")
    pi05.add_argument("--pi05-gradient-checkpointing", action="store_true",
                      help="Trade speed for memory (not needed for LoRA on an 80 GB GPU)")
    groot = parser.add_argument_group("GR00T")
    groot.add_argument("--groot-chunk", type=int, default=16, help="Action chunk length (LeRobot docs: 16)")
    groot.add_argument("--groot-tune-llm", action="store_true", help="Also train the language model (much more memory)")
    groot.add_argument("--groot-tune-visual", action="store_true", help="Also train the vision encoder")
    parser.add_argument("--dry-run", action="store_true", help="Print the lerobot-train command and exit")

    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1 :]
    args = parser.parse_args(argv)
    args.extra = extra

    cli = build_lerobot_args(args)
    print("lerobot-train " + " ".join(shlex.quote(a) for a in cli), flush=True)
    if args.dry_run:
        return

    from lerobot.scripts.lerobot_train import main as lerobot_train_main

    sys.argv = ["lerobot-train", *cli]
    lerobot_train_main()


if __name__ == "__main__":
    main()
