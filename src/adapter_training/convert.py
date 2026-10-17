"""Convert a MimicGen / robomimic HDF5 file into a LeRobot dataset.

robomimic layout (used by MimicGen, DexMimicGen and Isaac Lab Mimic exports):

    data/                     attrs: env_args (JSON)
      demo_0/
        actions               (T, action_dim)
        obs/<key>             (T, ...)   e.g. agentview_image, robot0_eef_pos
      demo_1/ ...

Example (single-arm MimicGen Stack, first 50 demos):

    uv run at-convert data/raw/core/stack_d1.hdf5 --max-episodes 50

For a bimanual robosuite dataset you would pass e.g.
    --state-keys robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos,robot1_eef_pos,robot1_eef_quat,robot1_gripper_qpos
    --cameras agentview_image:head,robot0_eye_in_hand_image:wrist_right,robot1_eye_in_hand_image:wrist_left
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

import adapter_training  # noqa: F401  (TLS setup)
from adapter_training.common import dataset_root
from adapter_training.tasks import task_for_env

DEFAULT_CAMERAS = "agentview_image:front,robot0_eye_in_hand_image:wrist"
DEFAULT_STATE_KEYS = "robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos"
OSC_POSE_ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def parse_cameras(spec: str) -> dict[str, str]:
    """'agentview_image:front,robot0_eye_in_hand_image:wrist' -> {hdf5_key: lerobot_name}."""
    cameras = {}
    for item in spec.split(","):
        src, _, dst = item.partition(":")
        cameras[src.strip()] = (dst or src.removesuffix("_image")).strip()
    return cameras


def sorted_demos(data: h5py.Group, use_mask: str | None, h5: h5py.File) -> list[str]:
    if use_mask:
        demos = [d.decode() if isinstance(d, bytes) else d for d in h5["mask"][use_mask][:]]
    else:
        demos = list(data.keys())
    return sorted(demos, key=lambda d: int(re.sub(r"\D", "", d) or 0))


def build_features(demo: h5py.Group, cameras: dict[str, str], state_keys: list[str], use_videos: bool) -> dict:
    features = {}
    for src, dst in cameras.items():
        h, w, c = demo["obs"][src].shape[1:]
        features[f"observation.images.{dst}"] = {
            "dtype": "video" if use_videos else "image",
            "shape": (h, w, c),
            "names": ["height", "width", "channels"],
        }

    state_names = []
    for key in state_keys:
        state_names += stored_names(demo["obs"][key]) or [f"{key}_{i}" for i in range(demo["obs"][key].shape[1])]
    features["observation.state"] = {"dtype": "float32", "shape": (len(state_names),), "names": state_names}

    action_dim = demo["actions"].shape[1]
    action_names = stored_names(demo["actions"]) or (
        OSC_POSE_ACTION_NAMES if action_dim == 7 else [f"a{i}" for i in range(action_dim)]
    )
    features["action"] = {"dtype": "float32", "shape": (action_dim,), "names": action_names}
    return features


def stored_names(dataset: h5py.Dataset) -> list[str] | None:
    """Per-dimension names saved by at-video2robot (e.g. LeRobot motor names like 'left_joint_1.pos')."""
    if "names" not in dataset.attrs:
        return None
    return [n.decode() if isinstance(n, bytes) else str(n) for n in dataset.attrs["names"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="robomimic/MimicGen HDF5 file")
    parser.add_argument("--repo-id", help="LeRobot dataset id (default: local/mimicgen_<file stem>)")
    parser.add_argument("--root", type=Path, help="Output folder (default: data/lerobot/<repo name>)")
    parser.add_argument("--task", help="Language instruction (default: looked up from env name)")
    parser.add_argument("--cameras", default=DEFAULT_CAMERAS, help="hdf5_key:name pairs, comma separated")
    parser.add_argument("--state-keys", default=DEFAULT_STATE_KEYS, help="obs keys concatenated into the state")
    parser.add_argument("--max-episodes", type=int, help="Only convert the first N demos")
    parser.add_argument("--mask", help="Use a robomimic filter key, e.g. 'train' or 'valid'")
    parser.add_argument("--fps", type=int, help="Override fps (default: env control_freq)")
    parser.add_argument("--robot-type", default="panda")
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Flip images vertically (raw robosuite renders are upside down; MimicGen releases are already upright)",
    )
    parser.add_argument("--no-videos", action="store_true", help="Store frames as images instead of MP4")
    parser.add_argument(
        "--no-streaming-encoding",
        action="store_true",
        help="Encode each episode's videos after it is saved (~30x slower than streaming on Windows)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing output folder first")
    args = parser.parse_args()

    repo_id = args.repo_id or f"local/mimicgen_{args.input.stem}"
    root = args.root or dataset_root(repo_id)
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(root)

    cameras = parse_cameras(args.cameras)
    state_keys = [k.strip() for k in args.state_keys.split(",")]

    with h5py.File(args.input, "r") as h5:
        data = h5["data"]
        env_args = json.loads(data.attrs["env_args"])
        env_name = env_args["env_name"]
        fps = args.fps or int(env_args["env_kwargs"].get("control_freq", 20))
        task = args.task or data.attrs.get("task") or task_for_env(env_name)
        if task is None:
            raise SystemExit(f"No instruction known for env '{env_name}'. Pass --task.")

        demos = sorted_demos(data, args.mask, h5)[: args.max_episodes]
        features = build_features(data[demos[0]], cameras, state_keys, use_videos=not args.no_videos)

        print(f"env={env_name} fps={fps} demos={len(demos)} task={task!r}")
        print(f"writing {repo_id} -> {root}")

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=root,
            robot_type=args.robot_type,
            use_videos=not args.no_videos,
            image_writer_threads=4,
            streaming_encoding=not args.no_streaming_encoding,
        )

        for name in tqdm(demos, desc="episodes"):
            demo = data[name]
            obs = demo["obs"]
            images = {dst: obs[src][:] for src, dst in cameras.items()}
            if args.flip:
                images = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in images.items()}
            state = np.concatenate([obs[k][:] for k in state_keys], axis=1).astype(np.float32)
            actions = demo["actions"][:].astype(np.float32)

            for t in range(len(actions)):
                frame = {f"observation.images.{k}": v[t] for k, v in images.items()}
                frame["observation.state"] = state[t]
                frame["action"] = actions[t]
                frame["task"] = task
                dataset.add_frame(frame)
            dataset.save_episode()

        dataset.finalize()

    print(f"Done: {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames in {root}")


if __name__ == "__main__":
    main()
