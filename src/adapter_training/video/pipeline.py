"""Human demonstration videos -> OpenArm joint trajectories -> robomimic-style HDF5.

    uv run at-video2robot data/videos/handover_*.mp4 --calib data/calib/camera.json \
        --task "Pick up the block with the left hand and pass it to the right hand."

Outputs (default outputs/video2robot/<name>/):
  demos.hdf5          one demo per video: joint trajectories (LeRobot OpenArm names), gripper poses,
                      camera frames -- feed it to `at-convert` (command printed at the end)
  <video>_check.mp4   original video with tracked hands | robot following them in MuJoCo, same viewpoint
  <video>_tracks.npz  raw hand tracks (re-run later steps with --reuse-tracks)
  summary.json        tracking coverage and IK errors per video
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import mujoco
import numpy as np
from tqdm import tqdm

import adapter_training  # noqa: F401  (TLS setup)
from adapter_training.common import OUTPUTS_DIR
from adapter_training.video.camera import Calibration, draw_table_axes, track_object_tag
from adapter_training.video.hands import load_tracks, make_tracker, save_tracks, track_video
from adapter_training.video.ik import BimanualIK
from adapter_training.video.retarget import GripperTrack, retarget_hand, to_table_frame
from adapter_training.video.robot import RobotModel, openarm_config

SIDES = ("left", "right")
HAND_COLORS = {"left": (0, 220, 0), "right": (255, 60, 60)}


# -- video io -------------------------------------------------------------------------------------------
def read_video(path: Path, stride: int = 1, max_frames: int | None = None, max_width: int = 1280):
    """RGB frames (downscaled to max_width to bound memory) and the effective fps."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and len(frames) >= max_frames):
            break
        if i % stride == 0:
            if frame.shape[1] > max_width:
                scale = max_width / frame.shape[1]
                frame = cv2.resize(frame, (max_width, round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames, fps / stride


class VideoWriter:
    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, rgb: np.ndarray) -> None:
        self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self.writer.release()


# -- trajectory -----------------------------------------------------------------------------------------
@dataclass
class RobotTrajectory:
    qpos: np.ndarray  # (T, nq) full MuJoCo qpos
    joint_deg: np.ndarray  # (T, 16) LeRobot order: left joints 1-7, left gripper, right ..., degrees
    eef: dict  # side -> {"pos": (T,3), "quat": (T,4), "opening": (T,)}
    pos_error: dict  # side -> (T,) metres (NaN where no target)
    rot_error: dict


def solve_trajectory(robot: RobotModel, grippers: dict[str, GripperTrack]) -> RobotTrajectory:
    ik = BimanualIK(robot)
    robot.reset()
    q = robot.data.qpos.copy()
    T = len(next(iter(grippers.values())).pos)
    qpos = np.zeros((T, robot.model.nq))
    pos_err = {s: np.full(T, np.nan) for s in SIDES}
    rot_err = {s: np.full(T, np.nan) for s in SIDES}
    for t in tqdm(range(T), desc="IK", leave=False):
        targets = {}
        for side in SIDES:
            g = grippers[side]
            targets[side] = None if np.isnan(g.pos[t]).any() else (g.pos[t], g.rot[t])
        res = ik.solve(targets, q)
        q = res.qpos
        for side in SIDES:
            if targets[side] is not None:
                pos_err[side][t], rot_err[side][t] = res.pos_error[side], res.rot_error[side]
        robot.data.qpos[:] = q
        for side in SIDES:
            robot.set_gripper(side, grippers[side].opening[t])
        qpos[t] = robot.data.qpos

    joint_deg = np.zeros((T, 16))
    eef = {s: {"pos": np.zeros((T, 3)), "quat": np.zeros((T, 4)), "opening": grippers[s].opening.copy()} for s in SIDES}
    for t in range(T):
        robot.data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(robot.model, robot.data)
        for k, side in enumerate(SIDES):
            arm = robot.cfg.arms[side]
            joints = [robot.data.qpos[robot.qpos_addr(j)] for j in arm.joints]
            closed, opened = arm.gripper_motor_range
            gripper = closed + grippers[side].opening[t] * (opened - closed)
            joint_deg[t, k * 8 : k * 8 + 7] = np.degrees(joints)
            joint_deg[t, k * 8 + 7] = gripper
            pos, rot = robot.tcp_pose(side)
            eef[side]["pos"][t] = pos
            mujoco.mju_mat2Quat(eef[side]["quat"][t], rot.ravel())
    return RobotTrajectory(qpos, joint_deg, eef, pos_err, rot_err)


# -- debug video ------------------------------------------------------------------------------------------
def draw_hands(frame: np.ndarray, tracks: dict, t: int, grippers: dict, calib: Calibration) -> np.ndarray:
    out = draw_table_axes(frame, calib)
    T_cam_table = np.linalg.inv(calib.T_np)
    rvec = cv2.Rodrigues(T_cam_table[:3, :3])[0]
    for side in SIDES:
        kp2d = tracks[side]["kp2d"][t]
        if not np.isnan(kp2d).any():
            for p in kp2d.astype(int):
                cv2.circle(out, tuple(p), 3, HAND_COLORS[side], -1)
        target = grippers[side].pos[t]
        if not np.isnan(target).any():  # gripper target projected back into the image
            p = cv2.projectPoints(target[None], rvec, T_cam_table[:3, 3], calib.K_np, calib.dist_np)[0].ravel()
            radius = int(4 + 10 * grippers[side].opening[t])
            cv2.circle(out, (int(p[0]), int(p[1])), radius, HAND_COLORS[side], 2)
    return out


def render_check_video(
    path: Path, frames: list, tracks: dict, grippers: dict, traj: RobotTrajectory, robot: RobotModel, calib, fps
) -> None:
    h, w = frames[0].shape[:2]
    scale = min(1.0, 640 / w)
    rw, rh = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    renderer = mujoco.Renderer(robot.model, rh, rw)
    writer = VideoWriter(path, fps, (rw * 2, rh))
    for t, frame in enumerate(frames):
        left = cv2.resize(draw_hands(frame, tracks, t, grippers, calib), (rw, rh))
        robot.data.qpos[:] = traj.qpos[t]
        mujoco.mj_forward(robot.model, robot.data)
        renderer.update_scene(robot.data, camera="scene_cam")
        right = renderer.render()
        txt = " ".join(
            f"{s[0].upper()}:{traj.pos_error[s][t] * 100:.1f}cm" if not np.isnan(traj.pos_error[s][t]) else f"{s[0].upper()}:--"
            for s in SIDES
        )
        cv2.putText(right, f"t={t / fps:5.2f}s IK err {txt}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(np.concatenate([left, right], axis=1))
    writer.close()
    renderer.close()


# -- export -------------------------------------------------------------------------------------------------
def write_demo(
    h5: h5py.File, name: str, traj: RobotTrajectory, frames: list, image_size, source: str, robot, object_poses=None
) -> None:
    demo = h5["data"].create_group(name)
    demo.attrs["source_video"] = source
    demo.attrs["num_samples"] = len(frames)
    obs = demo.create_group("obs")
    images = np.stack([cv2.resize(f, image_size, interpolation=cv2.INTER_AREA) for f in frames])
    obs.create_dataset("front_image", data=images, compression="gzip", chunks=(1, *images.shape[1:]))
    names = [n for side in SIDES for n in robot.cfg.arms[side].lerobot_names]
    ds = obs.create_dataset("joint_pos", data=traj.joint_deg.astype(np.float32))
    ds.attrs["names"] = names
    for side in SIDES:
        obs.create_dataset(f"{side}_eef_pos", data=traj.eef[side]["pos"].astype(np.float32))
        obs.create_dataset(f"{side}_eef_quat", data=traj.eef[side]["quat"].astype(np.float32))
        obs.create_dataset(f"{side}_gripper_opening", data=traj.eef[side]["opening"][:, None].astype(np.float32))
    if object_poses is not None:  # (T, 4, 4) table frame, NaN where the object tag wasn't seen
        quats = np.full((len(object_poses), 4), np.nan)
        for t, T in enumerate(object_poses):
            if not np.isnan(T).any():
                mujoco.mju_mat2Quat(quats[t], T[:3, :3].ravel())
        obs.create_dataset("object_pos", data=object_poses[:, :3, 3].astype(np.float32))
        obs.create_dataset("object_quat", data=quats.astype(np.float32))
    # Action = joint position target for the next step (what LeRobot's OpenArm driver takes).
    actions = np.concatenate([traj.joint_deg[1:], traj.joint_deg[-1:]], axis=0).astype(np.float32)
    ds = demo.create_dataset("actions", data=actions)
    ds.attrs["names"] = names


def open_hdf5(path: Path, fps: float, task: str, robot_name: str, append: bool) -> h5py.File:
    if path.exists() and append:
        h5 = h5py.File(path, "a")
        if abs(json.loads(h5["data"].attrs["env_args"])["env_kwargs"]["control_freq"] - fps) > 1e-3:
            raise SystemExit(f"{path} was written at a different fps")
        return h5
    h5 = h5py.File(path, "w")
    data = h5.create_group("data")
    env_args = {"env_name": "VideoDemo", "type": "video2robot", "env_kwargs": {"control_freq": fps}}
    data.attrs["env_args"] = json.dumps(env_args)
    data.attrs["task"] = task
    data.attrs["robot"] = robot_name
    return h5


def next_demo_index(h5: h5py.File) -> int:
    existing = [int(k.split("_")[1]) for k in h5["data"].keys()]
    return max(existing, default=-1) + 1


# -- main -------------------------------------------------------------------------------------------------
def process_video(video: Path, args, calib: Calibration, robot: RobotModel, tracker_factory, out_dir: Path):
    frames, fps = read_video(video, args.stride, args.max_frames)
    h, w = frames[0].shape[:2]
    if (w, h) != (calib.width, calib.height):
        calib = calib.scaled(w, h)

    tracks_path = out_dir / f"{video.stem}_tracks.npz"
    if args.reuse_tracks and tracks_path.exists():
        tracks, _ = load_tracks(tracks_path)
    else:
        tracker = tracker_factory(fps)
        tracks = track_video(tqdm(frames, desc=f"tracking {video.name}", leave=False), tracker, calib.K_np, calib.dist_np)
        if hasattr(tracker, "landmarker"):
            tracker.landmarker.close()
        save_tracks(tracks_path, tracks, fps)

    grippers = {}
    for side in SIDES:
        kp_table = to_table_frame(tracks[side]["kp_cam"], calib.T_np)
        grippers[side] = retarget_hand(kp_table, tracks[side]["valid"], max_gap=int(args.max_gap_s * fps))

    object_poses = None
    if args.object_tag_id is not None:
        object_poses = track_object_tag(
            frames, calib, args.object_tag_id, args.object_tag_size, args.object_size[2], args.tag_family
        )

    traj = solve_trajectory(robot, grippers)
    if not args.no_check_video:
        render_check_video(out_dir / f"{video.stem}_check.mp4", frames, tracks, grippers, traj, robot, calib, fps)

    def stats(x):
        x = x[~np.isnan(x)]
        return {"median": float(np.median(x)), "p95": float(np.percentile(x, 95))} if len(x) else None

    summary = {
        "frames": len(frames),
        "fps": fps,
        **{
            side: {
                "tracked_fraction": float(tracks[side]["valid"].mean()),
                "ik_pos_error_m": stats(traj.pos_error[side]),
                "ik_rot_error_rad": stats(traj.rot_error[side]),
                "frames_over_2cm": int(np.nansum(traj.pos_error[side] > 0.02)),
            }
            for side in SIDES
        },
    }
    if object_poses is not None:
        seen = ~np.isnan(object_poses[:, 0, 0])
        summary["object_tag_seen_fraction"] = float(seen.mean())
        summary["object_initial_xyz"] = object_poses[seen][0, :3, 3].round(4).tolist() if seen.any() else None
    return frames, fps, traj, summary, object_poses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--calib", type=Path, required=True, help="Calibration JSON from at-calibrate")
    parser.add_argument("--task", required=True, help="Language instruction stored with every demo")
    parser.add_argument("--out", type=Path, help="Output folder (default outputs/video2robot/<first video stem>)")
    parser.add_argument("--tracker", default="mediapipe", choices=["mediapipe", "wilor"])
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame (e.g. 2 turns 30 fps into 15)")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-gap-s", type=float, default=0.3, help="Interpolate tracking dropouts up to this long")
    parser.add_argument(
        "--image-size", default="auto", help="Exported frame size WxH, or auto = 320 wide with the camera's aspect"
    )
    parser.add_argument("--base-xyz", help="Robot base position in the table frame, e.g. -0.30,0,-0.45")
    parser.add_argument("--reuse-tracks", action="store_true", help="Skip tracking if <video>_tracks.npz exists")
    parser.add_argument("--no-check-video", action="store_true")
    parser.add_argument("--append", action="store_true", help="Add demos to an existing demos.hdf5")
    obj = parser.add_argument_group("object tracking (needed for Part 2 / at-datagen)")
    obj.add_argument("--object-tag-id", type=int, help="ID of the AprilTag on the object's top face")
    obj.add_argument("--object-tag-size", type=float, default=0.04, help="Object tag black square, metres")
    obj.add_argument("--object-size", default="0.14,0.05,0.04", help="Object box size x,y,z in metres")
    obj.add_argument("--tag-family", default="36h11")
    return parser


def main() -> None:
    run(build_parser().parse_args())


def run(args) -> dict:
    """Process videos into demos.hdf5. Returns per-video summaries."""
    if isinstance(args.object_size, str):
        args.object_size = tuple(float(v) for v in args.object_size.split(","))

    out_dir = args.out or OUTPUTS_DIR / "video2robot" / args.videos[0].stem
    out_dir.mkdir(parents=True, exist_ok=True)

    calib = Calibration.load(args.calib)
    if args.image_size == "auto":  # keep the camera's aspect ratio so frames match the live camera
        image_size = (320, round(320 * calib.height / calib.width / 2) * 2)
    else:
        image_size = tuple(int(v) for v in args.image_size.lower().split("x"))
    cfg = openarm_config()
    if args.base_xyz:
        cfg.base_xyz = tuple(float(v) for v in args.base_xyz.split(","))

    tracker_factory = lambda fps: make_tracker(args.tracker, fps)  # noqa: E731
    h5_path = out_dir / "demos.hdf5"
    summaries = {}
    h5 = None
    for video in args.videos:
        frames0, _ = read_video(video, max_frames=1)
        h, w = frames0[0].shape[:2]
        robot = RobotModel(cfg, camera=calib.scaled(w, h).mujoco_camera())
        frames, fps, traj, summary, object_poses = process_video(video, args, calib, robot, tracker_factory, out_dir)
        if h5 is None:
            h5 = open_hdf5(h5_path, fps, args.task, cfg.name, args.append)
            if args.object_tag_id is not None:
                h5["data"].attrs["object_size"] = args.object_size
        name = f"demo_{next_demo_index(h5)}"
        write_demo(h5, name, traj, frames, image_size, str(video), robot, object_poses)
        summaries[video.name] = {"demo": name, **summary}
        print(f"{video.name} -> {name}: " + json.dumps({s: summary[s] for s in SIDES}))
    h5.close()

    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nWrote {h5_path}")
    print(
        "Next: uv run at-convert "
        f'"{h5_path}" --repo-id local/{out_dir.name} --cameras front_image:front --state-keys joint_pos '
        f'--robot-type bi_openarm_follower --task "{args.task}"'
    )
    return summaries


if __name__ == "__main__":
    main()
