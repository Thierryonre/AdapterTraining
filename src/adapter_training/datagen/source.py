"""Turn Part 1 video demos into simulation-validated source demos (Part 2, steps 2-4).

For each video demo:
  1. place the object where its tag was seen at the start of the video
  2. replay the gripper trajectories in physics, nudging each grasp onto the object
     (hand tracking is off by centimetres; without this most grasps miss)
  3. keep the demo only if both grasps hold and the handover succeeds
  4. record what actually happened in simulation (poses, object pose, states) for generation
"""

from dataclasses import dataclass

import h5py
import numpy as np

from adapter_training.datagen.dataset import Episode, Recorder, quat_to_mat
from adapter_training.datagen.env import SIDES, HandoverEnv, pose_matrix
from adapter_training.datagen.task import gripper_events


@dataclass
class VideoDemo:
    name: str
    eef: dict  # side -> (T, 4, 4) gripper poses from Part 1
    openings: np.ndarray  # (T, 2)
    arm_qpos: dict  # side -> (7,) radians at frame 0
    object_pose: np.ndarray | None  # (4, 4) initial pose, or None if the tag was never seen


def load_video_demos(path, object_height: float) -> tuple[list[VideoDemo], dict]:
    demos = []
    with h5py.File(path, "r") as h5:
        meta = {k: h5["data"].attrs[k] for k in h5["data"].attrs}
        for name in sorted(h5["data"], key=lambda n: int(n.split("_")[1])):
            obs = h5["data"][name]["obs"]
            eef = {}
            for s in SIDES:
                pos, quat = obs[f"{s}_eef_pos"][:], obs[f"{s}_eef_quat"][:]
                eef[s] = np.stack([pose_matrix(p, quat_to_mat(q)) for p, q in zip(pos, quat)])
            openings = np.concatenate([obs["left_gripper_opening"][:], obs["right_gripper_opening"][:]], axis=1)
            joints = np.radians(obs["joint_pos"][0])
            arm_qpos = {"left": joints[0:7], "right": joints[8:15]}
            obj = None
            if "object_pos" in obs:
                obj = initial_object_pose(obs["object_pos"][:], obs["object_quat"][:], openings[:, 0], object_height)
            demos.append(VideoDemo(name, eef, openings, arm_qpos, obj))
    return demos, meta


def initial_object_pose(pos: np.ndarray, quat: np.ndarray, left_opening: np.ndarray, height: float) -> np.ndarray | None:
    """Robust resting pose: median over tag detections before the first grasp, flat on the table."""
    events = gripper_events(left_opening)
    end = events["close"][0] if events["close"] else len(pos)
    seen = [t for t in range(end) if not np.isnan(pos[t]).any()]
    if not seen:
        return None
    xy = np.median(pos[seen, :2], axis=0)
    yaws = [np.arctan2(quat_to_mat(quat[t])[1, 0], quat_to_mat(quat[t])[0, 0]) for t in seen]
    yaw = float(np.angle(np.mean(np.exp(1j * np.array(yaws)))))  # circular mean
    c, s = np.cos(yaw), np.sin(yaw)
    return pose_matrix(np.array([xy[0], xy[1], height / 2]), np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))


def grasp_offset(env: HandoverEnv, grasp: np.ndarray, margin: float = 0.015) -> np.ndarray:
    """Shift that turns a roughly-placed source grasp (4x4 gripper pose) into a secure one.

    The grasp point is slid along the finger-closing axis (gripper y) onto the object's centre plane,
    so the fingers meet opposite faces symmetrically, then kept `margin` inside the object's edges.
    """
    O = env.object_pose()
    R, centre = O[:3, :3], O[:3, 3]
    half = np.maximum(np.asarray(env.sim.obj.size) / 2 - margin, 0.0)
    closing = R.T @ grasp[:3, 1]  # closing axis in object coordinates
    local = R.T @ (grasp[:3, 3] - centre)
    local -= closing * np.dot(local, closing)
    return R @ np.clip(local, -half, half) + centre - grasp[:3, 3]


def binary_openings(opening: np.ndarray) -> np.ndarray:
    """Fully open until each close event, fully closed until the next open event (as in MimicGen).

    The human's pinch width only maps to ~85% open, which leaves the robot's wider fingers too
    little clearance around the object on approach.
    """
    out = np.ones_like(opening)
    events = gripper_events(opening)
    changes = sorted([(t, 0.0) for t in events["close"]] + [(t, 1.0) for t in events["open"]])
    t_prev, value = 0, 1.0
    for t, new in changes:
        out[t_prev:t] = value
        t_prev, value = t, new
    out[t_prev:] = value
    return out


def replay_with_correction(
    env: HandoverEnv, demo: VideoDemo, object_pose: np.ndarray, ramp_s: float = 1.0, settle_s: float = 0.5
) -> tuple[Episode, dict]:
    """Replay a video demo in physics. Returns the recorded episode and diagnostics."""
    fps = env.sim.control_hz
    ramp = max(1, int(ramp_s * fps))
    T = len(demo.openings)
    env.reset(object_pose=object_pose)
    robot_q = env.data.qpos[: env.obj_qadr].copy()  # start the arms where the video demo starts
    for s in SIDES:
        for j, q in zip(env.cfg.arms[s].joints, demo.arm_qpos[s]):
            robot_q[env.robot.qpos_addr(j)] = q
    env.reset(object_pose=object_pose, arm_qpos=robot_q)

    gripper_cmd = np.stack([binary_openings(demo.openings[:, k]) for k in range(len(SIDES))], axis=1)
    events = {s: gripper_events(demo.openings[:, k]) for k, s in enumerate(SIDES)}
    closes = {s: (events[s]["close"][0] if events[s]["close"] else None) for s in SIDES}
    opens = {s: next((t for t in events[s]["open"] if closes[s] is not None and t > closes[s]), None) for s in SIDES}
    offsets = {s: np.zeros(3) for s in SIDES}
    frozen = {s: None for s in SIDES}
    diag = {"grasp_correction_m": {}, "grasped": {}}

    rec = Recorder(env)
    for t in range(T + int(settle_s * fps)):
        k = min(t, T - 1)
        targets, openings = {}, {}
        for i, s in enumerate(SIDES):
            tc, to = closes[s], opens[s]
            if tc is not None and t <= tc:
                if t >= tc - ramp:  # ramp the correction in, tracking the object's current pose
                    c = grasp_offset(env, demo.eef[s][tc])
                    offsets[s] = c * (t - (tc - ramp)) / ramp
                if t == tc:
                    frozen[s] = offsets[s].copy()
                    diag["grasp_correction_m"][s] = float(np.linalg.norm(frozen[s]))
            elif frozen[s] is not None:
                fade = 1.0 if to is None or t < to else max(0.0, 1 - (t - to) / ramp)
                offsets[s] = frozen[s] * fade
            T_target = demo.eef[s][k].copy()
            T_target[:3, 3] += offsets[s]
            targets[s] = T_target
            openings[s] = float(gripper_cmd[k, i])
        rec.before_step()
        arm = env.step_ee(targets, openings)
        rec.after_command(targets, arm, openings)
        for s in SIDES:  # did the grasp hold shortly after closing?
            if closes[s] is not None and t == closes[s] + int(0.4 * fps):
                diag["grasped"][s] = env.grasped(s)
    diag["success"] = env.is_success()
    return rec.ep, diag
