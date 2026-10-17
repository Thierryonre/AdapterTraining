"""Turn Part 1 video demos into simulation-validated source demos.

For each video demo:
  1. place the objects where perception (or a tag) found them at the start of the video
  2. replay the gripper trajectories in physics, nudging each grasp onto the object being grasped
     (hand tracking is off by centimetres; without this most grasps miss)
  3. keep the demo only if every grasp holds and every release lets go
  4. record what actually happened in simulation (poses, object poses, states) for task inference
"""

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from adapter_training.datagen.dataset import Episode, Recorder, quat_to_mat
from adapter_training.datagen.env import SIDES, SceneEnv, SceneObject, pose_matrix, yaw_pose
from adapter_training.datagen.task import gripper_events, nearest_object


@dataclass
class VideoDemo:
    name: str
    source_video: str
    eef: dict  # side -> (T, 4, 4) gripper poses from Part 1
    openings: np.ndarray  # (T, 2)
    arm_qpos: dict  # side -> (7,) radians at frame 0
    objects: list[SceneObject] | None  # scene objects (shape models) for this clip
    object_poses: dict | None  # name -> 4x4 initial pose in the table frame


def load_scene(path: Path) -> tuple[list[SceneObject], dict]:
    """Scene file written by object perception: {"objects": [{...SceneObject..., "pose": 4x4}, ...]}."""
    data = json.loads(Path(path).read_text())
    objects = [SceneObject.from_dict(o) for o in data["objects"]]
    poses = {o["name"]: np.asarray(o["pose"]) for o in data["objects"]}
    return objects, poses


def load_video_demos(path, scenes_dir: Path | None = None, tag_object: SceneObject | None = None) -> list[VideoDemo]:
    """Demos from at-video2robot. Objects come from `scenes_dir/<video stem>.json` (perception), or from the
    object tag recorded in the HDF5 (`tag_object` gives its box model)."""
    demos = []
    with h5py.File(path, "r") as h5:
        for name in sorted(h5["data"], key=lambda n: int(n.split("_")[1])):
            demo = h5["data"][name]
            obs = demo["obs"]
            eef = {}
            for s in SIDES:
                pos, quat = obs[f"{s}_eef_pos"][:], obs[f"{s}_eef_quat"][:]
                eef[s] = np.stack([pose_matrix(p, quat_to_mat(q)) for p, q in zip(pos, quat)])
            openings = np.concatenate([obs["left_gripper_opening"][:], obs["right_gripper_opening"][:]], axis=1)
            joints = np.radians(obs["joint_pos"][0])
            source = str(demo.attrs.get("source_video", name))
            objects, poses = None, None
            scene_file = scenes_dir / f"{Path(source).stem}.json" if scenes_dir else None
            if scene_file is not None and scene_file.exists():
                objects, poses = load_scene(scene_file)
            elif "object_pos" in obs and tag_object is not None:
                T = initial_object_pose(obs["object_pos"][:], obs["object_quat"][:], openings, tag_object.size[2])
                if T is not None:
                    objects, poses = [tag_object], {tag_object.name: T}
            demos.append(VideoDemo(name, source, eef, openings, {"left": joints[0:7], "right": joints[8:15]}, objects, poses))
    return demos


def initial_object_pose(pos: np.ndarray, quat: np.ndarray, openings: np.ndarray, height: float) -> np.ndarray | None:
    """Robust resting pose from tag detections before the first grasp, flat on the table."""
    first_close = [gripper_events(openings[:, k])["close"] for k in range(openings.shape[1])]
    end = min([c[0] for c in first_close if c], default=len(pos))
    seen = [t for t in range(end) if not np.isnan(pos[t]).any()]
    if not seen:
        return None
    xy = np.median(pos[seen, :2], axis=0)
    yaws = [np.arctan2(quat_to_mat(quat[t])[1, 0], quat_to_mat(quat[t])[0, 0]) for t in seen]
    yaw = float(np.angle(np.mean(np.exp(1j * np.array(yaws)))))  # circular mean
    return yaw_pose(xy[0], xy[1], height / 2, yaw)


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


def grasp_offset(env: SceneEnv, grasp: np.ndarray, name: str, margin: float = 0.015) -> np.ndarray:
    """Shift that turns a roughly-placed source grasp (4x4 gripper pose) into a secure one on `name`.

    The grasp point is slid along the finger-closing axis (gripper y) onto the object's centre plane,
    so the fingers meet opposite faces symmetrically, then kept `margin` inside its bounding box.
    """
    O = env.object_pose(name)
    R, centre = O[:3, :3], O[:3, 3]
    half = np.maximum(env.object_size(name) / 2 - margin, 0.0)
    closing = R.T @ grasp[:3, 1]  # closing axis in object coordinates
    local = R.T @ (grasp[:3, 3] - centre)
    local -= closing * np.dot(local, closing)
    return R @ np.clip(local, -half, half) + centre - grasp[:3, 3]


def replay_with_correction(
    env: SceneEnv, demo: VideoDemo, ramp_s: float = 1.0, settle_s: float = 0.5
) -> tuple[Episode, dict]:
    """Replay a video demo in physics. Returns the recorded episode and diagnostics."""
    fps = env.sim.control_hz
    ramp = max(1, int(ramp_s * fps))
    T = len(demo.openings)
    env.reset(object_pose=demo.object_poses)
    robot_q = env.data.qpos[: env.robot_qpos_size].copy()  # start the arms where the video demo starts
    for s in SIDES:
        for j, q in zip(env.cfg.arms[s].joints, demo.arm_qpos[s]):
            robot_q[env.robot.qpos_addr(j)] = q
    env.reset(object_pose=demo.object_poses, arm_qpos=robot_q)

    gripper_cmd = np.stack([binary_openings(demo.openings[:, k]) for k in range(len(SIDES))], axis=1)
    # Every grasp of every arm: (close frame, matching open frame or None).
    grasps = {s: [] for s in SIDES}
    for k, s in enumerate(SIDES):
        ev = gripper_events(demo.openings[:, k])
        for tc in ev["close"]:
            to = next((t for t in ev["open"] if t > tc), None)
            grasps[s].append([tc, to, None, None])  # close, open, target object, frozen offset
    offsets = {s: np.zeros(3) for s in SIDES}
    diag = {"grasps": [], "releases": []}
    sizes = {n: env.object_size(n) for n in env.object_names}

    rec = Recorder(env)
    for t in range(T + int(settle_s * fps)):
        k = min(t, T - 1)
        targets, openings = {}, {}
        for i, s in enumerate(SIDES):
            offsets[s] = np.zeros(3)
            for g in grasps[s]:
                tc, to, target, frozen = g
                if tc - ramp <= t <= tc:  # ramp the correction in, tracking the object's current pose
                    if target is None:
                        g[2] = target = nearest_object(demo.eef[s][tc][:3, 3], env.object_poses(), sizes)
                    offsets[s] = grasp_offset(env, demo.eef[s][tc], target) * (t - (tc - ramp)) / ramp
                    if t == tc:
                        g[3] = offsets[s].copy()
                elif frozen is not None and t > tc:
                    fade = 1.0 if to is None or t < to else max(0.0, 1 - (t - to) / ramp)
                    if fade > 0:
                        offsets[s] = frozen * fade
            T_target = demo.eef[s][k].copy()
            T_target[:3, 3] += offsets[s]
            targets[s] = T_target
            openings[s] = float(gripper_cmd[k, i])
        rec.before_step()
        arm = env.step_ee(targets, openings)
        rec.after_command(targets, arm, openings)
        for s in SIDES:  # did each grasp hold, and did each release let go?
            for tc, to, target, frozen in grasps[s]:
                if target is not None and t == tc + int(0.4 * fps):
                    diag["grasps"].append({"side": s, "object": target, "frame": tc, "held": env.grasped(s, target)})
                if target is not None and to is not None and t == to + int(0.4 * fps):
                    diag["releases"].append({"side": s, "object": target, "frame": to, "let_go": not env.grasped(s, target)})
    on_table = all(env.object_pose(n)[2, 3] > -0.02 for n in env.object_names)
    diag["objects_on_table"] = on_table
    diag["ok"] = bool(
        diag["grasps"] and all(g["held"] for g in diag["grasps"]) and all(r["let_go"] for r in diag["releases"]) and on_table
    )
    return rec.ep, diag
