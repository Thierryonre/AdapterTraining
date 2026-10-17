"""Recording simulated episodes and reading/writing them as robomimic-style HDF5.

Layout (compatible with at-convert):
  data/                       attrs: env_args (JSON), task, robot, scene (JSON: objects), task_spec (JSON)
    demo_i/                   attrs: source, boundaries (JSON phase -> [start, end)), success
      actions                 (T, 16) joint targets, LeRobot bi_openarm_follower names (degrees)
      states                  (T, S)  full MuJoCo state before each action
      obs/joint_pos           (T, 16) measured joints (degrees), same names
      obs/front_image         (T, H, W, 3)
      obs/{left,right}_eef_pos/_eef_quat/_gripper_opening, obs/object_<name>_pos/_quat
      datagen_info/{left,right}_eef_pose     (T, 4, 4) achieved gripper poses
      datagen_info/{left,right}_target_pose  (T, 4, 4) commanded gripper poses
      datagen_info/object_pose/<name>        (T, 4, 4)
      datagen_info/gripper_opening           (T, 2) commanded openings (left, right)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import mujoco
import numpy as np

from adapter_training.datagen.env import SIDES, SceneEnv


@dataclass
class Episode:
    obs: dict = field(default_factory=dict)  # key -> list
    actions: list = field(default_factory=list)
    states: list = field(default_factory=list)
    eef_pose: dict = field(default_factory=lambda: {s: [] for s in SIDES})
    target_pose: dict = field(default_factory=lambda: {s: [] for s in SIDES})
    object_pose: dict = field(default_factory=dict)  # name -> list of 4x4
    openings: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.actions)


class Recorder:
    """Call `before_step()` then `after_command(...)` around every control step."""

    def __init__(self, env: SceneEnv, render: bool = False):
        self.env = env
        self.render = render
        self.ep = Episode()

    def before_step(self) -> None:
        env = self.env
        obs = env.observation() if self.render else _obs_without_image(env)
        for k, v in obs.items():
            self.ep.obs.setdefault(k, []).append(v)
        self.ep.states.append(env.get_state())
        for s in SIDES:
            self.ep.eef_pose[s].append(env.eef_pose(s))
        for name, T in env.object_poses().items():
            self.ep.object_pose.setdefault(name, []).append(T)

    def after_command(self, targets: dict, arm_cmd: dict, openings: dict) -> None:
        for s in SIDES:
            T = targets.get(s)
            self.ep.target_pose[s].append(self.ep.eef_pose[s][-1] if T is None else T)
        self.ep.openings.append([openings[s] for s in SIDES])
        self.ep.actions.append(self.env.joint_command_deg(arm_cmd, openings))


def _obs_without_image(env: SceneEnv) -> dict:
    renderer, env.renderer = env.renderer, None
    try:
        return env.observation()
    finally:
        env.renderer = renderer


def render_images(env: SceneEnv, states: list, compositor=None, background=None, appearance=None) -> np.ndarray:
    """Re-render camera frames from saved states (only done for episodes worth keeping).

    With a compositor, the robot and objects are pasted onto the real `background` photo instead of
    the plain simulated scene, using `appearance` (domain randomisation) if given.
    """
    saved_geom, saved_mat = {}, {}
    if appearance is not None and appearance.object_rgba:
        for name, rgba in appearance.object_rgba.items():
            gid = env.model.geom(f"{name}_geom").id
            mid = env.model.geom_matid[gid]
            if mid >= 0:  # textured: tint the material (keeps the real texture recognisable)
                saved_mat[mid] = env.model.mat_rgba[mid].copy()
                env.model.mat_rgba[mid] = rgba
            else:
                saved_geom[gid] = env.model.geom_rgba[gid].copy()
                env.model.geom_rgba[gid] = rgba
    frames = []
    for state in states:
        env.set_state(state)
        if compositor is None:
            frames.append(env.observation()["front_image"])
        else:
            frames.append(compositor.compose(background, env.data, appearance))
    for gid, rgba in saved_geom.items():
        env.model.geom_rgba[gid] = rgba
    for mid, rgba in saved_mat.items():
        env.model.mat_rgba[mid] = rgba
    return np.stack(frames)


def open_output(path: Path, env: SceneEnv, task: str, task_spec: dict, extra_attrs: dict | None = None) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    h5 = h5py.File(path, "w")
    data = h5.create_group("data")
    env_args = {"env_name": "OpenArmSceneSim", "type": "mujoco", "env_kwargs": {"control_freq": env.sim.control_hz}}
    data.attrs["env_args"] = json.dumps(env_args)
    data.attrs["task"] = task
    data.attrs["robot"] = env.cfg.name
    data.attrs["scene"] = json.dumps([o.to_dict() for o in env.sim.objects])
    data.attrs["task_spec"] = json.dumps(task_spec)
    for k, v in (extra_attrs or {}).items():
        data.attrs[k] = v
    return h5


def write_episode(h5: h5py.File, ep: Episode, env: SceneEnv, images: np.ndarray | None, attrs: dict) -> str:
    name = f"demo_{len(h5['data'])}"
    demo = h5["data"].create_group(name)
    for k, v in attrs.items():
        demo.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
    names = [n for s in SIDES for n in env.cfg.arms[s].lerobot_names]
    demo.create_dataset("actions", data=np.asarray(ep.actions, np.float32)).attrs["names"] = names
    demo.create_dataset("states", data=np.asarray(ep.states))
    obs = demo.create_group("obs")
    for k, v in ep.obs.items():
        if k == "front_image":
            continue
        ds = obs.create_dataset(k, data=np.asarray(v, np.float32))
        if k == "joint_pos":
            ds.attrs["names"] = names
    if images is not None:
        obs.create_dataset("front_image", data=images, compression="gzip", chunks=(1, *images.shape[1:]))
    info = demo.create_group("datagen_info")
    for s in SIDES:
        info.create_dataset(f"{s}_eef_pose", data=np.asarray(ep.eef_pose[s]))
        info.create_dataset(f"{s}_target_pose", data=np.asarray(ep.target_pose[s]))
    objs = info.create_group("object_pose")
    for obj_name, poses in ep.object_pose.items():
        objs.create_dataset(obj_name, data=np.asarray(poses))
    info.create_dataset("gripper_opening", data=np.asarray(ep.openings))
    return name


@dataclass
class SourceDemo:
    name: str
    target_pose: dict  # side -> (T, 4, 4)
    eef_pose: dict
    object_pose: dict  # name -> (T, 4, 4)
    openings: np.ndarray  # (T, 2)
    boundaries: dict  # phase -> (start, end)
    initial_state: np.ndarray


def load_sources(path: Path) -> tuple[list[SourceDemo], dict]:
    sources = []
    with h5py.File(path, "r") as h5:
        meta = {k: h5["data"].attrs[k] for k in h5["data"].attrs}
        for name in sorted(h5["data"], key=lambda n: int(n.split("_")[1])):
            demo = h5["data"][name]
            info = demo["datagen_info"]
            sources.append(
                SourceDemo(
                    name=name,
                    target_pose={s: info[f"{s}_target_pose"][:] for s in SIDES},
                    eef_pose={s: info[f"{s}_eef_pose"][:] for s in SIDES},
                    object_pose={k: info["object_pose"][k][:] for k in info["object_pose"]},
                    openings=info["gripper_opening"][:],
                    boundaries={k: tuple(v) for k, v in json.loads(demo.attrs["boundaries"]).items()},
                    initial_state=demo["states"][0],
                )
            )
    return sources, meta


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)
