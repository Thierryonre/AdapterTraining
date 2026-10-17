"""Robot model: OpenArm v1.0 (bimanual, 7-DoF arms, parallel grippers) in MuJoCo.

The URDF comes from https://github.com/enactic/openarm_description (fetched into third_party/ by
`at-fetch-robot`). MuJoCo can't read the .dae visual meshes, so `prepare_openarm()` writes a patched
URDF that uses the STL collision meshes for both visuals and collisions.

Everything robot-specific lives in `RobotConfig`, so another robot means another config.
"""

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from adapter_training.common import DATA_DIR, PROJECT_ROOT

OPENARM_REPO = "https://github.com/enactic/openarm_description.git"
OPENARM_ROOT = PROJECT_ROOT / "third_party" / "openarm_description"
OPENARM_URDF = OPENARM_ROOT / "assets/robot/openarm_v1.0/urdf/example/v1.urdf"
OPENARM_MODEL_DIR = DATA_DIR / "robots" / "openarm_v1"


@dataclass
class ArmConfig:
    joints: list[str]
    tcp_body: str  # grasp point between the fingertips
    finger_joints: list[str]
    finger_range: tuple[float, float]  # metres per finger, closed -> open
    # LeRobot motor names, in the same order as `joints` + gripper.
    lerobot_names: list[str]
    # Gripper motor command (degrees) for closed / fully open. Check on your robot.
    gripper_motor_range: tuple[float, float] = (0.0, -65.0)
    # Extra joint limits (degrees) on top of the URDF, e.g. LeRobot's safe defaults.
    joint_limits_deg: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class RobotConfig:
    name: str
    model_path: Path
    root_body: str
    arms: dict[str, ArmConfig]  # "left" / "right"
    # Pose of the robot root in the table (AprilTag) frame: xyz metres, rpy radians.
    base_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Joint configuration (radians) used as the IK starting point and posture target.
    home: dict[str, float] = field(default_factory=dict)


def _openarm_arm(side: str) -> ArmConfig:
    # LeRobot's OpenArm follower defaults (lerobot/robots/openarm_follower/config_openarm_follower.py).
    limits = {
        "joint1": (-75.0, 75.0),
        "joint2": (-90.0, 9.0) if side == "left" else (-9.0, 90.0),
        "joint3": (-85.0, 85.0),
        "joint4": (0.0, 135.0),
        "joint5": (-85.0, 85.0),
        "joint6": (-40.0, 40.0),
        "joint7": (-80.0, 80.0),
    }
    return ArmConfig(
        joints=[f"openarm_{side}_joint{i}" for i in range(1, 8)],
        tcp_body=f"openarm_{side}_hand_tcp",
        finger_joints=[f"openarm_{side}_finger_joint1", f"openarm_{side}_finger_joint2"],
        finger_range=(0.0, 0.044),
        lerobot_names=[f"{side}_joint_{i}.pos" for i in range(1, 8)] + [f"{side}_gripper.pos"],
        joint_limits_deg={f"openarm_{side}_{k}": v for k, v in limits.items()},
    )


# Right-arm joint angles are the left arm's with these signs (found by matching mirrored FK).
OPENARM_MIRROR = np.array([-1, -1, -1, 1, -1, -1, -1])
# Forward-reaching pose (degrees): gripper in front of the shoulder, pointing along +x.
OPENARM_HOME_LEFT_DEG = np.array([-25.0, -45.0, 7.0, 76.0, -2.0, -1.0, 2.0])


def openarm_config() -> RobotConfig:
    home_left = np.radians(OPENARM_HOME_LEFT_DEG)
    home_right = home_left * OPENARM_MIRROR
    return RobotConfig(
        name="openarm_v1",
        model_path=OPENARM_MODEL_DIR / "openarm_v1.urdf",
        root_body="openarm_body_link0",
        arms={"left": _openarm_arm("left"), "right": _openarm_arm("right")},
        # Robot stands 30 cm behind the tag (negative x) facing +x; shoulders ~25 cm above the table.
        # Each arm reaches ~0.6 m from its shoulder, so the usable table area is roughly x in [-0.2, 0.3].
        base_xyz=(-0.30, 0.0, -0.45),
        base_rpy=(0.0, 0.0, 0.0),
        home={
            **{f"openarm_left_joint{i}": float(q) for i, q in enumerate(home_left, start=1)},
            **{f"openarm_right_joint{i}": float(q) for i, q in enumerate(home_right, start=1)},
        },
    )


def fetch_openarm() -> None:
    """Clone the OpenArm description repo into third_party/."""
    if OPENARM_ROOT.exists():
        print(f"{OPENARM_ROOT} already exists")
        return
    OPENARM_ROOT.parent.mkdir(parents=True, exist_ok=True)
    # schannel = Windows certificate store (works behind TLS-inspecting proxies); ignored elsewhere.
    cmd = ["git", "clone", "--depth", "1", OPENARM_REPO, str(OPENARM_ROOT)]
    if shutil.which("git") and Path("C:/").exists():
        cmd[1:1] = ["-c", "http.sslBackend=schannel"]
    subprocess.run(cmd, check=True)


def prepare_openarm(force: bool = False) -> Path:
    """Write a MuJoCo-loadable copy of the OpenArm v1 URDF with flat STL meshes."""
    out = OPENARM_MODEL_DIR / "openarm_v1.urdf"
    if out.exists() and not force:
        return out
    if not OPENARM_URDF.exists():
        fetch_openarm()

    mesh_dir = OPENARM_MODEL_DIR / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    urdf = re.sub(r"<!--.*?-->", "", OPENARM_URDF.read_text(), flags=re.S)

    def use_stl(match: re.Match) -> str:
        rel = match.group(1).removeprefix("package://openarm_description/")
        if rel.endswith(".dae"):
            rel = rel.replace("/visual/", "/collision/").removesuffix(".dae")
            rel += "_symp.stl" if "/openarm_v1.0/" in rel else ".stl"
        src = OPENARM_ROOT / rel
        shutil.copy(src, mesh_dir / src.name)
        return f'filename="{src.name}"'

    urdf = re.sub(r'filename="(package://[^"]+)"', use_stl, urdf)
    mujoco_ext = (
        '<mujoco><compiler meshdir="meshes" discardvisual="false" fusestatic="false" '
        'strippath="true" balanceinertia="true"/></mujoco>'
    )
    urdf = re.sub(r"(<robot[^>]*>)", rf"\1\n  {mujoco_ext}", urdf, count=1)
    out.write_text(urdf)
    return out


class RobotModel:
    """MuJoCo model of the robot standing at its configured pose in the table frame."""

    def __init__(self, cfg: RobotConfig, camera: dict | None = None, table_size=(0.6, 0.9), customize=None):
        """customize: optional callable(spec, robot_config) run before compiling (e.g. to add physics)."""
        self.cfg = cfg
        if cfg.name == "openarm_v1":
            prepare_openarm()
        spec = mujoco.MjSpec.from_file(str(cfg.model_path))
        self._place_root(spec)
        self._add_scene(spec, table_size)
        if camera is not None:
            self._add_camera(spec, **camera)
        if customize is not None:
            customize(spec, cfg)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self._apply_extra_limits()
        self.reset()

    # -- construction -------------------------------------------------------------------------------
    def _place_root(self, spec: mujoco.MjSpec) -> None:
        root = spec.body(self.cfg.root_body)
        root.pos = np.asarray(self.cfg.base_xyz, dtype=float)
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, np.asarray(self.cfg.base_rpy, dtype=float), "xyz")
        root.quat = quat

    @staticmethod
    def _add_scene(spec: mujoco.MjSpec, table_size) -> None:
        world = spec.worldbody
        spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
        spec.visual.global_.offwidth = 1920
        spec.visual.global_.offheight = 1080
        world.add_light(pos=[0, 0, 2.5], dir=[0, 0, -1], diffuse=[0.6, 0.6, 0.6])
        # Table top surface at z=0 (the AprilTag plane), centred on the tag.
        world.add_geom(
            name="table",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[table_size[0] / 2, table_size[1] / 2, 0.01],
            pos=[0, 0, -0.01],
            rgba=[0.75, 0.7, 0.62, 1],
            contype=1,  # lets the IK keep the arms above the table (this model is never physics-stepped)
            conaffinity=1,
            group=2,
        )
        # Tag marker, for visual alignment against the real video.
        world.add_geom(
            name="tag",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.03, 0.03, 0.0005],
            pos=[0, 0, 0.0005],
            rgba=[0.1, 0.1, 0.1, 1],
            contype=0,
            conaffinity=0,
            group=2,
        )

    @staticmethod
    def _add_camera(spec: mujoco.MjSpec, pos, xyaxes, fovy) -> None:
        spec.worldbody.add_camera(name="scene_cam", pos=pos, xyaxes=xyaxes, fovy=fovy)

    def _apply_extra_limits(self) -> None:
        for arm in self.cfg.arms.values():
            for joint, (lo, hi) in arm.joint_limits_deg.items():
                jid = self.model.joint(joint).id
                urdf_lo, urdf_hi = self.model.jnt_range[jid]
                self.model.jnt_range[jid] = [max(urdf_lo, np.radians(lo)), min(urdf_hi, np.radians(hi))]

    # -- state ----------------------------------------------------------------------------------------
    def reset(self) -> None:
        self.data.qpos[:] = 0
        for joint, q in self.cfg.home.items():
            self.set_joint(joint, q)
        mujoco.mj_forward(self.model, self.data)

    def qpos_addr(self, joint: str) -> int:
        return self.model.jnt_qposadr[self.model.joint(joint).id]

    def set_joint(self, joint: str, value: float) -> None:
        jid = self.model.joint(joint).id
        lo, hi = self.model.jnt_range[jid]
        self.data.qpos[self.model.jnt_qposadr[jid]] = np.clip(value, lo, hi)

    def set_gripper(self, side: str, opening: float) -> None:
        """opening in [0, 1]: 0 = closed, 1 = fully open."""
        arm = self.cfg.arms[side]
        lo, hi = arm.finger_range
        for joint in arm.finger_joints:
            self.set_joint(joint, lo + np.clip(opening, 0, 1) * (hi - lo))

    def arm_qpos(self, side: str) -> np.ndarray:
        return np.array([self.data.qpos[self.qpos_addr(j)] for j in self.cfg.arms[side].joints])

    def tcp_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        body = self.data.body(self.cfg.arms[side].tcp_body)
        return body.xpos.copy(), body.xmat.reshape(3, 3).copy()


def main() -> None:
    """at-fetch-robot: download the OpenArm description and build the MuJoCo model."""
    fetch_openarm()
    path = prepare_openarm(force=True)
    model = mujoco.MjModel.from_xml_path(str(path))
    print(f"OpenArm v1 ready: {path} ({model.nq} joints)")
