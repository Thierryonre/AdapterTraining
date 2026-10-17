"""Physics simulation of the OpenArm handover task in MuJoCo.

The same OpenArm model as the video pipeline, plus:
  - position-controlled joints (gravity-compensated arms) and position-controlled fingers
  - a collidable table and a free "bar" object
  - contact filtering: robot links don't collide with each other, only with the table and objects
    (so the two grippers can pass through each other -- keep handover grasps apart)

Actions follow LeRobot's OpenArm convention: 7 joint targets per arm (radians internally,
degrees when exported) + a gripper opening in [0, 1] per arm.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

from adapter_training.video.ik import BimanualIK
from adapter_training.video.robot import RobotConfig, RobotModel, openarm_config

SIDES = ("left", "right")
ROBOT_GEOM = (2, 1)  # contype, conaffinity: robot geoms never collide with each other
WORLD_GEOM = (1, 3)  # table/objects collide with the robot and with each other


@dataclass
class ObjectSpec:
    name: str = "bar"
    size: tuple[float, float, float] = (0.14, 0.05, 0.04)  # full x, y, z extent in metres
    mass: float = 0.08
    friction: float = 1.2
    rgba: tuple[float, float, float, float] = (0.85, 0.35, 0.1, 1.0)


@dataclass
class SimConfig:
    control_hz: float = 30.0
    substeps: int = 16
    arm_kp: float = 400.0
    arm_kv: float = 40.0
    finger_kp: float = 1500.0
    finger_kv: float = 30.0
    finger_force: float = 60.0
    image_size: tuple[int, int] = (320, 240)  # (width, height); None-like (0, 0) disables rendering
    shadows: bool = False  # ~3x slower to render on integrated graphics; cheap on a real GPU
    obj: ObjectSpec = field(default_factory=ObjectSpec)


def _robot_body_names(spec: mujoco.MjSpec, root: str) -> list[str]:
    names, stack = [], [spec.body(root)]
    while stack:
        body = stack.pop()
        names.append(body.name)
        stack.extend(body.bodies)
    return names


def build_physics(spec: mujoco.MjSpec, cfg: RobotConfig, sim: SimConfig) -> None:
    """Add actuators, gravity compensation, contact filtering and the object to the robot spec."""
    spec.option.timestep = 1.0 / (sim.control_hz * sim.substeps)
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10

    robot_bodies = set(_robot_body_names(spec, cfg.root_body))
    for body in robot_bodies:
        spec.body(body).gravcomp = 1.0
    for geom in spec.geoms:
        if geom.contype == 0 and geom.conaffinity == 0:
            continue  # visual-only geom
        parent = geom.parent.name if geom.parent is not None else ""
        if parent in robot_bodies:
            geom.contype, geom.conaffinity = ROBOT_GEOM
            geom.friction = [1.2, 0.01, 0.001]
    table = spec.geom("table")
    table.contype, table.conaffinity = WORLD_GEOM

    for arm in cfg.arms.values():
        for joint in arm.joints:
            act = spec.add_actuator(name=f"act_{joint}", target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            act.set_to_position(kp=sim.arm_kp, kv=sim.arm_kv)
        lead, follower = arm.finger_joints
        act = spec.add_actuator(name=f"act_{lead}", target=lead, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=sim.finger_kp, kv=sim.finger_kv)
        act.forcelimited = True
        act.forcerange = [-sim.finger_force, sim.finger_force]
        # The URDF <mimic> is ignored by MuJoCo: couple the second finger with an equality.
        spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=follower, name2=lead, data=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        for joint in arm.finger_joints:
            spec.joint(joint).damping = [2.0, 0.0, 0.0]
            spec.joint(joint).armature = 0.01
        for joint in arm.joints:
            spec.joint(joint).damping = [1.0, 0.0, 0.0]
            spec.joint(joint).armature = 0.05

    obj = sim.obj
    body = spec.worldbody.add_body(name=obj.name, pos=[0, 0, obj.size[2] / 2])
    body.add_freejoint(name=f"{obj.name}_joint")
    geom = body.add_geom(
        name=f"{obj.name}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[s / 2 for s in obj.size],
        mass=obj.mass,
        rgba=list(obj.rgba),
        friction=[obj.friction, 0.01, 0.001],
        condim=4,
    )
    geom.contype, geom.conaffinity = WORLD_GEOM


def pose_matrix(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rot, pos
    return T


class HandoverEnv:
    def __init__(self, robot_cfg: RobotConfig | None = None, sim: SimConfig | None = None, camera: dict | None = None):
        self.sim = sim or SimConfig()
        self.cfg = robot_cfg or openarm_config()
        self.robot = RobotModel(self.cfg, camera=camera, customize=lambda spec, cfg: build_physics(spec, cfg, self.sim))
        self.model, self.data = self.robot.model, self.robot.data
        self.ik = BimanualIK(self.robot)
        self.obj_body = self.model.body(self.sim.obj.name).id
        self.obj_qadr = self.model.jnt_qposadr[self.model.joint(f"{self.sim.obj.name}_joint").id]
        self.obj_geom = self.model.geom(f"{self.sim.obj.name}_geom").id
        root = self.model.body(self.cfg.root_body).id
        # Geoms that make up the robot and object in images (pasted onto real backgrounds).
        self.robot_geoms = {g for g in range(self.model.ngeom) if self.model.body_rootid[self.model.geom_bodyid[g]] == root}
        self.paste_geoms = self.robot_geoms | {self.obj_geom}
        self.arm_act = {s: [self.model.actuator(f"act_{j}").id for j in self.cfg.arms[s].joints] for s in SIDES}
        self.finger_act = {s: self.model.actuator(f"act_{self.cfg.arms[s].finger_joints[0]}").id for s in SIDES}
        # Finger collision geoms per arm, found via the finger joints' bodies (names like
        # "openarm_left_right_finger" make string matching ambiguous).
        self.finger_geoms = {}
        for s in SIDES:
            bodies = {self.model.jnt_bodyid[self.model.joint(j).id] for j in self.cfg.arms[s].finger_joints}
            self.finger_geoms[s] = {
                g for g in range(self.model.ngeom) if self.model.geom_contype[g] and self.model.geom_bodyid[g] in bodies
            }
        self.has_camera = camera is not None and self.sim.image_size[0] > 0
        self.renderer = None
        if self.has_camera:
            self.renderer = mujoco.Renderer(self.model, self.sim.image_size[1], self.sim.image_size[0])
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.sim.shadows
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = self.sim.shadows

    # -- state ----------------------------------------------------------------------------------------
    def reset(self, object_pose: np.ndarray | None = None, arm_qpos: np.ndarray | None = None) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        self.robot.reset()
        if arm_qpos is not None:
            self.data.qpos[: len(arm_qpos)] = arm_qpos
        for side in SIDES:
            self.robot.set_gripper(side, 1.0)
        if object_pose is not None:
            self.set_object_pose(object_pose)
        self.data.ctrl[:] = 0
        for side in SIDES:
            self.data.ctrl[self.arm_act[side]] = self.robot.arm_qpos(side)
            self.data.ctrl[self.finger_act[side]] = self.cfg.arms[side].finger_range[1]
        mujoco.mj_forward(self.model, self.data)
        for _ in range(10):  # let the object settle on the table
            self._physics_step()
        return self.observation()

    def get_state(self) -> np.ndarray:
        size = mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION)
        state = np.zeros(size)
        mujoco.mj_getState(self.model, self.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        return state

    def set_state(self, state: np.ndarray) -> None:
        mujoco.mj_setState(self.model, self.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.model, self.data)

    def set_object_pose(self, T: np.ndarray) -> None:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, T[:3, :3].ravel())
        self.data.qpos[self.obj_qadr : self.obj_qadr + 3] = T[:3, 3]
        self.data.qpos[self.obj_qadr + 3 : self.obj_qadr + 7] = quat
        mujoco.mj_forward(self.model, self.data)

    def object_pose(self) -> np.ndarray:
        return pose_matrix(self.data.xpos[self.obj_body].copy(), self.data.xmat[self.obj_body].reshape(3, 3).copy())

    def eef_pose(self, side: str) -> np.ndarray:
        return pose_matrix(*self.robot.tcp_pose(side))

    def gripper_opening(self, side: str) -> float:
        arm = self.cfg.arms[side]
        lo, hi = arm.finger_range
        return float((self.data.qpos[self.robot.qpos_addr(arm.finger_joints[0])] - lo) / (hi - lo))

    # -- control --------------------------------------------------------------------------------------
    def _physics_step(self) -> None:
        mujoco.mj_step(self.model, self.data, nstep=self.sim.substeps)

    def step_joints(self, arm_targets: dict[str, np.ndarray], openings: dict[str, float]) -> None:
        for side in SIDES:
            self.data.ctrl[self.arm_act[side]] = arm_targets[side]
            lo, hi = self.cfg.arms[side].finger_range
            self.data.ctrl[self.finger_act[side]] = lo + np.clip(openings[side], 0, 1) * (hi - lo)
        self._physics_step()

    def solve_ee(self, targets: dict[str, np.ndarray | None]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """IK from the commanded (not measured) joint targets, so tracking lag doesn't accumulate."""
        q = self.data.qpos.copy()
        for side in SIDES:
            for j, a in zip(self.cfg.arms[side].joints, self.arm_act[side]):
                q[self.robot.qpos_addr(j)] = self.data.ctrl[a]
        ik_targets = {s: None if T is None else (T[:3, 3], T[:3, :3]) for s, T in targets.items()}
        res = self.ik.solve(ik_targets, q)
        arm = {s: np.array([res.qpos[self.robot.qpos_addr(j)] for j in self.cfg.arms[s].joints]) for s in SIDES}
        return arm, res.pos_error

    def step_ee(self, targets: dict[str, np.ndarray | None], openings: dict[str, float]) -> dict[str, np.ndarray]:
        """Move both grippers towards 4x4 target poses (None = hold) and set gripper openings."""
        arm, _ = self.solve_ee(targets)
        self.step_joints(arm, openings)
        return arm

    # -- task -----------------------------------------------------------------------------------------
    def contacts(self) -> dict[str, bool]:
        touching = {s: False for s in SIDES}
        for c in self.data.contact[: self.data.ncon]:
            pair = {c.geom1, c.geom2}
            if self.obj_geom in pair:
                other = (pair - {self.obj_geom}).pop() if len(pair) > 1 else None
                for side in SIDES:
                    if other in self.finger_geoms[side]:
                        touching[side] = True
        return touching

    def grasped(self, side: str) -> bool:
        """Both fingers of `side` touch the object."""
        hits = set()
        for c in self.data.contact[: self.data.ncon]:
            if self.obj_geom in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.obj_geom else c.geom1
                if other in self.finger_geoms[side]:
                    hits.add(self.model.geom_bodyid[other])
        return len(hits) >= 2

    def is_success(self) -> bool:
        """Handover done: the right gripper holds the object off the table, the left one has let go."""
        lifted = self.object_pose()[2, 3] > self.sim.obj.size[2] / 2 + 0.03
        return bool(lifted and self.grasped("right") and not self.contacts()["left"])

    # -- observations ---------------------------------------------------------------------------------
    def joint_command_deg(self, arm_targets: dict[str, np.ndarray], openings: dict[str, float]) -> np.ndarray:
        """16-D action in LeRobot bi_openarm_follower order (degrees, gripper motor degrees)."""
        out = []
        for side in SIDES:
            closed, opened = self.cfg.arms[side].gripper_motor_range
            out += [*np.degrees(arm_targets[side]), closed + np.clip(openings[side], 0, 1) * (opened - closed)]
        return np.array(out)

    def observation(self) -> dict:
        obs = {}
        joints = []
        for side in SIDES:
            arm = self.cfg.arms[side]
            closed, opened = arm.gripper_motor_range
            opening = self.gripper_opening(side)
            joints += [*np.degrees(self.robot.arm_qpos(side)), closed + opening * (opened - closed)]
            T = self.eef_pose(side)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, T[:3, :3].ravel())
            obs[f"{side}_eef_pos"], obs[f"{side}_eef_quat"] = T[:3, 3], quat
            obs[f"{side}_gripper_opening"] = np.array([opening])
        obs["joint_pos"] = np.array(joints)
        T = self.object_pose()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, T[:3, :3].ravel())
        obs["object_pos"], obs["object_quat"] = T[:3, 3], quat
        if self.renderer is not None:
            self.renderer.update_scene(self.data, camera="scene_cam")
            obs["front_image"] = self.renderer.render().copy()
        return obs
