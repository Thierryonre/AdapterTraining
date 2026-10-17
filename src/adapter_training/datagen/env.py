"""Physics simulation of OpenArm manipulation scenes in MuJoCo.

The same OpenArm model as the video pipeline, plus:
  - position-controlled joints (gravity-compensated arms) and position-controlled fingers
  - a collidable table and any number of free objects (boxes, cylinders or reconstructed meshes)
  - contact filtering: robot links don't collide with each other, only with the table and objects
    (so the two grippers can pass through each other -- keep two-handed grasps apart)

Object frame convention: origin at the centre of the object's bounding box, z up when it rests on the
table, so a resting object's centre is at height size_z / 2.

Actions follow LeRobot's OpenArm convention: 7 joint targets per arm (radians internally,
degrees when exported) + a gripper opening in [0, 1] per arm.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from adapter_training.video.ik import BimanualIK
from adapter_training.video.robot import RobotConfig, RobotModel, openarm_config

SIDES = ("left", "right")
ROBOT_GEOM = (2, 1)  # contype, conaffinity: robot geoms never collide with each other
WORLD_GEOM = (1, 3)  # table/objects collide with the robot and with each other
HIDDEN_GROUP = 3  # collision-only geoms of mesh objects (not rendered by default)


@dataclass
class SceneObject:
    name: str = "bar"
    shape: str = "box"  # "box" | "cylinder" | "mesh"
    size: tuple[float, float, float] = (0.14, 0.05, 0.04)  # bounding box: full x, y, z extents (m)
    mass: float = 0.08
    friction: float = 1.2
    rgba: tuple[float, float, float, float] = (0.85, 0.35, 0.1, 1.0)
    visual_mesh: str | None = None  # mesh file in the object frame (metres)
    texture: str | None = None  # PNG for visual_mesh (needs texture coordinates in the mesh file)
    collision_meshes: list[str] = field(default_factory=list)  # convex parts, same frame
    collision_volumes: list[float] = field(default_factory=list)  # to split the mass between parts

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SceneObject":
        d = dict(d)
        d["size"] = tuple(d["size"])
        d["rgba"] = tuple(d["rgba"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


ObjectSpec = SceneObject  # backwards-compatible name


@dataclass
class SimConfig:
    control_hz: float = 30.0
    substeps: int = 16
    arm_kp: float = 400.0
    arm_kv: float = 40.0
    finger_kp: float = 1500.0
    finger_kv: float = 30.0
    finger_force: float = 60.0
    image_size: tuple[int, int] = (320, 240)  # (width, height); (0, 0) disables the built-in renderer
    shadows: bool = False  # ~3x slower to render on integrated graphics; cheap on a real GPU
    objects: list[SceneObject] = field(default_factory=lambda: [SceneObject()])


def _robot_body_names(spec: mujoco.MjSpec, root: str) -> list[str]:
    names, stack = [], [spec.body(root)]
    while stack:
        body = stack.pop()
        names.append(body.name)
        stack.extend(body.bodies)
    return names


def _add_object(spec: mujoco.MjSpec, obj: SceneObject) -> None:
    body = spec.worldbody.add_body(name=obj.name, pos=[0, 0, obj.size[2] / 2])
    body.add_freejoint(name=f"{obj.name}_joint")
    common = dict(friction=[obj.friction, 0.01, 0.001], condim=4)
    if obj.shape == "box":
        geom = body.add_geom(
            name=f"{obj.name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in obj.size],
            mass=obj.mass, rgba=list(obj.rgba), **common,
        )
        geom.contype, geom.conaffinity = WORLD_GEOM
    elif obj.shape == "cylinder":
        radius = min(obj.size[0], obj.size[1]) / 2
        geom = body.add_geom(
            name=f"{obj.name}_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[radius, obj.size[2] / 2, 0],
            mass=obj.mass, rgba=list(obj.rgba), **common,
        )
        geom.contype, geom.conaffinity = WORLD_GEOM
    elif obj.shape == "mesh":
        visual = spec.add_mesh(name=f"{obj.name}_visual", file=obj.visual_mesh)
        visual.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX  # reconstructed meshes may not be watertight
        geom = body.add_geom(
            name=f"{obj.name}_geom", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{obj.name}_visual",
            rgba=list(obj.rgba), contype=0, conaffinity=0, density=0, mass=0,
        )
        if obj.texture:
            spec.add_texture(name=f"{obj.name}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=obj.texture)
            # Partly self-lit: the baked pixels already contain the real scene's shading.
            mat = spec.add_material(name=f"{obj.name}_mat", rgba=[1, 1, 1, 1], emission=0.6, specular=0.1, shininess=0.1)
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{obj.name}_tex"
            geom.material = f"{obj.name}_mat"
            geom.rgba = [1, 1, 1, 1]
        parts = obj.collision_meshes or [obj.visual_mesh]
        volumes = obj.collision_volumes or [1.0] * len(parts)
        total = float(sum(volumes)) or 1.0
        for i, (path, vol) in enumerate(zip(parts, volumes)):
            spec.add_mesh(name=f"{obj.name}_col{i}", file=path)
            col = body.add_geom(
                name=f"{obj.name}_col{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{obj.name}_col{i}",
                mass=max(obj.mass * vol / total, 1e-4), group=HIDDEN_GROUP, rgba=[0.5, 0.5, 0.5, 0.3], **common,
            )
            col.contype, col.conaffinity = WORLD_GEOM
    else:
        raise ValueError(f"unknown object shape {obj.shape!r}")


def build_physics(spec: mujoco.MjSpec, cfg: RobotConfig, sim: SimConfig) -> None:
    """Add actuators, gravity compensation, contact filtering and the objects to the robot spec."""
    spec.strippath = False  # object meshes use absolute paths (robot meshes are bare names already)
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

    for obj in sim.objects:
        _add_object(spec, obj)


def pose_matrix(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rot, pos
    return T


def yaw_pose(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return pose_matrix(np.array([x, y, z]), np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))


class SceneEnv:
    def __init__(self, robot_cfg: RobotConfig | None = None, sim: SimConfig | None = None, camera: dict | None = None):
        self.sim = sim or SimConfig()
        self.cfg = robot_cfg or openarm_config()
        self.robot = RobotModel(self.cfg, camera=camera, customize=lambda spec, cfg: build_physics(spec, cfg, self.sim))
        self.model, self.data = self.robot.model, self.robot.data
        self.ik = BimanualIK(self.robot)
        m = self.model
        self.objects = {o.name: o for o in self.sim.objects}
        self.object_names = [o.name for o in self.sim.objects]
        self.obj_body = {n: m.body(n).id for n in self.object_names}
        self.obj_qadr = {n: m.jnt_qposadr[m.joint(f"{n}_joint").id] for n in self.object_names}
        self.obj_geoms = {n: {g for g in range(m.ngeom) if m.geom_bodyid[g] == self.obj_body[n]} for n in self.object_names}
        self.obj_contact_geoms = {n: {g for g in self.obj_geoms[n] if m.geom_contype[g]} for n in self.object_names}
        self.robot_qpos_size = min(self.obj_qadr.values()) if self.obj_qadr else m.nq
        root = m.body(self.cfg.root_body).id
        # Geoms that make up the robot and objects in images (pasted onto real backgrounds).
        self.robot_geoms = {g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root}
        self.paste_geoms = self.robot_geoms | set().union(*self.obj_geoms.values())
        self.arm_act = {s: [m.actuator(f"act_{j}").id for j in self.cfg.arms[s].joints] for s in SIDES}
        self.finger_act = {s: m.actuator(f"act_{self.cfg.arms[s].finger_joints[0]}").id for s in SIDES}
        # Finger collision geoms per arm, found via the finger joints' bodies (names like
        # "openarm_left_right_finger" make string matching ambiguous).
        self.finger_geoms = {}
        for s in SIDES:
            bodies = {m.jnt_bodyid[m.joint(j).id] for j in self.cfg.arms[s].finger_joints}
            self.finger_geoms[s] = {g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g] in bodies}
        self.success_fn: Callable[["SceneEnv"], bool] | None = None
        self.has_camera = camera is not None and self.sim.image_size[0] > 0
        self.renderer = None
        if self.has_camera:
            self.renderer = mujoco.Renderer(m, self.sim.image_size[1], self.sim.image_size[0])
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.sim.shadows
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = self.sim.shadows

    # -- objects (first object is the default, for single-object code) ---------------------------------
    def _name(self, name: str | None) -> str:
        return name or self.object_names[0]

    def object_size(self, name: str | None = None) -> np.ndarray:
        return np.asarray(self.objects[self._name(name)].size, float)

    @property
    def obj_geom(self) -> int:  # visual geom of the first object (colour randomisation)
        return self.model.geom(f"{self.object_names[0]}_geom").id

    def object_pose(self, name: str | None = None) -> np.ndarray:
        b = self.obj_body[self._name(name)]
        return pose_matrix(self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy())

    def object_poses(self) -> dict[str, np.ndarray]:
        return {n: self.object_pose(n) for n in self.object_names}

    def set_object_pose(self, T: np.ndarray, name: str | None = None) -> None:
        adr = self.obj_qadr[self._name(name)]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, T[:3, :3].ravel())
        self.data.qpos[adr : adr + 3] = T[:3, 3]
        self.data.qpos[adr + 3 : adr + 7] = quat
        mujoco.mj_forward(self.model, self.data)

    # -- state ----------------------------------------------------------------------------------------
    def reset(
        self, object_pose: np.ndarray | dict | None = None, arm_qpos: np.ndarray | None = None, settle_steps: int = 10
    ) -> dict:
        """object_pose: one 4x4 (first object) or {name: 4x4}."""
        mujoco.mj_resetData(self.model, self.data)
        self.robot.reset()
        if arm_qpos is not None:
            self.data.qpos[: len(arm_qpos)] = arm_qpos
        for side in SIDES:
            self.robot.set_gripper(side, 1.0)
        poses = object_pose if isinstance(object_pose, dict) else ({} if object_pose is None else {None: object_pose})
        for name, T in poses.items():
            self.set_object_pose(T, name)
        self.data.ctrl[:] = 0
        for side in SIDES:
            self.data.ctrl[self.arm_act[side]] = self.robot.arm_qpos(side)
            self.data.ctrl[self.finger_act[side]] = self.cfg.arms[side].finger_range[1]
        mujoco.mj_forward(self.model, self.data)
        for _ in range(settle_steps):  # let objects settle on the table
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

    # -- contacts ---------------------------------------------------------------------------------------
    def _finger_hits(self, side: str, name: str) -> set[int]:
        geoms, hits = self.obj_contact_geoms[name], set()
        for c in self.data.contact[: self.data.ncon]:
            if c.geom1 in geoms and c.geom2 in self.finger_geoms[side]:
                hits.add(self.model.geom_bodyid[c.geom2])
            elif c.geom2 in geoms and c.geom1 in self.finger_geoms[side]:
                hits.add(self.model.geom_bodyid[c.geom1])
        return hits

    def touching(self, side: str, name: str | None = None) -> bool:
        return bool(self._finger_hits(side, self._name(name)))

    def contacts(self, name: str | None = None) -> dict[str, bool]:
        return {s: self.touching(s, name) for s in SIDES}

    def grasped(self, side: str, name: str | None = None) -> bool:
        """Both fingers of `side` touch the object."""
        return len(self._finger_hits(side, self._name(name))) >= 2

    def holders(self, name: str) -> set[str]:
        return {s for s in SIDES if self.grasped(s, name)}

    def is_success(self) -> bool:
        if self.success_fn is not None:
            return bool(self.success_fn(self))
        # Legacy default (handover): the right gripper holds the first object off the table, left let go.
        lifted = self.object_pose()[2, 3] > self.object_size()[2] / 2 + 0.03
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
        for i, name in enumerate(self.object_names):
            T = self.object_pose(name)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, T[:3, :3].ravel())
            obs[f"object_{name}_pos"], obs[f"object_{name}_quat"] = T[:3, 3], quat
            if i == 0:
                obs["object_pos"], obs["object_quat"] = T[:3, 3], quat
        if self.renderer is not None:
            self.renderer.update_scene(self.data, camera="scene_cam")
            obs["front_image"] = self.renderer.render().copy()
        return obs


HandoverEnv = SceneEnv  # backwards-compatible name
