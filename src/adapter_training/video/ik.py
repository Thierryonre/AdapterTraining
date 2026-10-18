"""Inverse kinematics for both arms with mink (differential IK on the MuJoCo model)."""

from dataclasses import dataclass

import mink
import numpy as np

from adapter_training.video.robot import RobotModel


@dataclass
class IKResult:
    qpos: np.ndarray
    pos_error: dict[str, float]  # metres, per arm
    rot_error: dict[str, float]  # radians, per arm


class BimanualIK:
    def __init__(
        self,
        robot: RobotModel,
        position_cost: float = 1.0,
        orientation_cost: float = 0.3,
        posture_cost: float = 1e-2,
        iterations: int = 60,
        dt: float = 0.05,
        avoid_table: bool = True,
    ):
        self.robot = robot
        self.configuration = mink.Configuration(robot.model)
        self.tasks = {
            side: mink.FrameTask(
                frame_name=arm.tcp_body,
                frame_type="body",
                position_cost=position_cost,
                orientation_cost=orientation_cost,
                lm_damping=1e-3,
            )
            for side, arm in robot.cfg.arms.items()
        }
        self.posture = mink.PostureTask(robot.model, cost=posture_cost)
        robot.reset()
        self.posture.set_target(robot.data.qpos.copy())
        self.limits = [mink.ConfigurationLimit(robot.model)]
        if avoid_table:
            # Keep forearms, wrists and grippers above the table: matching gripper poses alone can
            # produce elbow-down solutions that go through the tabletop.
            self.limits.append(
                mink.CollisionAvoidanceLimit(
                    robot.model,
                    geom_pairs=[(self._arm_geoms(robot), ["table"])],
                    minimum_distance_from_collisions=0.01,
                    collision_detection_distance=0.05,
                )
            )
        self.iterations = iterations
        self.dt = dt

    @staticmethod
    def _arm_geoms(robot: RobotModel) -> list[int]:
        m = robot.model
        bodies = set()
        for arm in robot.cfg.arms.values():
            for joint in arm.joints[2:] + arm.finger_joints:  # links 3-7, hand and fingers
                bodies.add(m.jnt_bodyid[m.joint(joint).id])
            bodies.add(m.body(arm.tcp_body).id)
        bodies |= {b for b in range(m.nbody) if m.body_parentid[b] in bodies}  # hand bodies on fixed joints
        return [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies and m.geom_contype[g]]

    def solve(
        self,
        targets: dict[str, tuple[np.ndarray, np.ndarray] | None],
        q_init: np.ndarray,
    ) -> IKResult:
        """targets: side -> (position (3,), rotation matrix (3, 3)) in the table frame, or None to hold."""
        self.configuration.update(q_init.copy())
        active = []
        for side, task in self.tasks.items():
            target = targets.get(side)
            if target is None:
                # Hold the current pose of an arm without a target (e.g. hand not visible).
                task.set_target(self.configuration.get_transform_frame_to_world(task.frame_name, task.frame_type))
            else:
                pos, rot = target
                rot = self._closest_grasp(rot, task)
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(rot), pos))
            active.append(task)

        for _ in range(self.iterations):
            try:
                vel = mink.solve_ik(
                    self.configuration, [*active, self.posture], self.dt, "daqp", damping=1e-3, limits=self.limits
                )
            except mink.NoSolutionFound:
                # The table constraint can be infeasible, e.g. when the arm already starts within its margin.
                # Take this step with joint limits only (lets it move clear); if even that fails, hold.
                try:
                    vel = mink.solve_ik(
                        self.configuration, [*active, self.posture], self.dt, "daqp", damping=1e-3,
                        limits=self.limits[:1],
                    )
                except mink.NoSolutionFound:
                    break
            self.configuration.integrate_inplace(vel, self.dt)

        pos_err, rot_err = {}, {}
        for side, task in self.tasks.items():
            err = task.compute_error(self.configuration)  # [translation(3), rotation(3)]
            pos_err[side] = float(np.linalg.norm(err[:3]))
            rot_err[side] = float(np.linalg.norm(err[3:]))
        return IKResult(self.configuration.q.copy(), pos_err, rot_err)

    def _closest_grasp(self, rot: np.ndarray, task: mink.FrameTask) -> np.ndarray:
        """A parallel gripper turned 180 deg about its approach (z) axis makes the same grasp.

        Pick whichever of the two is closer to the gripper's current orientation, so each arm
        uses the variant its wrist can reach and never flips between them.
        """
        current = self.configuration.get_transform_frame_to_world(task.frame_name, task.frame_type)
        current_rot = current.rotation().as_matrix()
        flipped = rot @ np.diag([-1.0, -1.0, 1.0])
        return rot if np.trace(current_rot.T @ rot) >= np.trace(current_rot.T @ flipped) else flipped
