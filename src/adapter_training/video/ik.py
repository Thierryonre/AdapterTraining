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
        self.iterations = iterations
        self.dt = dt

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
            vel = mink.solve_ik(
                self.configuration, [*active, self.posture], self.dt, "daqp", damping=1e-3, limits=self.limits
            )
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
