"""Deployment-time robot overlay: draw the simulated OpenArm over live camera frames.

Generated training images show a rendered robot. On the real robot, the camera sees the real one.
`RobotOverlay` renders the robot at the measured joint angles from the calibrated viewpoint and
pastes it over the frame, covering the real robot, so the policy sees the same robot in both.
The object is *not* overlaid -- the policy has to see the real one.

    overlay = RobotOverlay("data/calib/camera.json")
    frame = overlay(frame, observation)   # observation: LeRobot keys like "left_joint_1.pos" (degrees)
"""

from pathlib import Path

import cv2
import mujoco
import numpy as np

from adapter_training.render.composite import CameraWarp, Compositor
from adapter_training.video.camera import Calibration
from adapter_training.video.robot import RobotModel, openarm_config

SIDES = ("left", "right")


class RobotOverlay:
    def __init__(
        self,
        calib: str | Path | Calibration,
        base_xyz: tuple[float, float, float] | None = None,
        render_width: int | None = 640,
        shadows: bool = False,
    ):
        """render_width: render the robot at this width and upscale (None = frame resolution)."""
        self.calib = calib if isinstance(calib, Calibration) else Calibration.load(calib)
        self.cfg = openarm_config()
        if base_xyz is not None:
            self.cfg.base_xyz = tuple(base_xyz)
        self.render_width = render_width
        self.shadows = shadows
        self._frame_size = None

    def _setup(self, width: int, height: int) -> None:
        rw = min(width, self.render_width or width)
        size = (rw, round(rw * height / width))
        self.robot = RobotModel(self.cfg, camera=CameraWarp(self.calib, size).camera)
        root = self.robot.model.body(self.cfg.root_body).id
        m = self.robot.model
        robot_geoms = {g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root}
        self.compositor = Compositor(m, self.calib, size, robot_geoms, shadows=self.shadows)
        self._frame_size = (width, height)

    def set_joints(self, joints: dict[str, float]) -> None:
        """Pose the model from LeRobot joint readings (arm joints in degrees, gripper motor degrees)."""
        for side in SIDES:
            arm = self.cfg.arms[side]
            for name, joint in zip(arm.lerobot_names[:7], arm.joints):
                self.robot.set_joint(joint, np.radians(float(joints[name])))
            closed, opened = arm.gripper_motor_range
            self.robot.set_gripper(side, (float(joints[arm.lerobot_names[7]]) - closed) / (opened - closed))
        mujoco.mj_kinematics(self.robot.model, self.robot.data)

    def __call__(self, frame: np.ndarray, joints: dict[str, float]) -> np.ndarray:
        h, w = frame.shape[:2]
        if self._frame_size != (w, h):
            self._setup(w, h)
        self.set_joints(joints)
        rgb, mask, shadow = self.compositor.render_layer(self.robot.data)
        if rgb.shape[1] != w:
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            if shadow is not None:
                shadow = cv2.resize(shadow, (w, h), interpolation=cv2.INTER_LINEAR)
        out = frame.astype(np.float32)
        if shadow is not None:
            out *= shadow[..., None]
        out[mask] = rgb[mask]
        return np.clip(out, 0, 255).astype(np.uint8)
