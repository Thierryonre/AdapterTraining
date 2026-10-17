from dataclasses import dataclass

from lerobot.robots.bi_openarm_follower import BiOpenArmFollowerConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("bi_openarm_overlay")
@dataclass(kw_only=True)
class BiOpenArmOverlayConfig(BiOpenArmFollowerConfig):
    """bi_openarm_follower whose camera frames get the simulated robot drawn over the real one.

    Use it for running policies trained on generated (rendered-robot) data, and for recording real
    data that should look the same.
    """

    id: str | None = "bi_openarm_overlay"
    # Camera calibration from `at-calibrate table` (the camera must not have moved since).
    overlay_calib: str = "data/calib/camera.json"
    # Observation key of the camera to overlay (a top-level camera in `cameras`).
    overlay_camera: str = "front"
    # Robot base in the table frame, same as used for data generation (default: robot.py's).
    overlay_base_xyz: list[float] | None = None
    # Render the robot at this width and upscale; lower = faster.
    overlay_render_width: int = 640
    overlay_shadows: bool = False
