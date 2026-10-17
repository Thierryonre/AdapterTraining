"""LeRobot plugin: `--robot.type=bi_openarm_overlay` (auto-discovered by lerobot-rollout / lerobot-record)."""

from .bi_openarm_overlay import BiOpenArmOverlay
from .config_bi_openarm_overlay import BiOpenArmOverlayConfig

__all__ = ["BiOpenArmOverlay", "BiOpenArmOverlayConfig"]
