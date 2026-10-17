from lerobot.robots.bi_openarm_follower import BiOpenArmFollower

from .config_bi_openarm_overlay import BiOpenArmOverlayConfig


class BiOpenArmOverlay(BiOpenArmFollower):
    """bi_openarm_follower + simulated-robot overlay on one camera (see RobotOverlay)."""

    config_class = BiOpenArmOverlayConfig
    name = "bi_openarm_overlay"

    def __init__(self, config: BiOpenArmOverlayConfig):
        super().__init__(config)
        from adapter_training.render.overlay import RobotOverlay

        self.overlay = RobotOverlay(
            config.overlay_calib,
            base_xyz=config.overlay_base_xyz,
            render_width=config.overlay_render_width,
            shadows=config.overlay_shadows,
        )

    def get_observation(self):
        obs = super().get_observation()
        cam = self.config.overlay_camera
        if cam in obs:
            obs[cam] = self.overlay(obs[cam], obs)
        return obs
