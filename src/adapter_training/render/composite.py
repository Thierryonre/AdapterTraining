"""Render the simulated robot (and optionally the object) and paste it onto real camera images.

Used in two places that must match exactly:
  - training data: generated demos are rendered over a real photo of the empty workspace
  - deployment:    the robot is rendered at its measured joint angles over the live camera image,
                   covering the real robot (Phantom's trick), so the policy sees the same robot in both

MuJoCo renders an ideal pinhole camera with a centred principal point. `CameraWarp` resamples that
render into the real camera's image: its intrinsics, principal point and lens distortion.
"""

from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

from adapter_training.video.camera import Calibration

TABLE_GROUP = 2  # scene geoms (table, tag marker) are in this group and never pasted


def output_size(calib: Calibration, width: int) -> tuple[int, int]:
    """Image size with the camera's aspect ratio, `width` pixels wide (even numbers for video codecs)."""
    height = round(width * calib.height / calib.width / 2) * 2
    return width, height


class CameraWarp:
    """Maps a pinhole MuJoCo render to the real camera's distorted image at a given size."""

    def __init__(self, calib: Calibration, size: tuple[int, int], margin: float = 1.15):
        w, h = size
        cal = calib.scaled(w, h)
        K, dist = cal.K_np, cal.dist_np
        # Render a slightly wider field of view so distorted edges still have content.
        self.render_size = (int(w * margin) // 2 * 2, int(h * margin) // 2 * 2)
        rw, rh = self.render_size
        self.fovy = float(np.degrees(2 * np.arctan(margin * h / (2 * K[1, 1]))))
        f_render = rh / (2 * np.tan(np.radians(self.fovy) / 2))
        # For every output pixel: undistort -> normalised ray -> pixel in the pinhole render.
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        pts = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
        norm = cv2.undistortPoints(pts, K, dist).reshape(h, w, 2)
        self.map_x = (norm[..., 0] * f_render + rw / 2).astype(np.float32)
        self.map_y = (norm[..., 1] * f_render + rh / 2).astype(np.float32)
        self.camera = {**cal.mujoco_camera(), "fovy": self.fovy}
        self.size = size

    def __call__(self, image: np.ndarray, nearest: bool = False) -> np.ndarray:
        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.remap(image, self.map_x, self.map_y, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


@dataclass
class Appearance:
    """Per-episode rendering variation (domain randomisation)."""

    light_dir: tuple = (0.0, 0.0, -1.0)
    light_diffuse: float = 0.6
    ambient: float = 0.4
    object_rgba: dict | None = None  # object name -> rgba
    brightness: float = 1.0  # multiplies the whole composite
    contrast: float = 1.0
    camera_jitter: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # dx, dy, dz (m), droll, dpitch, dyaw (rad)


class Compositor:
    """Renders `model`/`data` from the calibrated camera and pastes robot (+ object) pixels on an image."""

    def __init__(
        self,
        model: mujoco.MjModel,
        calib: Calibration,
        size: tuple[int, int],
        paste_geoms: set[int],
        shadows: bool = False,
    ):
        self.model = model
        self.warp = CameraWarp(calib, size)
        rw, rh = self.warp.render_size
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, rw)
        model.vis.global_.offheight = max(model.vis.global_.offheight, rh)
        self.renderer = mujoco.Renderer(model, rh, rw)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadows
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        self.shadows = shadows
        self.paste = np.zeros(model.ngeom + 1, bool)
        self.paste[list(paste_geoms)] = True
        self.cam_id = model.camera("scene_cam").id  # the model must have a camera named scene_cam
        self.base_camera = self.warp.camera
        self.set_camera(self.base_camera)
        self.scene_option = mujoco.MjvOption()

    def set_camera(self, cam: dict, jitter=(0, 0, 0, 0, 0, 0)) -> None:
        """Place scene_cam like the calibrated camera (optionally perturbed), with the warp's field of view."""
        x, y = np.asarray(cam["xyaxes"][:3], float), np.asarray(cam["xyaxes"][3:], float)
        R = np.stack([x, y, np.cross(x, y)], axis=1)
        if any(jitter[3:]):
            q, dR = np.zeros(4), np.zeros(9)
            mujoco.mju_euler2Quat(q, np.asarray(jitter[3:], float), "xyz")
            mujoco.mju_quat2Mat(dR, q)
            R = R @ dR.reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.ravel())
        self.model.cam_pos[self.cam_id] = np.asarray(cam["pos"], float) + np.asarray(jitter[:3], float)
        self.model.cam_quat[self.cam_id] = quat
        self.model.cam_fovy[self.cam_id] = cam["fovy"]

    def _render(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        self.renderer.update_scene(data, camera=self.cam_id, scene_option=self.scene_option)
        rgb = self.renderer.render().copy()
        self.renderer.enable_segmentation_rendering()
        seg = self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()
        geom = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        return rgb, self.paste[geom]  # geom -1 maps to the extra False entry

    def _shadow_factor(self, data: mujoco.MjData) -> np.ndarray:
        """How much the robot/object darken the table: render with and without them (shadows on)."""
        with_robot, _ = self._render(data)
        hidden = self.scene_option.geomgroup.copy()
        self.scene_option.geomgroup[:] = 0
        self.scene_option.geomgroup[TABLE_GROUP] = 1
        table_only, _ = self._render(data)
        self.scene_option.geomgroup[:] = hidden
        lum = lambda im: im.astype(np.float32).mean(axis=2) + 1.0  # noqa: E731
        return np.clip(lum(with_robot) / lum(table_only), 0.3, 1.0)

    def render_layer(self, data: mujoco.MjData, appearance: Appearance | None = None):
        """Robot/object pixels in the real camera's image: (rgb, mask, shadow factor or None)."""
        if appearance is not None:
            self._apply_appearance(appearance)
        rgb, mask = self._render(data)
        rgb, mask = self.warp(rgb), self.warp(mask.astype(np.uint8), nearest=True).astype(bool)
        shadow = self.warp(self._shadow_factor(data)) if self.shadows else None
        return rgb, mask, shadow

    def compose(self, base: np.ndarray, data: mujoco.MjData, appearance: Appearance | None = None) -> np.ndarray:
        """Paste the rendered robot/object onto `base` (resized to the compositor's size)."""
        if base.shape[1::-1] != tuple(self.warp.size):
            base = cv2.resize(base, self.warp.size, interpolation=cv2.INTER_AREA)
        rgb, mask, shadow = self.render_layer(data, appearance)
        out = base.astype(np.float32)
        if shadow is not None:
            out *= shadow[..., None]
        out[mask] = rgb[mask]
        if appearance is not None and (appearance.brightness != 1 or appearance.contrast != 1):
            mean = out.mean()
            out = (out - mean) * appearance.contrast + mean * appearance.brightness
        return np.clip(out, 0, 255).astype(np.uint8)

    def _apply_appearance(self, a: Appearance) -> None:
        m = self.model
        if m.nlight:
            m.light_dir[0] = np.asarray(a.light_dir) / np.linalg.norm(a.light_dir)
            m.light_diffuse[0] = a.light_diffuse
        m.vis.headlight.ambient[:] = a.ambient
        self.set_camera(self.base_camera, a.camera_jitter)
