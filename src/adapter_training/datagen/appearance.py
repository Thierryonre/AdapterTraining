"""Real backgrounds and per-episode appearance randomisation for generated demos."""

from pathlib import Path

import cv2
import numpy as np

from adapter_training.render.composite import Appearance


def capture_background(video: Path, max_frames: int = 300, stride: int = 3) -> np.ndarray:
    """Median of a clip of the empty workspace: moving things (people, hands) disappear."""
    cap = cv2.VideoCapture(str(video))
    frames, i = [], 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    if not frames:
        raise FileNotFoundError(video)
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _jitter_colour(rng, rgba, s: float) -> tuple:
    hsv = cv2.cvtColor(np.uint8([[np.asarray(rgba[:3]) * 255]]), cv2.COLOR_RGB2HSV)[0, 0].astype(float)
    hsv += [rng.uniform(-6, 6) * s, rng.uniform(-30, 30) * s, rng.uniform(-30, 30) * s]
    hsv = np.clip(hsv, [0, 0, 0], [179, 255, 255])
    rgb = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0, 0] / 255.0
    return (*rgb, 1.0)


def _tint(rng, s: float) -> tuple:
    """Mild multiplicative tint for textured objects: +/-12% brightness, slight colour cast."""
    base = 1 + rng.uniform(-0.12, 0.12) * s
    cast = rng.uniform(-0.05, 0.05, 3) * s
    return (*np.clip(base + cast, 0.6, 1.0).tolist(), 1.0)


def sample_appearance(rng: np.random.Generator, object_colours: dict, strength: float = 1.0) -> Appearance:
    """Random lighting, per-object colour, global brightness/contrast and a small camera perturbation.

    object_colours: {object name: nominal rgba, or None for textured objects (they get a mild tint
    around white so the real texture stays recognisable)}. strength=0 gives the nominal appearance;
    1 is the default amount of variation.
    """
    s = strength
    return Appearance(
        light_dir=(rng.uniform(-0.5, 0.5) * s, rng.uniform(-0.5, 0.5) * s, -1.0),
        light_diffuse=float(0.6 + rng.uniform(-0.2, 0.2) * s),
        ambient=float(0.4 + rng.uniform(-0.15, 0.15) * s),
        object_rgba={
            name: _jitter_colour(rng, rgba, s) if rgba is not None else _tint(rng, s)
            for name, rgba in object_colours.items()
        },
        brightness=float(1 + rng.uniform(-0.15, 0.15) * s),
        contrast=float(1 + rng.uniform(-0.15, 0.15) * s),
        # ~1 cm / ~1 deg: covers calibration error between training images and the deployment overlay
        camera_jitter=tuple(np.concatenate([rng.normal(0, 0.005 * s, 3), rng.normal(0, np.radians(0.5) * s, 3)])),
    )
