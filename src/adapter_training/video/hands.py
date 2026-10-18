"""3D hand tracking from RGB video.

Each backend returns, per detected hand, 21 keypoints in the standard order
(0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 little; MediaPipe = OpenPose = MANO order):
  - `kp3d`: metric 3D keypoints in an arbitrary hand-centred frame (shape is accurate, position isn't)
  - `kp2d`: pixel coordinates in the full image
`lift_to_camera` then places the hand in the camera frame with PnP using the *calibrated* intrinsics,
which fixes the metric depth that single-image models can't know on their own.

Backends:
  mediapipe  CPU, no setup beyond a 7.8 MB model download. Fine for testing; noisier 3D.
  wilor      GPU recommended. `uv sync --extra wilor`. More accurate MANO-based reconstruction.
"""

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import adapter_training  # noqa: F401  (TLS setup)
from adapter_training.common import DATA_DIR

THUMB_TIP, INDEX_TIP, WRIST, THUMB_MCP, INDEX_MCP, MIDDLE_MCP = 4, 8, 0, 2, 5, 9

MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


@dataclass
class HandObservation:
    side: str  # "left" / "right" from the demonstrator's point of view
    kp3d: np.ndarray  # (21, 3) metres, hand-centred
    kp2d: np.ndarray  # (21, 2) pixels
    score: float


class MediaPipeTracker:
    def __init__(self, fps: float, min_confidence: float = 0.5, mirrored: bool = False):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        model = DATA_DIR / "models" / "hand_landmarker.task"
        if not model.exists():
            model.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, model)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._mp = mp
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.fps = fps
        # MediaPipe labels handedness assuming a mirrored (selfie) image.
        self.mirrored = mirrored

    def __call__(self, frame_rgb: np.ndarray, frame_index: int) -> list[HandObservation]:
        h, w = frame_rgb.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        result = self.landmarker.detect_for_video(image, int(frame_index * 1000 / self.fps))
        hands = []
        for lm2d, lm3d, handed in zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness):
            label = handed[0].category_name.lower()
            if not self.mirrored:
                label = "right" if label == "left" else "left"
            hands.append(
                HandObservation(
                    side=label,
                    kp3d=np.array([[p.x, p.y, p.z] for p in lm3d]),
                    kp2d=np.array([[p.x * w, p.y * h] for p in lm2d]),
                    score=float(handed[0].score),
                )
            )
        return hands


class WiLoRTracker:
    """WiLoR via the `wilor-mini` package (https://github.com/warmshao/WiLoR-mini).

    WiLoR reconstructs each hand with an assumed focal length, so only its keypoint *shape* is
    used; the 2D keypoints are re-projected from its own camera model and PnP with the real
    intrinsics recovers metric depth. Not tested on this machine (no GPU) -- check the debug video.
    """

    def __init__(self, device: str | None = None):
        import torch
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        # wilor-mini's hand detector (ultralytics 8.1) pickles whole modules, which torch >= 2.6 refuses
        # to load by default (weights_only=True). Allow it only while WiLoR loads its own weights.
        torch_load = torch.load
        torch.load = lambda *a, **kw: torch_load(*a, **{"weights_only": False, **kw})
        try:
            self.pipe = WiLorHandPose3dEstimationPipeline(device=torch.device(device), dtype=dtype, verbose=False)
        finally:
            torch.load = torch_load

    def __call__(self, frame_rgb: np.ndarray, frame_index: int) -> list[HandObservation]:
        h, w = frame_rgb.shape[:2]
        hands = []
        for out in self.pipe.predict(frame_rgb):
            preds = out["wilor_preds"]
            kp3d = np.asarray(preds["pred_keypoints_3d"]).reshape(-1, 3)[:21].astype(float)
            cam_t = np.asarray(preds["pred_cam_t_full"]).reshape(-1)[:3].astype(float)
            f = float(np.asarray(preds["scaled_focal_length"]).reshape(-1)[0])
            is_right = float(out["is_right"]) > 0.5
            bbox = np.asarray(out["hand_bbox"], dtype=float)
            # WiLoR mirrors left hands internally; pick whichever x-convention projects into the box.
            best = None
            for flip in (1.0, -1.0):
                pts = kp3d * np.array([flip, 1, 1]) + cam_t
                kp2d = np.stack([f * pts[:, 0] / pts[:, 2] + w / 2, f * pts[:, 1] / pts[:, 2] + h / 2], axis=1)
                inside = np.mean(
                    (kp2d[:, 0] >= bbox[0]) & (kp2d[:, 0] <= bbox[2]) & (kp2d[:, 1] >= bbox[1]) & (kp2d[:, 1] <= bbox[3])
                )
                if best is None or inside > best[0]:
                    best = (inside, kp3d * np.array([flip, 1, 1]), kp2d)
            hands.append(
                HandObservation(side="right" if is_right else "left", kp3d=best[1], kp2d=best[2], score=float(best[0]))
            )
        return hands


def make_tracker(name: str, fps: float, **kwargs):
    if name == "mediapipe":
        return MediaPipeTracker(fps, **kwargs)
    if name == "wilor":
        return WiLoRTracker(**kwargs)
    raise ValueError(f"Unknown hand tracker '{name}' (use mediapipe or wilor)")


def lift_to_camera(hand: HandObservation, K: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Metric 3D keypoints in the camera frame via PnP, and the reprojection error in pixels."""
    obj = hand.kp3d.astype(np.float64)
    img = hand.kp2d.astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    R = cv2.Rodrigues(rvec)[0]
    pts = obj @ R.T + tvec.ravel()
    if np.any(pts[:, 2] <= 0):
        return None
    proj = cv2.projectPoints(obj, rvec, tvec, K, dist)[0].reshape(-1, 2)
    return pts, float(np.linalg.norm(proj - img, axis=1).mean())


def track_video(frames, tracker, K: np.ndarray, dist: np.ndarray, max_reproj_px: float = 15.0) -> dict:
    """Run a tracker over frames. Returns per side: kp_cam (T,21,3) NaN where missing, kp2d, valid, reproj."""
    frames = list(frames)
    T = len(frames)
    out = {
        side: {
            "kp_cam": np.full((T, 21, 3), np.nan),
            "kp2d": np.full((T, 21, 2), np.nan),
            "reproj_px": np.full(T, np.nan),
            "score": np.zeros(T),
        }
        for side in ("left", "right")
    }
    for t, frame in enumerate(frames):
        for hand in tracker(frame, t):
            track = out[hand.side]
            if hand.score < track["score"][t]:
                continue  # keep the most confident detection per side
            lifted = lift_to_camera(hand, K, dist)
            if lifted is None or lifted[1] > max_reproj_px:
                continue
            track["kp_cam"][t], track["reproj_px"][t] = lifted
            track["kp2d"][t] = hand.kp2d
            track["score"][t] = hand.score
    for track in out.values():
        track["valid"] = ~np.isnan(track["kp_cam"][:, 0, 0])
    return out


def save_tracks(path: Path, tracks: dict, fps: float) -> None:
    arrays = {f"{side}/{k}": v for side, t in tracks.items() for k, v in t.items()}
    np.savez_compressed(path, fps=fps, **arrays)


def load_tracks(path: Path) -> tuple[dict, float]:
    data = np.load(path)
    tracks: dict = {"left": {}, "right": {}}
    for key in data.files:
        if "/" in key:
            side, name = key.split("/", 1)
            tracks[side][name] = data[key]
    return tracks, float(data["fps"])
