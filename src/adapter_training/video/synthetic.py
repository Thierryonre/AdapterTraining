"""Synthetic handover demo for testing the video pipeline without filming anything.

Creates, in data/synthetic/:
  tag.mp4                 1 s clip of an AprilTag on the table, for `at-calibrate table`
  handover.mp4            6 s clip with drawn hand keypoints doing a left-to-right handover
  handover_tracks.npz     the ground-truth hand tracks (with noise and a dropout), named so that
                          `at-video2robot --reuse-tracks --out data/synthetic` skips the hand tracker

    uv run at-synthetic-demo
    uv run at-calibrate table data/synthetic/tag.mp4 --tag-size 0.16 --hfov 70 --out data/synthetic/camera.json
    uv run at-video2robot data/synthetic/handover.mp4 --calib data/synthetic/camera.json --reuse-tracks \
        --out data/synthetic --task "Pass the block from the left hand to the right hand."
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from adapter_training.common import DATA_DIR
from adapter_training.video.camera import R_TAG_TABLE, intrinsics_from_fov
from adapter_training.video.hands import save_tracks
from adapter_training.video.pipeline import VideoWriter

W, H, FPS, HFOV, TAG_SIZE = 960, 540, 30, 70.0, 0.16
CAM_POS = np.array([0.95, 0.05, 0.55])  # opposite side of the table from the robot, looking back at it


def camera_pose() -> np.ndarray:
    z = -CAM_POS / np.linalg.norm(CAM_POS)  # look at the tag
    x = np.cross(z, [0, 0, 1])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
    T[:3, 3] = CAM_POS
    return T


def project(points_table: np.ndarray, T_table_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    T_cam_table = np.linalg.inv(T_table_cam)
    rvec = cv2.Rodrigues(T_cam_table[:3, :3])[0]
    return cv2.projectPoints(points_table.reshape(-1, 3), rvec, T_cam_table[:3, 3], K, np.zeros(5))[0].reshape(-1, 2)


def background(T_table_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Grey table with a 36h11 tag (id 0), top edge towards +x."""
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 0, 400)
    border = 50
    tag = cv2.copyMakeBorder(tag, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
    s = TAG_SIZE / 2
    corners = (R_TAG_TABLE.T @ np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]]).T).T
    dst = project(corners, T_table_cam, K).astype(np.float32)
    src = np.float32([[border, border], [border + 400, border], [border + 400, border + 400], [border, border + 400]])
    M = cv2.getPerspectiveTransform(src, dst)
    img = np.full((H, W), 110, np.uint8)
    warped = cv2.warpPerspective(tag, M, (W, H))
    mask = cv2.warpPerspective(np.full_like(tag, 255), M, (W, H)) > 0
    img[mask] = warped[mask]
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def hand_keypoints(grasp: np.ndarray, approach: np.ndarray, closing: np.ndarray, aperture: float) -> np.ndarray:
    """Plausible 21-point pinching hand around a grasp point (only thumb/index points matter)."""
    a = approach / np.linalg.norm(approach)
    c = closing / np.linalg.norm(closing)
    n = np.cross(a, c)
    kp = np.zeros((21, 3))
    kp[0] = grasp - 0.13 * a  # wrist
    kp[4] = grasp - c * aperture / 2  # thumb tip
    kp[8] = grasp + c * aperture / 2  # index tip
    for i, f in zip(range(1, 4), (0.25, 0.5, 0.75)):  # thumb chain
        kp[i] = (1 - f) * kp[0] + f * kp[4] - 0.02 * c
    kp[5] = grasp - 0.07 * a + 0.01 * c  # index MCP
    kp[6], kp[7] = (2 * kp[5] + kp[8]) / 3 + 0.01 * n, (kp[5] + 2 * kp[8]) / 3 + 0.01 * n
    for finger, offset in zip(range(9, 21, 4), (0.0, 0.02, 0.04)):  # curled middle/ring/little
        mcp = grasp - 0.075 * a + (0.0 - offset) * c + 0.01 * n
        kp[finger] = mcp
        kp[finger + 1] = mcp + 0.03 * a - 0.01 * n
        kp[finger + 2] = mcp + 0.04 * a - 0.03 * n
        kp[finger + 3] = mcp + 0.03 * a - 0.045 * n
    return kp


def handover_script(T: int) -> dict:
    """Keyframed handover in the table frame: (time s, grasp point, aperture m) per hand."""
    keys = {
        "left": [
            (0.0, [0.00, 0.26, 0.15], 0.08),
            (1.2, [0.12, 0.18, 0.04], 0.08),
            (1.6, [0.12, 0.18, 0.04], 0.025),
            (3.2, [0.16, 0.04, 0.16], 0.025),
            (4.2, [0.16, 0.04, 0.16], 0.025),
            (4.6, [0.16, 0.04, 0.16], 0.08),
            (6.0, [0.00, 0.26, 0.15], 0.08),
        ],
        "right": [
            (0.0, [0.00, -0.26, 0.15], 0.08),
            (2.5, [0.00, -0.26, 0.15], 0.08),
            (3.6, [0.16, -0.02, 0.16], 0.08),
            (4.0, [0.16, -0.02, 0.16], 0.025),
            (5.2, [0.12, -0.18, 0.05], 0.025),
            (6.0, [0.12, -0.18, 0.05], 0.08),
        ],
    }
    t = np.arange(T) / FPS
    out = {}
    for side, k in keys.items():
        times = np.array([x[0] for x in k])
        pos = np.stack([np.interp(t, times, [x[1][d] for x in k]) for d in range(3)], axis=1)
        aperture = np.interp(t, times, [x[2] for x in k])
        out[side] = (pos, aperture)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "synthetic")
    parser.add_argument("--noise-mm", type=float, default=3.0, help="Keypoint noise added to the tracks")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    K, _ = intrinsics_from_fov(W, H, HFOV)
    T_table_cam = camera_pose()
    bg = background(T_table_cam, K)

    writer = VideoWriter(args.out / "tag.mp4", FPS, (W, H))
    for _ in range(FPS):
        writer.write(bg)
    writer.close()

    T = 6 * FPS
    script = handover_script(T)
    T_cam_table = np.linalg.inv(T_table_cam)
    tracks = {}
    for side, (pos, aperture) in script.items():
        # Forward approach tilted slightly down; closing axis roughly vertical like a side pinch.
        approach = np.array([1.0, 0.0, -0.4])
        closing = np.array([0.0, 0.3 if side == "left" else -0.3, 1.0])
        kp = np.stack([hand_keypoints(pos[t], approach, closing, aperture[t]) for t in range(T)])
        kp += rng.normal(0, args.noise_mm / 1000, kp.shape)
        kp_cam = kp @ T_cam_table[:3, :3].T + T_cam_table[:3, 3]
        kp2d = np.stack([project(kp[t], T_table_cam, K) for t in range(T)])
        valid = np.ones(T, bool)
        if side == "right":
            valid[100:106] = False  # short tracking dropout, to exercise gap filling
        kp_cam[~valid] = np.nan
        kp2d[~valid] = np.nan
        tracks[side] = {"kp_cam": kp_cam, "kp2d": kp2d, "reproj_px": np.zeros(T), "score": valid * 1.0, "valid": valid}

    writer = VideoWriter(args.out / "handover.mp4", FPS, (W, H))
    for t in range(T):
        frame = bg.copy()
        for side, color in (("left", (0, 220, 0)), ("right", (255, 60, 60))):
            if tracks[side]["valid"][t]:
                for p in tracks[side]["kp2d"][t].astype(int):
                    cv2.circle(frame, tuple(p), 4, color, -1)
        writer.write(frame)
    writer.close()
    save_tracks(args.out / "handover_tracks.npz", tracks, FPS)
    print(f"Wrote tag.mp4, handover.mp4 and handover_tracks.npz to {args.out}")


if __name__ == "__main__":
    main()
