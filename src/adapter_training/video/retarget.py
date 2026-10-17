"""Hand keypoints -> parallel-gripper targets (Phantom-style mapping).

Per hand and frame:
  grasp point   = midpoint of the thumb tip and index tip
  closing axis  = thumb tip -> index tip         (gripper finger-sliding axis, TCP y)
  approach axis = knuckles -> grasp point, made orthogonal to the closing axis (TCP z)
  aperture      = thumb-index distance; the smallest 20% in each trajectory count as fully closed
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from adapter_training.video.hands import INDEX_MCP, INDEX_TIP, THUMB_MCP, THUMB_TIP


@dataclass
class GripperTrack:
    pos: np.ndarray  # (T, 3) table frame
    rot: np.ndarray  # (T, 3, 3) columns = TCP x, y, z axes in the table frame
    opening: np.ndarray  # (T,) 0 closed .. 1 open
    aperture_m: np.ndarray  # (T,) raw thumb-index distance
    valid: np.ndarray  # (T,) observed this frame (before gap filling)


def to_table_frame(kp_cam: np.ndarray, T_table_cam: np.ndarray) -> np.ndarray:
    return kp_cam @ T_table_cam[:3, :3].T + T_table_cam[:3, 3]


def fill_gaps(x: np.ndarray, valid: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate gaps up to max_gap frames; hold the ends. Returns filled array and coverage mask."""
    x = x.copy()
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return x, valid.copy()
    covered = valid.copy()
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            w = np.linspace(0, 1, b - a + 1)[1:-1].reshape(-1, *([1] * (x.ndim - 1)))
            x[a + 1 : b] = (1 - w) * x[a] + w * x[b]
            covered[a + 1 : b] = True
    flat = x.reshape(len(x), -1)
    for t in range(len(x)):  # hold first/last valid value outside the observed range
        if not covered[t]:
            src = idx[0] if t < idx[0] else idx[idx <= t][-1]
            flat[t] = flat[src]
    return x, covered


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    window = min(window, len(x) - (1 - len(x) % 2))
    if window < 5:
        return x
    return savgol_filter(x, window, polyorder=2, axis=0)


def orthonormal_frame(closing: np.ndarray, approach: np.ndarray) -> np.ndarray:
    y = closing / np.linalg.norm(closing, axis=-1, keepdims=True)
    z = approach - np.sum(approach * y, axis=-1, keepdims=True) * y
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=-1)


def retarget_hand(
    kp_table: np.ndarray,
    valid: np.ndarray,
    max_gap: int = 10,
    smooth_window: int = 9,
    closed_percentile: float = 20.0,
    open_percentile: float = 95.0,
    table_clearance: float = 0.005,
    min_aperture_range: float = 0.015,
) -> GripperTrack:
    """kp_table: (T, 21, 3) keypoints in the table frame, NaN where missing."""
    kp, covered = fill_gaps(kp_table, valid, max_gap)
    thumb, index = kp[:, THUMB_TIP], kp[:, INDEX_TIP]
    knuckles = (kp[:, THUMB_MCP] + kp[:, INDEX_MCP]) / 2

    pos = smooth((thumb + index) / 2, smooth_window)
    pos[:, 2] = np.maximum(pos[:, 2], table_clearance)  # the gripper can't go through the table
    closing = smooth(index - thumb, smooth_window)
    approach = smooth(pos - knuckles, smooth_window)

    # The two fingers are symmetric, so the closing axis sign is arbitrary: keep it continuous.
    for t in range(1, len(closing)):
        if np.dot(closing[t], closing[t - 1]) < 0:
            closing[t] *= -1
    rot = orthonormal_frame(closing, approach)

    aperture = np.linalg.norm(index - thumb, axis=1)
    if valid.any():
        lo, hi = np.percentile(aperture[valid], [closed_percentile, open_percentile])
    else:
        lo, hi = 0.0, 1.0
    if hi - lo < min_aperture_range:
        # The pinch width barely changes: this hand never grasped. Without this, per-hand
        # normalisation stretches tracking noise into repeated open/close "grasps".
        opening = np.ones(len(aperture))
    else:
        opening = np.clip((aperture - lo) / (hi - lo), 0.0, 1.0)
        opening = smooth(opening, smooth_window).clip(0, 1)
        opening[aperture <= lo] = 0.0

    if not covered.all():
        pos[~covered] = np.nan  # long gaps: no target (the arm holds its pose)
    return GripperTrack(pos=pos, rot=rot, opening=opening, aperture_m=aperture, valid=valid.copy())
