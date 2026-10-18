"""Camera calibration: intrinsics, and the camera pose relative to the table from an AprilTag.

Table frame (all robot targets are expressed in it):
  origin  = centre of the AprilTag lying flat on the table
  z       = up (out of the tag face)
  x       = the tag's "up" direction. Lay the tag so its top edge points AWAY from the robot,
            i.e. from the robot/demonstrator side of the table towards the opposite side.
  y       = z cross x (the robot's left)
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

TAG_FAMILIES = {
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "16h5": cv2.aruco.DICT_APRILTAG_16h5,
}

# Tag frame (OpenCV/ArUco: x right, y up in the printed image, z out of the face) -> table frame.
# table x = tag y, table y = -tag x, table z = tag z.
R_TAG_TABLE = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # columns: table axes in tag frame


@dataclass
class Calibration:
    width: int
    height: int
    K: list  # 3x3
    dist: list  # OpenCV distortion coefficients
    T_table_cam: list | None = None  # 4x4 camera pose in the table frame
    tag_reprojection_px: float | None = None
    tag_detections: int | None = None

    @property
    def K_np(self) -> np.ndarray:
        return np.asarray(self.K, dtype=float)

    @property
    def dist_np(self) -> np.ndarray:
        return np.asarray(self.dist, dtype=float)

    @property
    def T_np(self) -> np.ndarray:
        if self.T_table_cam is None:
            raise ValueError("Calibration has no camera pose; run at-calibrate with a tag visible.")
        return np.asarray(self.T_table_cam, dtype=float)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        return cls(**json.loads(Path(path).read_text()))

    def scaled(self, width: int, height: int) -> "Calibration":
        """Same camera at a different resolution (e.g. calibration and demo recorded at different sizes)."""
        sx, sy = width / self.width, height / self.height
        K = self.K_np.copy()
        K[0] *= sx
        K[1] *= sy
        return Calibration(width, height, K.tolist(), self.dist, self.T_table_cam, self.tag_reprojection_px)

    def mujoco_camera(self) -> dict:
        """MuJoCo camera (pos, xyaxes, fovy) matching this camera in the table frame."""
        T = self.T_np
        R = T[:3, :3]  # OpenCV camera axes: x right, y down, z forward
        fy = self.K_np[1, 1]
        fovy = float(np.degrees(2 * np.arctan(self.height / (2 * fy))))
        # MuJoCo cameras look along -z with y up.
        return {"pos": T[:3, 3].tolist(), "xyaxes": [*R[:, 0], *(-R[:, 1])], "fovy": fovy}


def intrinsics_from_fov(width: int, height: int, hfov_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Approximate pinhole intrinsics when no checkerboard calibration is available."""
    fx = width / (2 * np.tan(np.radians(hfov_deg) / 2))
    K = np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]])
    return K, np.zeros(5)


def intrinsics_from_checkerboard(
    frames: list[np.ndarray], pattern=(9, 6), square_m: float = 0.025
) -> tuple[np.ndarray, np.ndarray, float]:
    """Standard OpenCV calibration. pattern = inner corners (cols, rows). Returns K, dist, RMS px."""
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2) * square_m
    obj_pts, img_pts = [], []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, pattern)
        if ok:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
            )
            obj_pts.append(objp)
            img_pts.append(corners)
    if len(obj_pts) < 8:
        raise RuntimeError(f"Checkerboard found in only {len(obj_pts)} frames; need at least 8 varied views.")
    h, w = frames[0].shape[:2]
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (w, h), None, None)
    return K, dist.ravel(), float(rms)


class TagDetector:
    def __init__(self, family: str = "36h11", tag_id: int | None = None):
        dictionary = cv2.aruco.getPredefinedDictionary(TAG_FAMILIES[family])
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)
        self.tag_id = tag_id

    def detect(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """Return the 4 corner pixels (top-left, top-right, bottom-right, bottom-left) or None."""
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return None
        for c, i in zip(corners, ids.ravel()):
            if self.tag_id is None or i == self.tag_id:
                return c.reshape(4, 2)
        return None


ROBOT_SIDES = {"top": (0.0, 1.0), "bottom": (0.0, -1.0), "left": (1.0, 0.0), "right": (-1.0, 0.0)}  # away dir (image)


def orient_square(quad: np.ndarray, robot_side: str) -> np.ndarray:
    """Order 4 corners as the tag's (top-left, top-right, bottom-right, bottom-left), with the tag's top
    edge pointing away from the robot. robot_side = the image edge nearest the robot."""
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    d = quad - quad.mean(0)
    quad = quad[np.argsort(np.arctan2(d[:, 1], d[:, 0]))]  # clockwise on screen (y down), from any order
    away = np.asarray(ROBOT_SIDES[robot_side])
    best, best_score = quad, -np.inf
    for k in range(4):
        q = np.roll(quad, -k, axis=0)
        up = (q[0] + q[1]) / 2 - (q[3] + q[2]) / 2  # bottom edge -> top edge, in the image
        score = float(np.dot(up / np.linalg.norm(up), away))
        if score > best_score:
            best, best_score = q, score
    return best


class SquareDetector:
    """Fallback when the tag is visible but too small or blurred to decode: find its black square (a dark
    quadrilateral with lighter bits inside, surrounded by a light border) and orient it from robot_side.
    Same corners as TagDetector, so the calibration is identical; only the orientation is assumed."""

    def __init__(self, robot_side: str, min_area_px: float = 300.0):
        self.robot_side = robot_side
        self.min_area_px = min_area_px

    def detect(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        dark = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 10)
        contours, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area_px:
                continue
            quad = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
            if len(quad) != 4 or not cv2.isContourConvex(quad):
                continue
            quad = quad.reshape(4, 2).astype(np.float32)
            sides = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
            if sides.max() / sides.min() > 1.6:  # a tag seen from above is close to square
                continue
            inside = np.zeros_like(gray)
            cv2.fillConvexPoly(inside, quad.astype(np.int32), 255)
            centre = quad.mean(0)
            ring = np.zeros_like(gray)
            cv2.fillConvexPoly(ring, ((quad - centre) * 1.2 + centre).astype(np.int32), 255)
            ring[inside > 0] = 0
            if not ring.any():
                continue
            g_in, g_ring = gray[inside > 0].astype(float), gray[ring > 0].astype(float)
            threshold = (np.median(g_in) + np.median(g_ring)) / 2
            bits = float((g_in > threshold).mean())  # the tag's light bits
            contrast = float(np.median(g_ring) - np.percentile(g_in, 25))
            if not (0.1 < bits < 0.65) or contrast < 40:
                continue
            score = contrast * area ** 0.5
            if score > best_score:
                best, best_score = quad, score
        if best is None:
            return None
        refined = cv2.cornerSubPix(gray, best.reshape(-1, 1, 2).copy(), (3, 3), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        return orient_square(refined.reshape(4, 2), self.robot_side)


def tag_pose(corners: np.ndarray, tag_size_m: float, K: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, float]:
    """Pose of the tag in the camera frame (4x4) and the reprojection error in pixels."""
    s = tag_size_m / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, corners.astype(np.float64), K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise RuntimeError("solvePnP failed for the tag")
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.linalg.norm(proj.reshape(4, 2) - corners, axis=1).mean())
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T, err


def average_poses(poses: list[np.ndarray]) -> np.ndarray:
    """Average rigid transforms (chordal mean for rotation)."""
    R_sum = sum(p[:3, :3] for p in poses)
    U, _, Vt = np.linalg.svd(R_sum)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.median([p[:3, 3] for p in poses], axis=0)
    return T


def table_pose_from_frames(
    frames: list[np.ndarray], tag_size_m: float, K: np.ndarray, dist: np.ndarray, family="36h11", tag_id=None,
    robot_side: str | None = None, corners_px: list | None = None,
) -> tuple[np.ndarray, float, int]:
    """Camera pose in the table frame (4x4), mean reprojection error, number of frames used.

    Overrides, for a tag that's visible but can't be decoded:
      corners_px  the tag's black-square corners in pixels, clicked by hand (any order; with robot_side)
      robot_side  the image edge nearest the robot ("top", "bottom", "left", "right"): if the tag can't be
                  decoded, its black square is found by shape and oriented so its top points away from it
    """
    if corners_px is not None:
        if robot_side is None:
            raise RuntimeError("tag_corners_px needs robot_side (the image edge nearest the robot)")
        detectors = [lambda frame: orient_square(np.asarray(corners_px, float), robot_side)]
        frames = frames[:1]
    else:
        detectors = [TagDetector(family, tag_id).detect] + ([SquareDetector(robot_side).detect] if robot_side else [])
    poses, errors = [], []
    for detect in detectors:  # the decoded tag if possible, else the fallback
        for frame in frames:
            corners = detect(frame)
            if corners is None:
                continue
            T_cam_tag, err = tag_pose(corners, tag_size_m, K, dist)
            poses.append(T_cam_tag)
            errors.append(err)
        if poses:
            if detect is not detectors[0] or corners_px is not None:
                print(f"[calibrate] tag not decoded: used its black square, top edge away from the robot "
                      f"(robot_side={robot_side!r}); check the axes picture")
            break
    if not poses:
        hint = "" if robot_side else ' Set "robot_side" in setup.json to use the tag\'s square without decoding it.'
        raise RuntimeError("AprilTag not found in any frame. Check --tag-family / --tag-id and lighting." + hint)
    T_cam_tag = average_poses(poses)
    T_cam_table = T_cam_tag.copy()
    T_cam_table[:3, :3] = T_cam_tag[:3, :3] @ R_TAG_TABLE
    return np.linalg.inv(T_cam_table), float(np.mean(errors)), len(poses)


def track_object_tag(
    frames: list[np.ndarray],
    calib: Calibration,
    tag_id: int,
    tag_size_m: float,
    object_height_m: float,
    family: str = "36h11",
) -> np.ndarray:
    """Pose (T, 4, 4) of an object with an AprilTag on its top face, in the table frame; NaN where unseen.

    The object frame is the tag frame mapped like the table frame (x = tag's up direction, z up),
    with its origin moved from the tag (top face) down to the object's centre.
    """
    detector = TagDetector(family, tag_id)
    poses = np.full((len(frames), 4, 4), np.nan)
    for t, frame in enumerate(frames):
        corners = detector.detect(frame)
        if corners is None:
            continue
        T_cam_tag, _ = tag_pose(corners, tag_size_m, calib.K_np, calib.dist_np)
        T_table_tag = calib.T_np @ T_cam_tag
        T = np.eye(4)
        T[:3, :3] = T_table_tag[:3, :3] @ R_TAG_TABLE
        T[:3, 3] = T_table_tag[:3, 3] - T_table_tag[:3, 2] * object_height_m / 2
        poses[t] = T
    return poses


def draw_table_axes(frame_rgb: np.ndarray, calib: Calibration, length_m: float = 0.1) -> np.ndarray:
    """Draw the table frame (x red, y green, z blue) for checking the calibration by eye."""
    T_cam_table = np.linalg.inv(calib.T_np)
    pts = np.array([[0, 0, 0], [length_m, 0, 0], [0, length_m, 0], [0, 0, length_m]], dtype=np.float64)
    rvec = cv2.Rodrigues(T_cam_table[:3, :3])[0]
    proj, _ = cv2.projectPoints(pts, rvec, T_cam_table[:3, 3], calib.K_np, calib.dist_np)
    p = proj.reshape(-1, 2).astype(int)
    out = frame_rgb.copy()
    for i, color in zip(range(1, 4), [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        cv2.line(out, tuple(p[0]), tuple(p[i]), color, 3)
    return out
