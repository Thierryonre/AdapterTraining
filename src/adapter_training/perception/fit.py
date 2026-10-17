"""Place an object model on the table so its silhouette matches the object's mask in the camera image.

Single-image 3D reconstructions (SAM 3D Objects) give good *shape* but their scale and pose come from a
different camera model. Here only the shape is used: the mesh is put in one of its stable resting
orientations on the calibrated table plane, and position, yaw and scale are optimised so its projection
(real intrinsics + lens distortion) overlaps the segmentation mask (IoU). Boxes and cylinders are fitted
the same way with per-axis scale, as a fallback.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import trimesh
from scipy.optimize import minimize

from adapter_training.video.camera import Calibration


@dataclass
class FitResult:
    mesh: trimesh.Trimesh  # in the object frame: bbox-centred, z up, metres
    pose: np.ndarray  # 4x4 object -> table
    size: np.ndarray  # bbox extents (m)
    iou: float
    kind: str  # "mesh" | "box" | "cylinder"


class SilhouetteRenderer:
    """Fast mask rendering of a mesh through the real camera (OpenCV projection with distortion)."""

    def __init__(self, calib: Calibration, size: tuple[int, int]):
        self.cal = calib.scaled(*size)
        self.size = size
        T_cam_table = np.linalg.inv(self.cal.T_np)
        self.rvec = cv2.Rodrigues(T_cam_table[:3, :3])[0]
        self.tvec = T_cam_table[:3, 3]

    def project(self, points_table: np.ndarray) -> np.ndarray:
        return cv2.projectPoints(points_table.astype(np.float64), self.rvec, self.tvec, self.cal.K_np, self.cal.dist_np)[
            0
        ].reshape(-1, 2)

    def render(self, vertices_table: np.ndarray, faces: np.ndarray) -> np.ndarray:
        px = self.project(vertices_table)
        tris = np.round(px[faces] * 4).astype(np.int32)  # 2 bits of sub-pixel precision
        mask = np.zeros(self.size[::-1], np.uint8)
        # One call per triangle: fillPoly with many overlapping contours uses even-odd filling and
        # punches holes wherever front and back faces overlap.
        for tri in tris:
            cv2.fillConvexPoly(mask, tri, 1, lineType=cv2.LINE_8, shift=2)
        return mask.astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def simplify(mesh: trimesh.Trimesh, max_faces: int = 2000) -> trimesh.Trimesh:
    """Fewer faces for fast silhouette rendering (reconstructions can have 100k+ faces)."""
    if len(mesh.faces) <= max_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=max_faces)
    except Exception:
        return mesh


def resting_orientations(mesh: trimesh.Trimesh, max_candidates: int = 4) -> list[np.ndarray]:
    """Rotations (3x3) that put the mesh in its most likely stable resting poses (z up)."""
    try:
        transforms, probs = mesh.convex_hull.compute_stable_poses(n_samples=1)
        order = np.argsort(probs)[::-1][:max_candidates]
        return [transforms[i][:3, :3] for i in order]
    except Exception:  # degenerate hull: fall back to the mesh as given
        return [np.eye(3)]


def canonicalise(mesh: trimesh.Trimesh, rotation: np.ndarray) -> trimesh.Trimesh:
    """Rotate into a resting orientation, centre the bounding box at the origin, unit max extent."""
    m = mesh.copy()
    m.apply_transform(np.block([[rotation, np.zeros((3, 1))], [np.zeros((1, 3)), 1]]))
    m.apply_translation(-m.bounds.mean(axis=0))
    m.apply_scale(1.0 / max(m.extents.max(), 1e-9))
    return m


def _initial_xy(renderer: SilhouetteRenderer, mask: np.ndarray, height_guess: float) -> np.ndarray:
    """Intersect the ray through the mask centroid with the plane z = height_guess / 2."""
    ys, xs = np.nonzero(mask)
    pt = np.array([[[xs.mean(), ys.mean()]]], np.float64)
    ray = cv2.undistortPoints(pt, renderer.cal.K_np, renderer.cal.dist_np).reshape(2)
    T = renderer.cal.T_np
    origin, direction = T[:3, 3], T[:3, :3] @ np.array([ray[0], ray[1], 1.0])
    t = (height_guess / 2 - origin[2]) / direction[2]
    return (origin + t * direction)[:2]


SCALE_PARAMS = {"uniform": 1, "xyz": 3, "round": 2}


def _scales(p_scale: np.ndarray, mode: str) -> np.ndarray:
    if mode == "uniform":
        return np.full(3, np.exp(p_scale[0]))
    if mode == "round":  # diameter (x = y), height
        return np.exp(np.array([p_scale[0], p_scale[0], p_scale[1]]))
    return np.exp(p_scale)


def fit_shape(
    unit_mesh: trimesh.Trimesh,
    mask: np.ndarray,
    calib: Calibration,
    scale_mode: str = "uniform",
    yaw_period: float = 2 * np.pi,
    work_width: int = 320,
) -> tuple[np.ndarray, float]:
    """Optimise [x, y, yaw, log scale(s)] for a canonical (bbox-centred, upright) mesh. Returns params, IoU.

    scale_mode: "uniform" (meshes), "xyz" (boxes), "round" (cylinders: diameter + height, yaw unused).
    """
    n_scale = SCALE_PARAMS[scale_mode]
    h, w = mask.shape
    work = (work_width, round(work_width * h / w))
    target = cv2.resize(mask.astype(np.uint8), work, interpolation=cv2.INTER_NEAREST).astype(bool)
    renderer = SilhouetteRenderer(calib, work)
    V, F = unit_mesh.vertices, unit_mesh.faces
    ext = unit_mesh.extents

    def place(p):
        x, y, yaw = p[:3]
        scale = _scales(p[3:], scale_mode)
        Vs = V * scale
        z = ext[2] * scale[2] / 2  # resting on the table
        return Vs @ _yaw_matrix(yaw).T + np.array([x, y, z])

    def loss(p):
        return -iou(renderer.render(place(p), F), target)

    # Coarse search: scale from mask size, yaw grid, position from the mask centroid ray.
    best = None
    for log_s in np.log([0.03, 0.06, 0.1, 0.16, 0.25, 0.4]):
        xy = _initial_xy(renderer, target, np.exp(log_s) * ext[2])
        yaws = [0.0] if scale_mode == "round" else np.linspace(0, yaw_period, 8, endpoint=False)
        for yaw in yaws:
            p = np.array([xy[0], xy[1], yaw, *([log_s] * n_scale)])
            score = loss(p)
            if best is None or score < best[0]:
                best = (score, p)
    p0 = best[1]
    step = np.array([0.02, 0.02, 0.0 if scale_mode == "round" else 0.3, *([0.2] * n_scale)])
    simplex = np.vstack([p0] + [p0 + np.diag(np.where(step == 0, 1e-6, step))[i] for i in range(len(p0))])
    res = minimize(loss, p0, method="Nelder-Mead", options={"initial_simplex": simplex, "xatol": 1e-4, "fatol": 1e-4, "maxiter": 600})
    return res.x, -float(res.fun)


def fit_mesh(mesh: trimesh.Trimesh, mask: np.ndarray, calib: Calibration) -> FitResult:
    """Best stable resting orientation + uniform scale + pose for a reconstructed mesh."""
    coarse = simplify(mesh)
    best = None
    for R in resting_orientations(coarse):
        params, score = fit_shape(canonicalise(coarse, R), mask, calib)
        if best is None or score > best[1]:
            best = (R, score, params)
    R, score, (x, y, yaw, log_s) = best
    unit = canonicalise(mesh, R)  # full-detail mesh in the chosen orientation
    scaled = unit.copy()
    scaled.apply_scale(float(np.exp(log_s)))
    size = scaled.extents.copy()
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = _yaw_matrix(yaw), [x, y, size[2] / 2]
    return FitResult(scaled, pose, size, score, "mesh")


def fit_primitive(mask: np.ndarray, calib: Calibration, kind: str) -> FitResult:
    """Box or upright cylinder with free per-axis size, fitted to the mask."""
    if kind == "box":
        unit = trimesh.creation.box(extents=[1, 1, 1])
        params, score = fit_shape(unit, mask, calib, scale_mode="xyz", yaw_period=np.pi / 2)
        x, y, yaw, *log_s = params
        size = _scales(np.array(log_s), "xyz")
        mesh = trimesh.creation.box(extents=size)
    else:
        unit = trimesh.creation.cylinder(radius=0.5, height=1.0, sections=32)
        params, score = fit_shape(unit, mask, calib, scale_mode="round")
        x, y, _, *log_s = params
        size = _scales(np.array(log_s), "round")
        mesh = trimesh.creation.cylinder(radius=size[0] / 2, height=size[2], sections=32)
        yaw = 0.0
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = _yaw_matrix(yaw), [x, y, size[2] / 2]
    return FitResult(mesh, pose, np.asarray(size), score, kind)


def render_fit(fit: FitResult, calib: Calibration, size: tuple[int, int]) -> np.ndarray:
    """Silhouette of a fitted object at full resolution (for QA overlays)."""
    V = fit.mesh.vertices @ fit.pose[:3, :3].T + fit.pose[:3, 3]
    return SilhouetteRenderer(calib, size).render(V, fit.mesh.faces)
