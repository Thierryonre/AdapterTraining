"""Bake a texture for a fitted object, in layers:

  1. real pixels from the clip's frame, projected onto the faces the fixed camera actually saw
     (facing the camera, not hidden behind other faces, inside the object's mask)
  2. the reconstruction's own colours (SAM 3D Objects vertex colours, or its baked texture) elsewhere
  3. the median colour of the object's pixels if neither is available

The fixed camera sees objects from roughly the same direction at deployment, so layer 1 covers what
the policy will mostly look at; layer 2 only fills in the sides the camera never saw.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from adapter_training.video.camera import Calibration


@dataclass
class BakedTexture:
    vertices: np.ndarray  # (N, 3) object frame, split along UV seams
    faces: np.ndarray  # (F, 3)
    uvs: np.ndarray  # (N, 2) in [0, 1], OBJ convention (v up)
    image: np.ndarray  # (S, S, 3) uint8 RGB, row 0 = top (v = 1)
    seen_fraction: float  # share of surface texels coloured from the real frame


def vertex_colours(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """Per-vertex RGB (0-255) from a mesh's colours or texture, or None if it has neither."""
    visual = mesh.visual
    try:
        if visual.kind == "texture":
            visual = visual.to_color()
        if visual.kind == "vertex":
            cols = np.asarray(visual.vertex_colors)[:, :3].astype(float)
            if np.ptp(cols, axis=0).max() > 1:  # not one flat default colour
                return cols
    except Exception:
        pass
    return None


def _decimate(mesh: trimesh.Trimesh, colours: np.ndarray | None, max_faces: int):
    from adapter_training.perception.fit import simplify

    small = simplify(mesh, max_faces=max_faces)  # only if the shape is preserved
    if small is mesh:
        return mesh, colours
    if colours is not None:  # carry colours over from the nearest original vertex
        _, idx = cKDTree(mesh.vertices).query(small.vertices)
        colours = colours[idx]
    return small, colours


def bake_texture(
    mesh: trimesh.Trimesh,
    pose: np.ndarray,
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    calib: Calibration,
    median_rgb,
    size: int = 512,
    max_faces: int = 20000,
    depth_tolerance_m: float = 0.01,
) -> BakedTexture:
    """mesh: object frame (metres), pose: object -> table, calib: at the frame's resolution."""
    import xatlas

    colours = vertex_colours(mesh)
    mesh, colours = _decimate(mesh, colours, max_faces)
    vmap, faces, uvs = xatlas.parametrize(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    V = mesh.vertices[vmap]
    C = colours[vmap] if colours is not None else None

    # Project into the frame.
    h, w = frame_rgb.shape[:2]
    T_cam_obj = np.linalg.inv(calib.T_np) @ pose
    Vc = V @ T_cam_obj[:3, :3].T + T_cam_obj[:3, 3]  # camera frame
    rvec = cv2.Rodrigues(T_cam_obj[:3, :3])[0]
    px = cv2.projectPoints(V.astype(np.float64), rvec, T_cam_obj[:3, 3], calib.K_np, calib.dist_np)[0].reshape(-1, 2)

    # Depth of the nearest surface at each pixel (painter's algorithm, far to near, per-face depth).
    # A texel is "seen" if its own depth is within depth_tolerance_m of that: back faces and parts
    # hidden behind the object itself are further away and fail the test.
    face_depth = Vc[faces][:, :, 2].mean(axis=1)
    depth = np.full((h, w), np.inf, np.float32)
    for f in np.argsort(-face_depth):
        cv2.fillConvexPoly(depth, np.round(px[faces[f]] * 4).astype(np.int32), float(face_depth[f]), shift=2)
    on_object = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)  # not a hand or the table
    vz = Vc[:, 2]

    tex = np.zeros((size, size, 3), np.float32)
    written = np.zeros((size, size), bool)
    seen = 0
    median = np.asarray(median_rgb[:3], float) * 255
    frame_f = frame_rgb.astype(np.float32)
    uv_px = np.stack([uvs[:, 0] * (size - 1), (1 - uvs[:, 1]) * (size - 1)], axis=1)  # row 0 = top
    for f, tri in enumerate(faces):
        t2 = uv_px[tri]
        x0, y0 = np.floor(t2.min(axis=0)).astype(int)
        x1, y1 = np.ceil(t2.max(axis=0)).astype(int)
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)
        # Barycentric coordinates of texel centres in the UV triangle.
        a, b, c = t2
        m = np.array([b - a, c - a]).T
        if abs(np.linalg.det(m)) < 1e-9:
            continue
        l12 = np.linalg.solve(m, (pts - a).T).T
        bary = np.column_stack([1 - l12.sum(axis=1), l12])
        inside = (bary >= -1e-3).all(axis=1)
        if not inside.any():
            continue
        bary, pts = bary[inside], pts[inside].astype(int)
        pix = bary @ px[tri]  # where this texel appears in the frame
        cols = np.repeat(median[None], len(bary), axis=0)
        if C is not None:
            cols = bary @ C[tri]
        u = np.clip(np.round(pix[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.round(pix[:, 1]).astype(int), 0, h - 1)
        in_frame = (pix[:, 0] >= 0) & (pix[:, 0] < w) & (pix[:, 1] >= 0) & (pix[:, 1] < h)
        texel_depth = bary @ vz[tri]
        visible = in_frame & on_object[v, u] & (texel_depth <= depth[v, u] + depth_tolerance_m)
        if visible.any():
            cols[visible] = frame_f[v[visible], u[visible]]
            seen += int(visible.sum())
        tex[pts[:, 1], pts[:, 0]] = cols
        written[pts[:, 1], pts[:, 0]] = True

    # Bleed colours past island edges so texture filtering doesn't pull in black at the seams.
    img = tex.copy()
    filled = written.copy()
    for _ in range(4):
        grown = cv2.dilate(img, np.ones((3, 3), np.uint8))
        grown_mask = cv2.dilate(filled.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        new = grown_mask & ~filled
        img[new] = grown[new]
        filled = grown_mask
    img[~filled] = median
    return BakedTexture(V, faces.astype(np.int64), uvs, np.clip(img, 0, 255).astype(np.uint8), seen / max(written.sum(), 1))


def save_textured_obj(baked: BakedTexture, obj_path, png_path) -> None:
    """OBJ with texture coordinates (one per vertex) + the texture as PNG."""
    cv2.imwrite(str(png_path), cv2.cvtColor(baked.image, cv2.COLOR_RGB2BGR))
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in baked.vertices]
    lines += [f"vt {u:.6f} {v:.6f}" for u, v in baked.uvs]
    lines += [f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}" for a, b, c in baked.faces]
    with open(obj_path, "w") as f:
        f.write("\n".join(lines) + "\n")
