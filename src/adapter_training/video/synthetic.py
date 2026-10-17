"""Synthetic handover demos for testing the pipeline without filming anything.

Scene: a 14 x 5 x 4 cm bar lies on the table with AprilTag id 1 (4.4 cm) on its top face. The left
hand pinches its near end, lifts it to the middle, the right hand takes the far end, the left lets go.
The table tag (id 0, 16 cm) defines the table frame.

Creates, in data/synthetic/:
  tag.mp4                       1 s clip of the table tag, for `at-calibrate table`
  handover_<i>.mp4              drawn hand keypoints + the bar's tag (the bar moves between demos)
  handover_<i>_tracks.npz       ground-truth hand tracks with noise and a dropout, named so that
                                `at-video2robot --reuse-tracks --out data/synthetic` skips the hand tracker

    uv run at-synthetic-demo --demos 3
    uv run at-calibrate table data/synthetic/tag.mp4 --tag-size 0.16 --hfov 70 --out data/synthetic/camera.json
    uv run at-video2robot data/synthetic/handover_*.mp4 --calib data/synthetic/camera.json --reuse-tracks \
        --out data/synthetic --object-tag-id 1 --object-tag-size 0.044 \
        --task "Pass the bar from the left hand to the right hand."
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from adapter_training.common import DATA_DIR
from adapter_training.video.camera import R_TAG_TABLE, intrinsics_from_fov
from adapter_training.video.hands import save_tracks
from adapter_training.video.pipeline import VideoWriter

W, H, FPS, HFOV = 1280, 720, 30, 70.0
TABLE_TAG, OBJECT_TAG = (0, 0.16), (1, 0.044)  # (id, black square size in metres)
BAR_SIZE = np.array([0.14, 0.05, 0.04])
CAM_POS = np.array([0.95, 0.05, 0.55])  # opposite side of the table from the robot, looking back at it

# Hand orientations: left pinches the bar's side faces from behind, right takes the far end from the side.
APPROACH = {"left": np.array([1.0, 0.0, -0.5]), "right": np.array([0.6, 0.8, 0.0])}
CLOSING = {"left": np.array([0.0, 1.0, 0.0]), "right": np.array([0.0, 0.0, 1.0])}
OPEN, CLOSED = 0.09, 0.05  # thumb-index distance (m); "closed" = holding the 5 cm bar


def camera_pose() -> np.ndarray:
    z = -CAM_POS / np.linalg.norm(CAM_POS)  # look at the table tag
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


def draw_tag(img: np.ndarray, tag_id: int, size: float, T_table_tagframe: np.ndarray, T_table_cam, K) -> None:
    """Warp a 36h11 tag lying in the x-y plane of T_table_tagframe (top edge towards its +x) into img."""
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), tag_id, 400)
    border = 50
    tag = cv2.copyMakeBorder(tag, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
    s = size / 2
    corners_tag = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])  # tl, tr, br, bl in tag axes
    local = (R_TAG_TABLE.T @ corners_tag.T).T  # tag axes -> frame with x = tag up
    world = local @ T_table_tagframe[:3, :3].T + T_table_tagframe[:3, 3]
    dst = project(world, T_table_cam, K).astype(np.float32)
    src = np.float32([[border, border], [border + 400, border], [border + 400, border + 400], [border, border + 400]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(tag, M, (img.shape[1], img.shape[0]))
    mask = cv2.warpPerspective(np.full_like(tag, 255), M, (img.shape[1], img.shape[0])) > 0
    img[mask] = warped[mask][:, None]


def hand_keypoints(grasp: np.ndarray, approach: np.ndarray, closing: np.ndarray, aperture: float) -> np.ndarray:
    """Plausible 21-point pinching hand around a grasp point (only thumb/index points matter)."""
    a = approach / np.linalg.norm(approach)
    c = closing - np.dot(closing, a) * a
    c /= np.linalg.norm(c)
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
        mcp = grasp - 0.075 * a - offset * c + 0.01 * n
        kp[finger] = mcp
        kp[finger + 1] = mcp + 0.03 * a - 0.01 * n
        kp[finger + 2] = mcp + 0.04 * a - 0.03 * n
        kp[finger + 3] = mcp + 0.03 * a - 0.045 * n
    return kp


def handover_script(bar_xy: np.ndarray) -> tuple[dict, int]:
    """Keyframes (time s, grasp point, aperture) per hand for a bar lying along x at bar_xy."""
    aL = APPROACH["left"] / np.linalg.norm(APPROACH["left"])
    aR = APPROACH["right"] / np.linalg.norm(APPROACH["right"])
    grasp_L = np.array([bar_xy[0] - 0.035, bar_xy[1], BAR_SIZE[2] / 2])  # near end, mid-height
    hold_L = np.array([0.13, 0.04, 0.18])
    grasp_R = hold_L + 0.095 * aL  # far end of the bar hanging along the left gripper (roughly)
    keys = {
        "left": [
            (0.0, [0.00, 0.26, 0.15], OPEN),
            (1.0, grasp_L - 0.10 * aL, OPEN),
            (1.6, grasp_L, OPEN),
            (2.0, grasp_L, CLOSED),
            (3.2, hold_L, CLOSED),
            (5.0, hold_L, CLOSED),
            (5.3, hold_L, OPEN),
            (6.5, [0.02, 0.20, 0.20], OPEN),
        ],
        "right": [
            (0.0, [0.00, -0.26, 0.15], OPEN),
            (3.2, [0.00, -0.26, 0.15], OPEN),
            (4.0, grasp_R - 0.10 * aR, OPEN),
            (4.5, grasp_R, OPEN),
            (4.8, grasp_R, CLOSED),
            (5.3, grasp_R, CLOSED),
            (6.5, [0.18, -0.15, 0.20], CLOSED),
        ],
    }
    T = int(6.5 * FPS)
    t = np.arange(T) / FPS
    out = {}
    for side, k in keys.items():
        times = np.array([x[0] for x in k])
        pts = np.array([x[1] for x in k], dtype=float)
        pos = np.stack([np.interp(t, times, pts[:, d]) for d in range(3)], axis=1)
        out[side] = (pos, np.interp(t, times, [x[2] for x in k]))
    return out, T


def make_demo(out: Path, name: str, bar_xy, bg, T_table_cam, K, rng, noise_mm: float) -> None:
    script, T = handover_script(bar_xy)
    T_cam_table = np.linalg.inv(T_table_cam)
    tracks = {}
    for side, (pos, aperture) in script.items():
        kp = np.stack([hand_keypoints(pos[t], APPROACH[side], CLOSING[side], aperture[t]) for t in range(T)])
        kp += rng.normal(0, noise_mm / 1000, kp.shape)
        kp_cam = kp @ T_cam_table[:3, :3].T + T_cam_table[:3, 3]
        kp2d = np.stack([project(kp[t], T_table_cam, K) for t in range(T)])
        valid = np.ones(T, bool)
        if side == "right":
            valid[100:106] = False  # short tracking dropout, to exercise gap filling
        kp_cam[~valid] = np.nan
        kp2d[~valid] = np.nan
        tracks[side] = {"kp_cam": kp_cam, "kp2d": kp2d, "reproj_px": np.zeros(T), "score": valid * 1.0, "valid": valid}

    bar_top = np.eye(4)
    bar_top[:3, 3] = [bar_xy[0], bar_xy[1], BAR_SIZE[2]]
    writer = VideoWriter(out / f"{name}.mp4", FPS, (W, H))
    for t in range(T):
        frame = bg.copy()
        if t < int(1.5 * FPS):  # the bar's tag is visible until the hand reaches it
            draw_tag(frame, *OBJECT_TAG, bar_top, T_table_cam, K)
        for side, color in (("left", (0, 220, 0)), ("right", (255, 60, 60))):
            if tracks[side]["valid"][t]:
                for p in tracks[side]["kp2d"][t].astype(int):
                    cv2.circle(frame, tuple(p), 4, color, -1)
        writer.write(frame)
    writer.close()
    save_tracks(out / f"{name}_tracks.npz", tracks, FPS)


def bottle_mesh():
    """A 7 cm x 22 cm water bottle (body, shoulder, neck), bbox-centred."""
    import trimesh

    body = trimesh.creation.cylinder(radius=0.035, height=0.16, sections=48)
    body.apply_translation([0, 0, 0.08])
    shoulder = trimesh.creation.cone(radius=0.035, height=0.03, sections=48)
    shoulder.apply_translation([0, 0, 0.16])
    neck = trimesh.creation.cylinder(radius=0.015, height=0.04, sections=32)
    neck.apply_translation([0, 0, 0.20])
    mesh = trimesh.util.concatenate([body, shoulder, neck])
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    return mesh.subdivide_to_size(max_edge=0.008)  # small faces, so colour bands follow the geometry


def bottle_colours(points: np.ndarray) -> np.ndarray:
    """RGB per point of the bbox-centred bottle: red cap, white label with a green logo on +x, blue body."""
    z, x, y = points[:, 2], points[:, 0], points[:, 1]
    cols = np.tile(np.array([60, 120, 220], float), (len(points), 1))  # blue body
    label = (z > -0.05) & (z < 0.02)
    cols[label] = [235, 235, 235]
    logo = label & (x > 0.02) & (np.abs(y) < 0.015) & (z > -0.035) & (z < 0.005)
    cols[logo] = [40, 170, 60]
    cols[z > 0.09] = [210, 40, 40]  # cap
    return cols


def draw_mesh(frame: np.ndarray, mesh, colours_per_face: np.ndarray, T_table_obj: np.ndarray, T_table_cam, K) -> None:
    """Paint a mesh into the image with flat face colours (far to near)."""
    T_cam_obj = np.linalg.inv(T_table_cam) @ T_table_obj
    V = mesh.vertices
    Vc = V @ T_cam_obj[:3, :3].T + T_cam_obj[:3, 3]
    px = cv2.projectPoints(V, cv2.Rodrigues(T_cam_obj[:3, :3])[0], T_cam_obj[:3, 3], K, np.zeros(5))[0].reshape(-1, 2)
    order = np.argsort(-Vc[mesh.faces][:, :, 2].mean(axis=1))
    for f in order:
        cv2.fillConvexPoly(frame, np.round(px[mesh.faces[f]] * 4).astype(np.int32), colours_per_face[f].tolist(), shift=2)


def make_bottle_demo(out: Path, name: str, bottle_xy, bg, T_table_cam, K, rng, noise_mm: float) -> None:
    """Right hand picks up a standing water bottle and lifts it. Writes video, tracks and the GT mask."""
    from adapter_training.perception.fit import SilhouetteRenderer
    from adapter_training.video.camera import Calibration

    mesh = bottle_mesh()
    height = mesh.extents[2]
    centre = np.array([bottle_xy[0], bottle_xy[1], height / 2])
    approach, closing = np.array([1.0, 0.0, -0.2]), np.array([0.0, 1.0, 0.0])
    a = approach / np.linalg.norm(approach)
    grasp = centre + np.array([0, 0, 0.0])  # mid-height
    lift = grasp + np.array([0.0, 0.0, 0.15])
    keys = {
        "left": [(0.0, [0.00, 0.26, 0.15], OPEN), (5.0, [0.00, 0.26, 0.15], OPEN)],
        "right": [
            (0.0, [0.00, -0.26, 0.15], OPEN),
            (1.2, grasp - 0.12 * a, OPEN),
            (2.0, grasp, OPEN),
            (2.4, grasp, 0.07),
            (3.8, lift, 0.07),
            (5.0, lift, 0.07),
        ],
    }
    T = int(5.0 * FPS)
    t = np.arange(T) / FPS
    T_cam_table = np.linalg.inv(T_table_cam)
    tracks = {}
    for side, k in keys.items():
        times = np.array([x[0] for x in k])
        pts = np.array([x[1] for x in k], dtype=float)
        pos = np.stack([np.interp(t, times, pts[:, d]) for d in range(3)], axis=1)
        aperture = np.interp(t, times, [x[2] for x in k])
        app = approach if side == "right" else np.array([1.0, 0.0, -0.3])
        kp = np.stack([hand_keypoints(pos[i], app, closing, aperture[i]) for i in range(T)])
        kp += rng.normal(0, noise_mm / 1000, kp.shape)
        kp_cam = kp @ T_cam_table[:3, :3].T + T_cam_table[:3, 3]
        kp2d = np.stack([project(kp[i], T_table_cam, K) for i in range(T)])
        valid = np.ones(T, bool)
        tracks[side] = {"kp_cam": kp_cam, "kp2d": kp2d, "reproj_px": np.zeros(T), "score": valid * 1.0, "valid": valid}

    calib = Calibration(W, H, K.tolist(), [0.0] * 5, T_table_cam.tolist())
    mask = SilhouetteRenderer(calib, (W, H)).render(mesh.vertices + centre, mesh.faces)
    (out / "masks" / name).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "masks" / name / "water_bottle.png"), mask.astype(np.uint8) * 255)
    writer = VideoWriter(out / f"{name}.mp4", FPS, (W, H))
    T_table_obj = np.eye(4)
    T_table_obj[:3, 3] = centre
    face_cols = bottle_colours(mesh.triangles_center)
    still = bg.copy()
    draw_mesh(still, mesh, face_cols, T_table_obj, T_table_cam, K)
    for i in range(T):
        frame = bg.copy()
        if i < int(2.2 * FPS):  # the bottle is drawn where it stands until it's grasped
            frame = still.copy()
        for side, color in (("left", (0, 220, 0)), ("right", (255, 60, 60))):
            for p in tracks[side]["kp2d"][i].astype(int):
                cv2.circle(frame, tuple(p), 4, color, -1)
        writer.write(frame)
    writer.close()
    save_tracks(out / f"{name}_tracks.npz", tracks, FPS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "synthetic")
    parser.add_argument("--demos", type=int, default=3)
    parser.add_argument("--noise-mm", type=float, default=3.0, help="Keypoint noise added to the tracks")
    parser.add_argument("--scenario", choices=["handover", "bottle"], default="handover")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    K, _ = intrinsics_from_fov(W, H, HFOV)
    T_table_cam = camera_pose()
    bg = np.full((H, W, 3), 110, np.uint8)
    draw_tag(bg, *TABLE_TAG, np.eye(4), T_table_cam, K)

    writer = VideoWriter(args.out / "tag.mp4", FPS, (W, H))
    for _ in range(FPS):
        writer.write(bg)
    writer.close()

    if args.scenario == "bottle":
        truth = {}
        for i in range(args.demos):
            xy = np.array([0.14, -0.12]) + rng.uniform([-0.02, -0.02], [0.02, 0.02])
            make_bottle_demo(args.out, f"bottle_{i}", xy, bg, T_table_cam, K, rng, args.noise_mm)
            truth[f"bottle_{i}"] = {"bottle_centre": [*xy, float(bottle_mesh().extents[2] / 2)]}
        # What a single-image reconstruction might return: arbitrary orientation, scale and offset.
        import trimesh

        recon = bottle_mesh()
        recon.visual.vertex_colors = np.column_stack([bottle_colours(recon.vertices), np.full(len(recon.vertices), 255)]).astype(np.uint8)
        recon.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        recon.apply_transform(trimesh.transformations.rotation_matrix(0.7, [0, 0, 1]))
        recon.apply_scale(3.7)
        recon.apply_translation([1, 2, 3])
        recon.export(args.out / "reconstructed_bottle.ply")  # PLY keeps the vertex colours
        (args.out / "ground_truth.json").write_text(json.dumps(truth, indent=2))
        print(f"Wrote tag.mp4, {args.demos} bottle demos, masks/ and reconstructed_bottle.ply to {args.out}")
        return

    truth = {}
    for i in range(args.demos):
        bar_xy = np.array([0.10, 0.15]) + rng.uniform([-0.02, -0.02], [0.02, 0.02])
        make_demo(args.out, f"handover_{i}", bar_xy, bg, T_table_cam, K, rng, args.noise_mm)
        truth[f"handover_{i}"] = {"bar_center": [*bar_xy, BAR_SIZE[2] / 2]}
    (args.out / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"Wrote tag.mp4 and {args.demos} handover demos to {args.out}")


if __name__ == "__main__":
    main()
