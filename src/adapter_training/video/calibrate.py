"""Calibrate the fixed camera: intrinsics + camera pose relative to the table (AprilTag).

1. (Recommended) Intrinsics from a checkerboard video, moving the board around the view:
       uv run at-calibrate intrinsics data/calib/checkerboard.mp4 --pattern 9x6 --square 0.025 \
           --out data/calib/intrinsics.json
2. Table pose from a short clip (or the start of a demo) with the AprilTag flat on the table,
   top edge pointing away from the robot side:
       uv run at-calibrate table data/calib/tag.mp4 --tag-size 0.16 --intrinsics data/calib/intrinsics.json \
           --out data/calib/camera.json
   Without a checkerboard, use an approximate field of view instead: --hfov 70

The camera must not move between calibration and the demonstrations. The tag can be removed afterwards.
"""

import argparse
import json
from pathlib import Path

import cv2

import adapter_training  # noqa: F401  (TLS setup)
from adapter_training.video.camera import (
    Calibration,
    draw_table_axes,
    intrinsics_from_checkerboard,
    intrinsics_from_fov,
    table_pose_from_frames,
)
from adapter_training.video.pipeline import read_video


def cmd_intrinsics(args) -> None:
    frames, _ = read_video(args.video, stride=args.stride, max_width=10_000)
    cols, rows = (int(v) for v in args.pattern.lower().split("x"))
    K, dist, rms = intrinsics_from_checkerboard(frames, (cols, rows), args.square)
    h, w = frames[0].shape[:2]
    Calibration(w, h, K.tolist(), dist.tolist()).save(args.out)
    print(f"RMS reprojection {rms:.3f} px (good: < 0.5). fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} -> {args.out}")


def cmd_table(args) -> None:
    frames, _ = read_video(args.video, stride=args.stride, max_frames=args.max_frames, max_width=10_000)
    h, w = frames[0].shape[:2]
    if args.intrinsics:
        base = Calibration.load(args.intrinsics)
        if (base.width, base.height) != (w, h):
            base = base.scaled(w, h)
        K, dist = base.K_np, base.dist_np
    elif args.hfov:
        K, dist = intrinsics_from_fov(w, h, args.hfov)
    else:
        raise SystemExit("Pass --intrinsics (checkerboard calibration) or --hfov (approximate field of view).")

    corners = getattr(args, "corners", None)
    if isinstance(corners, str):
        corners = [float(v) for v in corners.split(",")]
    if corners is not None:
        corners = [corners[i:i + 2] for i in range(0, 8, 2)] if len(corners) == 8 else corners
    T, err, n = table_pose_from_frames(frames, args.tag_size, K, dist, args.tag_family, args.tag_id,
                                       robot_side=getattr(args, "robot_side", None), corners_px=corners)
    calib = Calibration(w, h, K.tolist(), dist.tolist(), T.tolist(), err, n)
    calib.save(args.out)
    check = Path(args.out).with_suffix(".png")
    cv2.imwrite(str(check), cv2.cvtColor(draw_table_axes(frames[0], calib), cv2.COLOR_RGB2BGR))
    distance = float((T[:3, 3] ** 2).sum() ** 0.5)
    print(json.dumps({"tag_frames": n, "reprojection_px": round(err, 3), "camera_distance_m": round(distance, 3)}))
    print(f"Saved {args.out}. Check {check}: x (red) should point away from the robot, z (blue) up.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("intrinsics", help="Camera intrinsics from a checkerboard video")
    p.add_argument("video", type=Path)
    p.add_argument("--pattern", default="9x6", help="Inner corners, cols x rows")
    p.add_argument("--square", type=float, default=0.025, help="Square size in metres")
    p.add_argument("--stride", type=int, default=10, help="Use every Nth frame")
    p.add_argument("--out", type=Path, default=Path("data/calib/intrinsics.json"))
    p.set_defaults(func=cmd_intrinsics)

    p = sub.add_parser("table", help="Camera pose relative to the table from an AprilTag")
    p.add_argument("video", type=Path)
    p.add_argument("--tag-size", type=float, required=True, help="Black square edge length in metres")
    p.add_argument("--tag-family", default="36h11")
    p.add_argument("--tag-id", type=int, default=0, help="Table tag ID (keep it different from the object tag)")
    p.add_argument("--intrinsics", type=Path)
    p.add_argument("--hfov", type=float, help="Approximate horizontal field of view in degrees")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--robot-side", choices=["top", "bottom", "left", "right"],
                   help="Image edge nearest the robot. If the tag can't be decoded (too small/blurred), its "
                        "black square is used instead, top edge away from the robot")
    p.add_argument("--corners", help="Override: the tag's black-square corners in pixels, 'x1,y1,x2,y2,x3,y3,x4,y4' "
                                     "(any order; needs --robot-side)")
    p.add_argument("--out", type=Path, default=Path("data/calib/camera.json"))
    p.set_defaults(func=cmd_table)

    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
