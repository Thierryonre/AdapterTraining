"""Object perception for one clip: names -> masks -> 3D shape -> metric pose on the table -> sim assets.

Writes <out>/scenes/<clip stem>.json (read by `at-datagen prepare --scenes`) and an overlay image
<out>/scenes/<clip stem>_objects.png showing each fitted object's outline on the first frame.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from adapter_training.datagen.env import SceneObject
from adapter_training.perception.assets import build_scene_object
from adapter_training.perception.fit import FitResult, fit_mesh, fit_primitive, render_fit
from adapter_training.perception.objects import ObjectInfo
from adapter_training.video.camera import Calibration


@dataclass
class PerceptionConfig:
    frame_index: int = 0  # frame used for perception (objects visible, hands away)
    min_mesh_iou: float = 0.80  # below this the reconstructed mesh is replaced by a box/cylinder
    min_primitive_iou: float = 0.65  # below this the object is rejected
    size_range_m: tuple[float, float] = (0.01, 0.6)
    hint_margin: float = 0.05  # prefer the described shape if its IoU is within this of the other
    texture_size: int = 512  # texture resolution baked per object
    asset_cache: dict = field(default_factory=dict)  # object name -> SceneObject, shared across clips


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "object"


def read_frame(video: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError(f"cannot read frame {index} of {video}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _best_primitive(mask, calib, hint: str, cfg: PerceptionConfig) -> FitResult:
    fits = {kind: fit_primitive(mask, calib, kind) for kind in ("cylinder", "box")}
    best = max(fits.values(), key=lambda f: f.iou)
    if hint in fits and fits[hint].iou >= best.iou - cfg.hint_margin:
        return fits[hint]  # a single view can't tell a box from a cylinder well; trust the description
    return best


def perceive_clip(
    video: Path,
    calib: Calibration,
    objects: list[ObjectInfo],
    segmenter,
    reconstructor,
    out_dir: Path,
    cfg: PerceptionConfig,
) -> dict:
    """Returns the scene dict ({"objects": [...], "qa": {...}}) and writes it to out_dir/scenes/."""
    frame = read_frame(video, cfg.frame_index)
    h, w = frame.shape[:2]
    cal = calib.scaled(w, h) if (w, h) != (calib.width, calib.height) else calib
    found = segmenter(frame, [o.name for o in objects])
    scene_objects, qa = [], {}
    overlay = frame.copy()
    # One scene object per instance: identical objects (count > 1) become <name>_1, <name>_2, ... numbered
    # left to right in the image, so the numbering is stable when clips start from similar layouts.
    instances = []
    for info in objects:
        count = max(1, getattr(info, "count", 1))
        masks = found.get(info.name, [])[:count]
        masks.sort(key=lambda m: float(np.nonzero(m[0])[1].mean()))
        for k in range(count):
            name = slug(info.name) if count == 1 else f"{slug(info.name)}_{k + 1}"
            if k < len(masks):
                instances.append((name, info, *masks[k]))
            else:
                qa[name] = {"ok": False, "reason": f"'{info.name}' found {len(masks)} time(s) in the first frame, "
                                                   f"expected {count}"}
    # Identical objects share one shape: reconstruct every instance, keep the mesh that fits its own outline
    # best, and fit that mesh to all of them (seen from above, a wrong height barely changes an outline).
    meshes, shared = {}, {}
    if reconstructor is not None:
        for name, info, mask, _ in instances:
            try:
                meshes[name] = reconstructor(frame, mask)
                if max(1, getattr(info, "count", 1)) > 1:
                    iou = fit_mesh(meshes[name], mask, cal).iou
                    if iou > shared.get(info.name, (None, -1.0))[1]:
                        shared[info.name] = (meshes[name], iou, name)
            except Exception as e:  # noqa: BLE001  (reported per object below)
                meshes[name] = e
    for name, info, mask, score in instances:
        entry = {"segmentation_score": score, "mask_pixels": int(mask.sum())}
        fit, mesh_iou = None, None
        if reconstructor is not None:
            try:
                mesh = shared[info.name][0] if info.name in shared else meshes[name]
                if isinstance(mesh, Exception):
                    raise mesh
                if info.name in shared:
                    entry["shape_from"] = shared[info.name][2]
                fit = fit_mesh(mesh, mask, cal)
                mesh_iou = fit.iou
            except Exception as e:  # reconstruction or fitting failed: fall back to primitives
                entry["reconstruction_error"] = f"{type(e).__name__}: {e}"[:300]
        if fit is None or fit.iou < cfg.min_mesh_iou:
            fit = _best_primitive(mask, cal, info.shape_hint, cfg)
        entry.update(kind=fit.kind, iou=round(fit.iou, 3), mesh_iou=None if mesh_iou is None else round(mesh_iou, 3))
        entry["size_cm"] = np.round(fit.size * 100, 1).tolist()
        lo, hi = cfg.size_range_m
        if fit.iou < cfg.min_primitive_iou:
            entry.update(ok=False, reason=f"no shape matched its outline well (IoU {fit.iou:.2f})")
        elif not np.all((fit.size >= lo) & (fit.size <= hi)):
            entry.update(ok=False, reason=f"implausible size {entry['size_cm']} cm")
        else:
            entry["ok"] = True
            if name not in cfg.asset_cache:  # build assets once per object per skill
                obj_model, asset_info = build_scene_object(
                    name, fit, out_dir / "objects" / name, frame, mask, cal, info.mass_kg, texture_size=cfg.texture_size
                )
                cfg.asset_cache[name] = obj_model
                entry.update(asset_info)
            obj: SceneObject = cfg.asset_cache[name]
            pose = fit.pose.copy()
            pose[2, 3] = obj.size[2] / 2  # resting on the table at the cached model's height
            scene_objects.append({**obj.to_dict(), "pose": pose.tolist()})
            outline = render_fit(fit, cal, (w, h)).astype(np.uint8)
            contours, _ = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
            m_contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, m_contours, -1, (255, 0, 255), 1)
            x, y = contours[0][0][0] if contours else (10, 20)
            cv2.putText(overlay, f"{name} {fit.kind} IoU {fit.iou:.2f}", (int(x), int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        qa[name] = entry
    scene = {"objects": scene_objects, "qa": qa, "frame_index": cfg.frame_index}
    scenes = out_dir / "scenes"
    scenes.mkdir(parents=True, exist_ok=True)
    (scenes / f"{video.stem}.json").write_text(json.dumps(scene, indent=2))
    cv2.imwrite(str(scenes / f"{video.stem}_objects.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return scene
