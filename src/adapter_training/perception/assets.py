"""Turn a fitted object into MuJoCo assets: visual mesh, convex collision parts, mass and colour."""

from pathlib import Path

import cv2
import numpy as np
import trimesh

from adapter_training.datagen.env import SceneObject
from adapter_training.perception.fit import FitResult

DEFAULT_DENSITY = 400.0  # kg/m^3 of the bounding hull: household objects are part air (bottles, boxes)


def mask_colour(image_rgb: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float]:
    """Median colour of the object's pixels (eroded mask, avoids edges)."""
    m = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if m.sum() < 20:
        m = mask
    rgb = np.median(image_rgb[m], axis=0) / 255.0
    return (*rgb.tolist(), 1.0)


def convex_parts(mesh: trimesh.Trimesh, max_parts: int = 12, threshold: float = 0.08) -> list[trimesh.Trimesh]:
    """Approximate convex decomposition (CoACD); falls back to the convex hull."""
    try:
        import coacd

        coacd.set_log_level("error")
        parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=threshold, max_convex_hull=max_parts)
        hulls = [trimesh.Trimesh(v, f).convex_hull for v, f in parts]
        return [h for h in hulls if h.volume > 1e-9] or [mesh.convex_hull]
    except Exception:
        return [mesh.convex_hull]


def build_scene_object(
    name: str,
    fit: FitResult,
    out_dir: Path,
    rgba,
    mass_kg: float | None = None,
    friction: float = 1.2,
) -> SceneObject:
    """Write mesh files for a fitted object and describe it for the simulator."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if fit.kind in ("box", "cylinder"):
        volume = float(np.prod(fit.size)) * (np.pi / 4 if fit.kind == "cylinder" else 1.0)
        mass = mass_kg or float(np.clip(DEFAULT_DENSITY * volume, 0.02, 1.5))
        return SceneObject(name=name, shape=fit.kind, size=tuple(map(float, fit.size)), mass=mass, friction=friction, rgba=tuple(rgba))

    visual = out_dir / "visual.obj"
    fit.mesh.export(visual)
    parts = convex_parts(fit.mesh)
    paths, volumes = [], []
    for i, part in enumerate(parts):
        path = out_dir / f"collision_{i}.obj"
        part.export(path)
        paths.append(str(path.resolve()))
        volumes.append(float(part.volume))
    hull_volume = float(fit.mesh.convex_hull.volume)
    mass = mass_kg or float(np.clip(DEFAULT_DENSITY * hull_volume, 0.02, 1.5))
    return SceneObject(
        name=name,
        shape="mesh",
        size=tuple(map(float, fit.size)),
        mass=mass,
        friction=friction,
        rgba=tuple(rgba),
        visual_mesh=str(visual.resolve()),
        collision_meshes=paths,
        collision_volumes=volumes,
    )
