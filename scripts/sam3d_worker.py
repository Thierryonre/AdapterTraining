"""Run SAM 3D Objects on one image + mask and export the mesh. Runs INSIDE the sam3d-objects environment.

    $SAM3D_PYTHON scripts/sam3d_worker.py --repo $SAM3D_REPO --image image.png --mask mask.png --out object.glb

Writes object.glb (mesh with a baked UV texture if the pipeline's texture baking works here, else vertex
colours; object-local frame) and object.json (the model's own
pose estimate: rotation quaternion, translation, scale -- informational; adapter_training refits pose
and scale against the calibrated table).

Follows notebook/inference.py and sam3d_objects/pipeline/inference_pipeline.py from the repo. Untested
here (requires Linux + a >= 32 GB GPU); if their API changes, adjust `run()` below.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def run(repo: Path, image_path: Path, mask_path: Path, out: Path, seed: int, texture_bake: bool = True) -> None:
    sys.path.insert(0, str(repo / "notebook"))
    sys.path.insert(0, str(repo))
    from inference import Inference  # noqa: E402  (repo's notebook/inference.py)

    inference = Inference(str(repo / "checkpoints" / "hf" / "pipeline.yaml"), compile=False)
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L")) > 127
    output, colour_source = None, None
    if texture_bake:
        # Same call as Inference.__call__ but with the pipeline's texture baking on (UV texture in the
        # GLB). The repo's wrapper turns it off; if it fails here, fall back to vertex colours.
        try:
            rgba = inference.merge_mask_to_rgba(image, mask)
            output = inference._pipeline.run(
                rgba, None, seed, stage1_only=False, with_mesh_postprocess=False, with_texture_baking=True,
                with_layout_postprocess=False, use_vertex_color=False, stage1_inference_steps=None, pointmap=None,
            )
            colour_source = "baked_texture"
        except Exception as e:  # noqa: BLE001
            print(f"texture baking failed ({type(e).__name__}: {e}); using vertex colours", file=sys.stderr)
    if output is None:
        output = inference(image, mask, seed=seed)  # vertex colours
        colour_source = "vertex_colours"

    glb = output.get("glb")
    if glb is None:
        raise RuntimeError("SAM 3D Objects returned no mesh ('glb'); is 'mesh' in decode_formats?")
    glb.export(str(out))

    def to_list(x):
        try:
            return np.asarray(x.detach().cpu()).reshape(-1).tolist()
        except AttributeError:
            return np.asarray(x).reshape(-1).tolist()

    info = {k: to_list(output[k]) for k in ("rotation", "translation", "scale") if k in output}
    info["colour_source"] = colour_source
    out.with_suffix(".json").write_text(json.dumps(info))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-texture-bake", action="store_true", help="Skip the pipeline's texture baking")
    a = p.parse_args()
    run(a.repo, a.image, a.mask, a.out, a.seed, texture_bake=not a.no_texture_bake)
