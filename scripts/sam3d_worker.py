"""Run SAM 3D Objects on one image + mask and export the mesh. Runs INSIDE the sam3d-objects environment.

    $SAM3D_PYTHON scripts/sam3d_worker.py --repo $SAM3D_REPO --image image.png --mask mask.png --out object.glb

Writes object.glb (mesh with vertex colours, object-local frame) and object.json (the model's own
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


def run(repo: Path, image_path: Path, mask_path: Path, out: Path, seed: int) -> None:
    sys.path.insert(0, str(repo / "notebook"))
    sys.path.insert(0, str(repo))
    from inference import Inference  # noqa: E402  (repo's notebook/inference.py)

    inference = Inference(str(repo / "checkpoints" / "hf" / "pipeline.yaml"), compile=False)
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L")) > 127
    output = inference(image, mask, seed=seed)

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
    out.with_suffix(".json").write_text(json.dumps(info))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    run(a.repo, a.image, a.mask, a.out, a.seed)
