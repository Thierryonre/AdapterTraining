"""Single-image 3D shape reconstruction.

Sam3DObjectsReconstructor  SAM 3D Objects (https://github.com/facebookresearch/sam-3d-objects). It needs
                           Linux, an NVIDIA GPU with >= 32 GB VRAM and its own conda environment
                           (PyTorch 2.5 / Kaolin / PyTorch3D), so it runs as a separate process:
                           scripts/sam3d_worker.py inside that environment. Configure with
                             SAM3D_PYTHON=/path/to/envs/sam3d-objects/bin/python
                             SAM3D_REPO=/path/to/sam-3d-objects   (contains checkpoints/hf/pipeline.yaml)
                           Only the *shape* is used; size and pose are fitted to the calibrated table (fit.py).
MeshFileReconstructor      returns a given mesh (tests, or a scanned/CAD model of a known object).

Not tested on this machine (no GPU). The worker's output handling follows the repo's notebook/inference.py.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import trimesh

from adapter_training.common import PROJECT_ROOT

WORKER = PROJECT_ROOT / "scripts" / "sam3d_worker.py"


class Sam3DObjectsReconstructor:
    def __init__(self, python: str | None = None, repo: str | None = None, timeout_s: int = 900):
        self.python = python or os.environ.get("SAM3D_PYTHON")
        self.repo = repo or os.environ.get("SAM3D_REPO")
        if not self.python or not self.repo:
            raise RuntimeError("Set SAM3D_PYTHON and SAM3D_REPO (see adapter_training/perception/reconstruct.py).")
        self.timeout_s = timeout_s

    def __call__(self, image_rgb: np.ndarray, mask: np.ndarray, seed: int = 42) -> trimesh.Trimesh:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cv2.imwrite(str(tmp / "image.png"), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(tmp / "mask.png"), mask.astype(np.uint8) * 255)
            cmd = [self.python, str(WORKER), "--repo", self.repo, "--image", str(tmp / "image.png"),
                   "--mask", str(tmp / "mask.png"), "--out", str(tmp / "object.glb"), "--seed", str(seed)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s)
            if proc.returncode != 0:
                raise RuntimeError(f"SAM 3D Objects failed:\n{proc.stderr[-2000:]}")
            info = json.loads((tmp / "object.json").read_text())
            mesh = trimesh.load(tmp / "object.glb", force="mesh")
            mesh.metadata["sam3d"] = info
            return mesh


class MeshFileReconstructor:
    def __init__(self, path: Path):
        self.path = Path(path)

    def __call__(self, image_rgb: np.ndarray, mask: np.ndarray, seed: int = 0) -> trimesh.Trimesh:
        return trimesh.load(self.path, force="mesh")
