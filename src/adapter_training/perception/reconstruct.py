"""Single-image 3D shape reconstruction.

Sam3DObjectsReconstructor  SAM 3D Objects (https://github.com/facebookresearch/sam-3d-objects). It needs
                           Linux, an NVIDIA GPU with >= 32 GB VRAM and its own conda environment
                           (PyTorch 2.5 / Kaolin / PyTorch3D), so it runs as a separate process:
                           scripts/sam3d_worker.py inside that environment.
                           Found automatically when installed as in QUICKSTART.md:
                             repo   <project>/third_party/sam-3d-objects  (with checkpoints/hf/pipeline.yaml)
                             python <conda root>/envs/sam3d-objects/bin/python  (Miniforge, Mambaforge,
                                    Miniconda or Anaconda in your home folder, or via `conda run`)
                           Override with SAM3D_REPO / SAM3D_PYTHON for other locations.
                           Only the *shape* is used; size and pose are fitted to the calibrated table (fit.py).
MeshFileReconstructor      returns a given mesh (tests, or a scanned/CAD model of a known object).

Not tested on this machine (no GPU). The worker's output handling follows the repo's notebook/inference.py.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import trimesh

from adapter_training.common import PROJECT_ROOT

WORKER = PROJECT_ROOT / "scripts" / "sam3d_worker.py"
ENV_NAME = "sam3d-objects"  # the name SAM 3D Objects' environments/default.yml creates
DEFAULT_REPO = PROJECT_ROOT / "third_party" / "sam-3d-objects"
CONDA_ROOTS = ("miniforge3", "mambaforge", "miniconda3", "anaconda3")


def find_sam3d_repo() -> Path | None:
    candidates = [os.environ.get("SAM3D_REPO"), DEFAULT_REPO]
    for c in candidates:
        if c and (Path(c) / "checkpoints" / "hf" / "pipeline.yaml").exists():
            return Path(c)
    return None


def find_sam3d_python() -> Path | None:
    env = os.environ.get("SAM3D_PYTHON")
    if env:
        return Path(env) if Path(env).exists() else None
    home = Path.home()
    for root in CONDA_ROOTS:
        python = home / root / "envs" / ENV_NAME / "bin" / "python"
        if python.exists():
            return python
    conda = shutil.which("mamba") or shutil.which("conda")
    if conda:  # ask conda where the environment is
        try:
            out = subprocess.run([conda, "run", "-n", ENV_NAME, "python", "-c", "import sys; print(sys.executable)"],
                                 capture_output=True, text=True, timeout=120)
            path = Path(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 and out.stdout.strip() else None
            if path and path.exists():
                return path
        except (subprocess.SubprocessError, OSError):
            pass
    return None


class Sam3DObjectsReconstructor:
    def __init__(self, python: str | None = None, repo: str | None = None, timeout_s: int = 900):
        self.python = Path(python) if python else find_sam3d_python()
        self.repo = Path(repo) if repo else find_sam3d_repo()
        if self.python is None or self.repo is None:
            missing = []
            if self.python is None:
                missing.append(f"the '{ENV_NAME}' conda environment's python (or set SAM3D_PYTHON)")
            if self.repo is None:
                missing.append(f"{DEFAULT_REPO} with checkpoints/hf/pipeline.yaml (or set SAM3D_REPO)")
            raise RuntimeError("SAM 3D Objects not found: " + "; ".join(missing) + ". See QUICKSTART.md, section 1.")
        self.timeout_s = timeout_s

    def __call__(self, image_rgb: np.ndarray, mask: np.ndarray, seed: int = 42) -> trimesh.Trimesh:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cv2.imwrite(str(tmp / "image.png"), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(tmp / "mask.png"), mask.astype(np.uint8) * 255)
            cmd = [str(self.python), str(WORKER), "--repo", str(self.repo), "--image", str(tmp / "image.png"),
                   "--mask", str(tmp / "mask.png"), "--out", str(tmp / "object.glb"), "--seed", str(seed)]
            # What `conda activate` would set: SAM 3D's notebook/inference.py reads CONDA_PREFIX (CUDA_HOME).
            prefix = str(self.python.resolve().parent.parent)
            env = {**os.environ, "CONDA_PREFIX": prefix, "CUDA_HOME": prefix,
                   "PATH": f"{prefix}/bin{os.pathsep}{os.environ.get('PATH', '')}"}
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s, env=env)
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


def main() -> None:
    """`uv run python -m adapter_training.perception.reconstruct`: show where SAM 3D Objects was found."""
    python, repo = find_sam3d_python(), find_sam3d_repo()
    print(f"SAM 3D python: {python or 'NOT FOUND'}")
    print(f"SAM 3D repo:   {repo or 'NOT FOUND'}")
    raise SystemExit(0 if python and repo else 1)


if __name__ == "__main__":
    main()
