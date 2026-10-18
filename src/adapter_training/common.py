"""Shared paths and helpers."""

import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
# Overridable so datasets and outputs can live on a separate disk (e.g. a Modal Volume, see modal_app.py).
LEROBOT_DIR = Path(os.environ.get("AT_LEROBOT_DIR", DATA_DIR / "lerobot"))
OUTPUTS_DIR = Path(os.environ.get("AT_OUTPUTS_DIR", PROJECT_ROOT / "outputs"))


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_video_backend() -> str:
    # torchcodec needs FFmpeg shared libraries, which are rarely installed on Windows.
    return os.environ.get("AT_VIDEO_BACKEND") or ("pyav" if sys.platform == "win32" else "torchcodec")


def dataset_root(repo_id: str) -> Path:
    """Local folder for a converted dataset, e.g. local/mimicgen_stack_d1 -> data/lerobot/mimicgen_stack_d1."""
    return LEROBOT_DIR / repo_id.split("/")[-1]
