#!/usr/bin/env bash
# Build the SAM 3D Objects environment inside the Modal image (see modal_app.py).
# Same steps as QUICKSTART.md section 1 / SAM 3D Objects' doc/setup.md, adapted to a container:
#   - conda in /opt/conda, code in /opt/sam-3d-objects
#   - CUDA extensions (PyTorch3D, flash-attn, gsplat) compiled for H100/H200 (sm_90) without a GPU
#     attached (FORCE_CUDA=1), built against the environment's own torch (--no-build-isolation)
#   - checkpoints are not baked in: checkpoints/hf points at the cache volume, filled on first run
# Run in stages (one image layer each, so a failure doesn't redo the earlier ones):
#   install_sam3d.sh code | packages | finish   (conda + the environment itself: inline in modal_app.py)
# No `set -u`: conda's activation scripts read unset variables.
set -eo pipefail

CONDA=/opt/conda
REPO=/opt/sam-3d-objects

activate() {
  source "$CONDA/etc/profile.d/conda.sh"
  conda activate sam3d-objects
  export CUDA_HOME="$CONDA_PREFIX"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
  export FORCE_CUDA=1
  export MAX_JOBS="${MAX_JOBS:-8}"
  # SAM 3D's docs also list https://pypi.ngc.nvidia.com, which no longer resolves (pip would retry it
  # for every package); its packages are on PyPI now.
  export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
  export PIP_NO_CACHE_DIR=1
}

case "${1:?stage: code | packages | finish}" in
  code)
    activate
    cd "$REPO"
    pip install -e '.[dev]'
    ;;
  packages)
    activate
    cd "$REPO"
    # These extras compile CUDA code that imports torch at build time, so build them in this
    # environment rather than pip's isolated one (which has no torch).
    pip install hatchling hatch-requirements-txt editables
    pip install --no-build-isolation -e '.[p3d]'
    export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
    pip install --no-build-isolation -e '.[inference]'
    ;;
  finish)
    activate
    cd "$REPO"
    chmod +x ./patching/hydra && ./patching/hydra
    # Checkpoints live on the cache volume (mounted at /cache at run time).
    mkdir -p checkpoints
    ln -sfn /cache/sam3d/hf checkpoints/hf
    python -c "import torch, pytorch3d, kaolin, gsplat; print('SAM 3D env OK: torch', torch.__version__, 'cuda', torch.version.cuda)"
    ;;
esac
