#!/usr/bin/env bash
# Install the WiLoR hand tracker (GPU machine). wilor-mini's setup.py imports pip, so it can't be
# built in uv's isolated build environment: install pip into the venv and build without isolation.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync
uv pip install pip setuptools wheel
uv pip install --no-build-isolation "git+https://github.com/warmshao/WiLoR-mini"
uv run python -c "from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline; print('WiLoR OK')"
