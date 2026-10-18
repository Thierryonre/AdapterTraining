#!/usr/bin/env bash
# Install the WiLoR hand tracker (GPU machine). wilor-mini's setup.py imports pip, so it can't be
# built in uv's isolated build environment: install pip into the venv and build without isolation.
#
# wilor-mini's requirements cap torch at 2.5 and add opencv-python. Installed as-is, that downgrades
# this project's torch (breaking LeRobot and torchcodec) and puts a second cv2 next to mediapipe's
# opencv-contrib-python. The overrides keep the locked torch/torchvision and drop opencv-python.
#
#   bash scripts/install_wilor.sh              # normal use
#   NO_SYNC=1 bash scripts/install_wilor.sh    # venv already synced (Modal image build)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -n "${NO_SYNC:-}" ] || uv sync
uv pip install --python "$PY" pip setuptools wheel

OVERRIDES=$(mktemp)
uv pip freeze --python "$PY" | grep -E '^(torch|torchvision)==' > "$OVERRIDES"
echo 'opencv-python; sys_platform == "never"' >> "$OVERRIDES"
echo "Keeping: $(tr '\n' ' ' < "$OVERRIDES")"
uv pip install --python "$PY" --no-build-isolation --override "$OVERRIDES" "git+https://github.com/warmshao/WiLoR-mini"
rm -f "$OVERRIDES"

"$PY" -c "from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline; print('WiLoR OK')"
