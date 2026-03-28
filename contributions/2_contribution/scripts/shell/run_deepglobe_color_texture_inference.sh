#!/usr/bin/env bash
set -euo pipefail

# Run feature-reliance inference on the DeepGlobe color+texture augmented model.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID="${GPU_ID:-0}"
EXP_DIR="${EXP_DIR:-${ROOT_DIR}/logs_radio_sensing/robust_color_texture}"

python3 "${ROOT_DIR}/reliance_protocol.py" \
  -d deepglobe \
  -m resnet50 \
  -l "${EXP_DIR}" \
  --cuda-no "${GPU_ID}"
