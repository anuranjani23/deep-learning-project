#!/usr/bin/env bash
set -euo pipefail

# Run feature-reliance inference on the BloodMNIST shape+color augmented model.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID="${GPU_ID:-0}"
EXP_DIR="${EXP_DIR:-${ROOT_DIR}/logs_medical/shape_color_aug}"

python3 "${ROOT_DIR}/reliance_protocol.py" \
  -d bloodmnist \
  -m resnet50 \
  -l "${EXP_DIR}" \
  --cuda-no "${GPU_ID}"
