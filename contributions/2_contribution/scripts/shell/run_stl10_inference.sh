#!/usr/bin/env bash
set -euo pipefail

# Run reliance protocol (baseline/shape/texture/color) on STL10 finetuned models.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGS_DIR="${ROOT_DIR}/logs_cv"
GPU_ID="${GPU_ID:-0}"

infer_one() {
  local name="$1"
  echo "[protocol] ${name}"
  python3 "${ROOT_DIR}/reliance_protocol.py" \
    -d stl10 \
    -m resnet50 \
    -l "${LOGS_DIR}/${name}" \
    --cuda-no "${GPU_ID}"
}

infer_one baseline
infer_one shape_removed
infer_one texture_removed
infer_one color_removed

echo "All inference runs finished."
