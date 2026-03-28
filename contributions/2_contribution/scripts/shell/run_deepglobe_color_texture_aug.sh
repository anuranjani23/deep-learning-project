#!/usr/bin/env bash
set -euo pipefail

# Robust DeepGlobe finetune based on observed feature reliance.
# Default combo targets the most impactful features (color + texture).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-15}"
P_AUG="${P_AUG:-0.5}"
COMBO="${COMBO:-color_texture}"  # options: color_texture, shape_texture, shape_color

case "${COMBO}" in
  color_texture)
    TRAIN_AUG="resize_bilateral_grayscale"
    LOG_NAME="robust_color_texture"
    ;;
  shape_texture)
    TRAIN_AUG="resize_patchshuffle_bilateral"
    LOG_NAME="robust_shape_texture"
    ;;
  shape_color)
    TRAIN_AUG="resize_patchshuffle_grayscale"
    LOG_NAME="robust_shape_color"
    ;;
  *)
    echo "Unknown COMBO=${COMBO}. Use color_texture, shape_texture, or shape_color." >&2
    exit 1
    ;;
esac

LOGS_DIR="${ROOT_DIR}/logs_radio_sensing/${LOG_NAME}"
mkdir -p "${LOGS_DIR}"

echo "[train] deepglobe ${COMBO} aug (p=${P_AUG}, epochs=${EPOCHS}) -> ${LOGS_DIR}"

python3 "${ROOT_DIR}/training.py" \
  params.dataset=deepglobe \
  params.slurm_bypass=True \
  params.cuda_no="${GPU_ID}" \
  params.max_epochs="${EPOCHS}" \
  model.name=resnet50 \
  model.timm_pretrained=True \
  logging.exp_dir="${LOGS_DIR}" \
  logging.save_checkpoint=True \
  dataaug.train_augmentations="${TRAIN_AUG}" \
  dataaug.test_augmentations=resize \
  dataaug.p="${P_AUG}" \
  dataaug.grid_size=4 \
  dataaug.gray_alpha=1.0 \
  dataaug.bilateral_d=5 \
  dataaug.sigma_color=75 \
  dataaug.sigma_space=75
