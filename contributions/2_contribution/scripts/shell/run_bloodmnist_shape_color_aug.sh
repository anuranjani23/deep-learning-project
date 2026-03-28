#!/usr/bin/env bash
set -euo pipefail

# Finetune ResNet50 on BloodMNIST with combined shape + color augmentation.
# Keeps baseline resize, and applies shape/color aug with probability < 1.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGS_DIR="${ROOT_DIR}/logs_medical/shape_color_aug"
GPU_ID="${GPU_ID:-0}"
P_AUG="${P_AUG:-0.5}"
EPOCHS="${EPOCHS:-10}"

mkdir -p "${LOGS_DIR}"

echo "[train] bloodmnist shape+color aug (p=${P_AUG}) -> ${LOGS_DIR}"

python3 "${ROOT_DIR}/training.py" \
  params.dataset=bloodmnist \
  params.slurm_bypass=True \
  params.cuda_no="${GPU_ID}" \
  params.max_epochs="${EPOCHS}" \
  model.name=resnet50 \
  model.timm_pretrained=True \
  logging.exp_dir="${LOGS_DIR}" \
  logging.save_checkpoint=True \
  dataaug.train_augmentations=resize_patchshuffle_grayscale \
  dataaug.test_augmentations=resize \
  dataaug.grid_size=4 \
  dataaug.gray_alpha=1.0 \
  dataaug.p="${P_AUG}"
