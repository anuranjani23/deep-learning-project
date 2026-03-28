#!/usr/bin/env bash
set -euo pipefail

# Run 4 BloodMNIST finetunes (baseline + shape/texture/color removed) with up to 3 concurrent jobs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGS_DIR="${ROOT_DIR}/logs_medical"
# Comma-separated GPU IDs to use, e.g. "0,1,2". Defaults to "0".
GPU_IDS="${GPU_IDS:-0}"
MAX_JOBS=3

mkdir -p "${LOGS_DIR}"/{baseline,shape_removed,texture_removed,color_removed}

run_train() {
  local name="$1"
  local gpu_id="$2"
  shift 2
  echo "[train] ${name}"
  python3 "${ROOT_DIR}/training.py" \
    params.dataset=bloodmnist \
    params.slurm_bypass=True \
    params.cuda_no="${gpu_id}" \
    model.name=resnet50 \
    model.timm_pretrained=True \
    logging.exp_dir="${LOGS_DIR}/${name}" \
    logging.save_checkpoint=True \
    "$@" &
}

PIDS=()
enqueue() {
  "$@"
  PIDS+=("$!")
  while [ "${#PIDS[@]}" -ge "${MAX_JOBS}" ]; do
    wait -n
    local new_pids=()
    for pid in "${PIDS[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        new_pids+=("${pid}")
      fi
    done
    PIDS=("${new_pids[@]}")
  done
}

GPU_LIST=()
IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
if [ "${#GPU_LIST[@]}" -lt 1 ]; then
  GPU_LIST=(0)
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  AVAIL=($(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' '))
  if [ "${#AVAIL[@]}" -gt 0 ]; then
    FILTERED=()
    for gid in "${GPU_LIST[@]}"; do
      for aid in "${AVAIL[@]}"; do
        if [ "${gid}" = "${aid}" ]; then
          FILTERED+=("${gid}")
        fi
      done
    done
    GPU_LIST=("${FILTERED[@]}")
  fi
fi

if [ "${#GPU_LIST[@]}" -lt 1 ]; then
  echo "[warn] No valid GPUs found in GPU_IDS, falling back to GPU 0"
  GPU_LIST=(0)
fi

if [ "${MAX_JOBS}" -gt "${#GPU_LIST[@]}" ]; then
  MAX_JOBS="${#GPU_LIST[@]}"
fi

# Baseline (no suppression)
enqueue run_train baseline "${GPU_LIST[0]}" \
  dataaug.train_augmentations=resize \
  dataaug.test_augmentations=resize \
  params.max_epochs=10

# Shape removed (patch shuffle) - 10 epochs
enqueue run_train shape_removed "${GPU_LIST[1]:-${GPU_LIST[0]}}" \
  dataaug.train_augmentations=resize_patchshuffle \
  dataaug.test_augmentations=resize_patchshuffle \
  dataaug.grid_size=4 \
  params.max_epochs=10

# Texture removed (bilateral) - 10 epochs
enqueue run_train texture_removed "${GPU_LIST[2]:-${GPU_LIST[0]}}" \
  dataaug.train_augmentations=resize_bilateral \
  dataaug.test_augmentations=resize_bilateral \
  dataaug.bilateral_d=5 \
  dataaug.sigma_color=75 \
  dataaug.sigma_space=75 \
  params.max_epochs=10

# Color removed (grayscale)
enqueue run_train color_removed "${GPU_LIST[0]}" \
  dataaug.train_augmentations=resize_grayscale \
  dataaug.test_augmentations=resize_grayscale \
  dataaug.gray_alpha=1.0 \
  params.max_epochs=10

# Wait for all training jobs
wait

echo "All training jobs finished."
