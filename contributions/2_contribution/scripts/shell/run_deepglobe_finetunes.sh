#!/usr/bin/env bash
set -euo pipefail

# Run 4 DeepGlobe (remote sensing) finetunes with up to 3 concurrent jobs.
# Optionally preprocess DeepGlobe if DOWNLOAD=1.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGS_DIR="${ROOT_DIR}/logs_radio_sensing"
GPU_IDS="${GPU_IDS:-0}"
MAX_JOBS=3
DOWNLOAD="${DOWNLOAD:-1}"
DEEPGLOBE_SRC="${DEEPGLOBE_SRC:-}"
DEEPGLOBE_DST="${DEEPGLOBE_DST:-${ROOT_DIR}/dataset/tomburgert}"

if [ "${DOWNLOAD}" = "1" ]; then
  if [ -n "${DEEPGLOBE_SRC}" ]; then
    echo "[prep] DeepGlobe from ${DEEPGLOBE_SRC} -> ${DEEPGLOBE_DST}"
    python3 "${ROOT_DIR}/scripts/preprocess_deepglobe.py" \
      --source_path "${DEEPGLOBE_SRC}" \
      --destination_path "${DEEPGLOBE_DST}"
  else
    echo "[prep] DeepGlobe via kagglehub -> ${DEEPGLOBE_DST}"
    python3 "${ROOT_DIR}/scripts/preprocess_deepglobe.py" \
      --use_kagglehub \
      --destination_path "${DEEPGLOBE_DST}"
  fi
fi

mkdir -p "${LOGS_DIR}"/{baseline,shape_removed,texture_removed,color_removed}

run_train() {
  local name="$1"
  local gpu_id="$2"
  shift 2
  echo "[train] ${name}"
  python3 "${ROOT_DIR}/training.py" \
    params.dataset=deepglobe \
    params.slurm_bypass=True \
    params.cuda_no="${gpu_id}" \
    params.max_epochs=5 \
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
  dataaug.test_augmentations=resize

# Shape removed (patch shuffle)
enqueue run_train shape_removed "${GPU_LIST[1]:-${GPU_LIST[0]}}" \
  dataaug.train_augmentations=resize_patchshuffle \
  dataaug.test_augmentations=resize_patchshuffle \
  dataaug.grid_size=4

# Texture removed (bilateral)
enqueue run_train texture_removed "${GPU_LIST[2]:-${GPU_LIST[0]}}" \
  dataaug.train_augmentations=resize_bilateral \
  dataaug.test_augmentations=resize_bilateral \
  dataaug.bilateral_d=5 \
  dataaug.sigma_color=75 \
  dataaug.sigma_space=75

# Color removed (grayscale)
enqueue run_train color_removed "${GPU_LIST[0]}" \
  dataaug.train_augmentations=resize_grayscale \
  dataaug.test_augmentations=resize_grayscale \
  dataaug.gray_alpha=1.0

# Wait for all training jobs
wait

echo "All training jobs finished."
