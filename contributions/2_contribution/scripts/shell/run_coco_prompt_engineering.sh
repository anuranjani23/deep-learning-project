#!/usr/bin/env bash
set -euo pipefail

CAPTIONS=${CAPTIONS:-./dataset/coco/coco_train2017_subset.jsonl}
MAX_SAMPLES=${MAX_SAMPLES:-200}
PROMPT_SETS=${PROMPT_SETS:-clip_basic,shape,texture,color}
LOG_DIR=${LOG_DIR:-./logs_prompt_eng}
DEVICE=${DEVICE:-}

if [[ ! -f "$CAPTIONS" ]]; then
  echo "Captions file not found: $CAPTIONS" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

run_mode() {
  local label_mode="$1"
  local out_csv="$LOG_DIR/coco_prompt_${label_mode}.csv"
  local out_log="$LOG_DIR/coco_prompt_${label_mode}.txt"

  echo "[coco prompt] label_mode=${label_mode} max_samples=${MAX_SAMPLES} prompt_sets=${PROMPT_SETS}"
  if [[ -n "$DEVICE" ]]; then
    python3 scripts/test_coco_prompt_engineering.py \
      --captions "$CAPTIONS" \
      --max-samples "$MAX_SAMPLES" \
      --prompt-sets "$PROMPT_SETS" \
      --label-mode "$label_mode" \
      --device "$DEVICE" \
      --save-csv "$out_csv" | tee "$out_log"
  else
    python3 scripts/test_coco_prompt_engineering.py \
      --captions "$CAPTIONS" \
      --max-samples "$MAX_SAMPLES" \
      --prompt-sets "$PROMPT_SETS" \
      --label-mode "$label_mode" \
      --save-csv "$out_csv" | tee "$out_log"
  fi
}

run_mode caption
run_mode noun
run_mode short

echo "All prompt-engineering comparisons saved under $LOG_DIR"
