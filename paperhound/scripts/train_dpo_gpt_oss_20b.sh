#!/usr/bin/env bash
set -euo pipefail

source scripts/hf_env.sh
start_training_log

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DATA_DIR="${DATA_DIR:-data/paperhound_dpo}"
SAVE_DIR="${SAVE_DIR:-checkpoints/paperhound-gpt-oss-20b-dpo}"
MODEL_PATH="${MODEL_PATH:-openai/gpt-oss-20b}"
DATASET_ID="${DATASET_ID:-paperbd/paper_preference_150K-v1}"
HYPERPARAMS="${HYPERPARAMS:-dpo-lr5e-6-ep1-beta0.1-lora16a32-seq1024}"

require_hf_env

# multi-gpu = data parallel via accelerate; lora base stays frozen MXFP4.
accelerate launch --num_processes "$NPROC_PER_NODE" \
  scripts/train_dpo.py \
  --data-dir "$DATA_DIR" \
  --model-path "$MODEL_PATH" \
  --output-dir "$SAVE_DIR" \
  --beta 0.1 \
  --lr 5e-6 \
  --epochs 1 \
  --micro-batch 4 \
  --grad-accum 4 \
  --max-length 1024 \
  --lora-rank 16 \
  --lora-alpha 32 \
  "$@"

python scripts/plot_training_log.py --log "$LOG_FILE" --out-dir artifacts/gpt-oss-20b-dpo

if [ "${PUSH_TO_HF:-1}" = "1" ]; then
  PRIVATE_FLAG=()
  if [ "${HF_PRIVATE:-0}" = "1" ]; then
    PRIVATE_FLAG=(--private)
  fi
  python scripts/push_to_hf.py \
    --model-dir "$SAVE_DIR" \
    --base-model "$MODEL_PATH" \
    --dataset-id "$DATASET_ID" \
    --hyperparams "$HYPERPARAMS" \
    "${PRIVATE_FLAG[@]}"
fi
