#!/usr/bin/env bash
set -euo pipefail

source scripts/hf_env.sh
start_training_log

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DATA_DIR="${DATA_DIR:-data/paperhound_dpo}"
SAVE_DIR="${SAVE_DIR:-checkpoints/paperhound-smollm2-135m-dpo}"
# sft -> dpo: load the cited-chunks sft *merged* model as the base (smollm2 sft was pushed merged,
# so no adapter to merge here — the merged checkpoint is already the sft model).
MODEL_PATH="${MODEL_PATH:-Pradheep1647/smollm2-135m-instruct-paper-cited-chunks-v1-sft-lr2e-5-ep8-lora32a64-seq4096-mbs8-merged}"
# display name for the pushed repo / model card (keep it short; MODEL_PATH is the merged sft).
BASE_MODEL="${BASE_MODEL:-HuggingFaceTB/SmolLM2-135M-Instruct}"
DATASET_ID="${DATASET_ID:-paperbd/paper_preference_150K-v1}"
HYPERPARAMS="${HYPERPARAMS:-sft-dpo-lr5e-6-ep1-beta0.1-lora16a32-seq1024}"

require_hf_env

# smollm2 is not quantized and supports sdpa -> --no-dequantize --attn sdpa.
# small enough to use a bigger micro-batch than the 20b.
accelerate launch --num_processes "$NPROC_PER_NODE" \
  scripts/train_dpo.py \
  --data-dir "$DATA_DIR" \
  --model-path "$MODEL_PATH" \
  --output-dir "$SAVE_DIR" \
  --attn sdpa \
  --no-dequantize \
  --beta 0.1 \
  --lr 5e-6 \
  --epochs 1 \
  --micro-batch 16 \
  --grad-accum 1 \
  --max-length 1024 \
  --lora-rank 16 \
  --lora-alpha 32 \
  "$@"

python scripts/plot_training_log.py --log "$LOG_FILE" --out-dir artifacts/smollm2-135m-dpo

if [ "${PUSH_TO_HF:-1}" = "1" ]; then
  PRIVATE_FLAG=()
  if [ "${HF_PRIVATE:-0}" = "1" ]; then
    PRIVATE_FLAG=(--private)
  fi
  python scripts/push_to_hf.py \
    --model-dir "$SAVE_DIR" \
    --base-model "$BASE_MODEL" \
    --dataset-id "$DATASET_ID" \
    --hyperparams "$HYPERPARAMS" \
    "${PRIVATE_FLAG[@]}"
fi
