#!/usr/bin/env bash
set -euo pipefail

source scripts/hf_env.sh
start_training_log

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DATA_DIR="${DATA_DIR:-data/paperhound}"
SAVE_DIR="${SAVE_DIR:-checkpoints/paperhound-smollm2-135m-sft}"
MODEL_PATH="${MODEL_PATH:-HuggingFaceTB/SmolLM2-135M-Instruct}"
DATASET_ID="${DATASET_ID:-paperbd/paper-cited-chunks-v1}"
HYPERPARAMS="${HYPERPARAMS:-sft-lr2e-5-ep8-lora32a64-seq2048-mbs8}"

require_hf_env

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  -m verl.trainer.sft_trainer \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/val.parquet" \
  data.messages_key=messages \
  data.micro_batch_size_per_gpu=8 \
  data.max_length=2048 \
  optim.lr=2e-5 \
  engine=fsdp \
  model.path="$MODEL_PATH" \
  model.lora_rank=32 \
  model.lora_alpha=64 \
  model.target_modules=all-linear \
  trainer.default_local_dir="$SAVE_DIR" \
  trainer.project_name=paperhound \
  trainer.experiment_name=smollm2-135m-sft \
  trainer.total_epochs=8 \
  trainer.logger=console \
  "$@"

python scripts/plot_training_log.py --log "$LOG_FILE" --out-dir artifacts/smollm2-135m-sft

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
