#!/usr/bin/env bash
set -euo pipefail

source scripts/hf_env.sh
start_training_log

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DATA_DIR="${DATA_DIR:-data/paperhound}"
SAVE_DIR="${SAVE_DIR:-checkpoints/paperhound-gpt-oss-20b-sft}"
MODEL_PATH="${MODEL_PATH:-openai/gpt-oss-20b}"
DATASET_ID="${DATASET_ID:-paperbd/paper-cited-chunks-v1}"
HYPERPARAMS="${HYPERPARAMS:-sft-lr8e-6-ep4-lora16a32-seq4096-mbs1}"

require_hf_env

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  -m verl.trainer.sft_trainer \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/val.parquet" \
  data.messages_key=messages \
  data.micro_batch_size_per_gpu=1 \
  data.max_length=4096 \
  optim.lr=8e-6 \
  engine=fsdp \
  model.path="$MODEL_PATH" \
  model.enable_gradient_checkpointing=True \
  model.lora_rank=16 \
  model.lora_alpha=32 \
  model.target_modules=all-linear \
  trainer.default_local_dir="$SAVE_DIR" \
  trainer.project_name=paperhound \
  trainer.experiment_name=gpt-oss-20b-sft \
  trainer.total_epochs=4 \
  trainer.logger=console \
  "$@"

python scripts/plot_training_log.py --log "$LOG_FILE" --out-dir artifacts/gpt-oss-20b-sft

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
