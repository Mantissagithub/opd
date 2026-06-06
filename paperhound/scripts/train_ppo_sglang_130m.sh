#!/usr/bin/env bash
set -euo pipefail

source scripts/hf_env.sh
start_training_log

export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK="${SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK:-True}"

DATA_DIR="${DATA_DIR:-data/paperhound}"
MODEL_PATH="${MODEL_PATH:-checkpoints/paperhound-smollm2-135m-sft}"
BASE_MODEL="${BASE_MODEL:-HuggingFaceTB/SmolLM2-135M-Instruct}"
SAVE_DIR="${SAVE_DIR:-checkpoints/paperhound-smollm2-135m-ppo-sglang}"
REWARD_PATH="${REWARD_PATH:-scripts/reward_cited_chunks.py}"
N_GPUS="${N_GPUS:-1}"
DATASET_ID="${DATASET_ID:-paperbd/paper-cited-chunks-v1}"
HYPERPARAMS="${HYPERPARAMS:-ppo-sglang-lr1e-6-ep10-kl0.001-rolloutn4-tp1-mbs4}"

require_hf_env

python -m verl.trainer.main_ppo \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/val.parquet" \
  data.prompt_key=prompt \
  data.train_batch_size=64 \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  custom_reward_function.path="$REWARD_PATH" \
  custom_reward_function.name=compute_score \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.critic_warmup=0 \
  trainer.logger=console \
  trainer.project_name=paperhound \
  trainer.experiment_name=smollm2-135m-ppo-sglang \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.default_local_dir="$SAVE_DIR" \
  trainer.save_freq=5 \
  trainer.test_freq=1 \
  trainer.total_epochs=10 \
  "$@"

python scripts/plot_training_log.py --log "$LOG_FILE" --out-dir artifacts/smollm2-135m-ppo-sglang

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
