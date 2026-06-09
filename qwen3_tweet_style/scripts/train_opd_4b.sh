#!/usr/bin/env bash
set -euo pipefail

# on-policy (speculative) distillation of the *trained* qwen3-4b tweet-style model
# (student, warm-started from the sft adapter) using the *trained* qwen3-30b-a3b tweet-style
# model as the teacher. native verl distillation: the student rolls out via sglang, the
# teacher scores top-k logprobs over the rollout, and a token-level forward-kl (the SKD loss,
# arXiv:2410.11325) pulls the student toward the teacher. K=25 per the paper.
#
# topology: 2x 80GB gpu. student (actor+rollout+ref) on the trainer pool (1 gpu), 30b teacher
# served on the distillation pool (1 gpu). run from qwen3_tweet_style/ on the pod.

ENV_FILE="${ENV_FILE:-.env}"
[ -f "$ENV_FILE" ] && { set -a; source "$ENV_FILE"; set +a; }

DATA_DIR="${DATA_DIR:-data/opd}"
STUDENT_PATH="${STUDENT_PATH:-checkpoints/qwen3-4b-sft-merged}"   # merged trained 4b sft
TEACHER_PATH="${TEACHER_PATH:-checkpoints/qwen3-30b-a3b-sft-merged}" # merged trained 30b sft
SAVE_DIR="${SAVE_DIR:-checkpoints/qwen3-4b-opd}"
REWARD_PATH="${REWARD_PATH:-scripts/reward_zero.py}"
TOPK="${TOPK:-25}"
ROLLOUT_N="${ROLLOUT_N:-4}"
EPOCHS="${EPOCHS:-4}"

python -m verl.trainer.main_ppo \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/val.parquet" \
  data.prompt_key=prompt \
  data.train_batch_size=64 \
  data.max_prompt_length=256 \
  data.max_response_length=128 \
  actor_rollout_ref.model.path="$STUDENT_PATH" \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=64 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.load_format=safetensors \
  custom_reward_function.path="$REWARD_PATH" \
  custom_reward_function.name=compute_score \
  algorithm.adv_estimator=grpo \
  distillation.enabled=True \
  distillation.n_gpus_per_node=1 \
  distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path="$TEACHER_PATH" \
  distillation.teacher_models.teacher_model.inference.name=sglang \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.85 \
  distillation.distillation_loss.loss_mode=forward_kl_topk \
  distillation.distillation_loss.topk="$TOPK" \
  distillation.distillation_loss.use_policy_gradient=False \
  distillation.distillation_loss.use_task_rewards=False \
  trainer.logger=console \
  trainer.project_name=qwen3-opd \
  trainer.experiment_name=qwen3-4b-opd-skd \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.default_local_dir="$SAVE_DIR" \
  trainer.save_freq=10 \
  trainer.test_freq=5 \
  trainer.total_epochs="$EPOCHS" \
  "$@"
