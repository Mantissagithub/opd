#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/paperhound}"
LOG_FILE="${LOG_FILE:-training.log}"
GPU_COUNT="${GPU_COUNT:-$(python - <<'PY'
try:
    import subprocess

    out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    print(max(1, len([line for line in out.splitlines() if line.startswith("GPU ")])))
except Exception:
    print(1)
PY
)}"

echo "paperhound run starting"
echo "gpu_count=$GPU_COUNT"

uv run python scripts/prepare_verl_data.py --local-dir "$DATA_DIR"

RUN_SMALL="${RUN_SMALL:-1}"
RUN_BIG="${RUN_BIG:-1}"
RUN_PPO="${RUN_PPO:-1}"

if [ "$RUN_SMALL" = "1" ]; then
  NPROC_PER_NODE="${SMALL_NPROC_PER_NODE:-1}" LOG_FILE="$LOG_FILE" uv run bash scripts/train_sft_130m.sh
  if [ "$RUN_PPO" = "1" ]; then
    N_GPUS="${SMALL_N_GPUS:-1}" LOG_FILE="$LOG_FILE" uv run bash scripts/train_ppo_sglang_130m.sh
  fi
fi

if [ "$RUN_BIG" = "1" ]; then
  NPROC_PER_NODE="${BIG_NPROC_PER_NODE:-$GPU_COUNT}" LOG_FILE="$LOG_FILE" uv run bash scripts/train_sft_gpt_oss_20b.sh
  if [ "$RUN_PPO" = "1" ]; then
    N_GPUS="${BIG_N_GPUS:-$GPU_COUNT}" TP_SIZE="${BIG_TP_SIZE:-$GPU_COUNT}" LOG_FILE="$LOG_FILE" uv run bash scripts/train_ppo_sglang_gpt_oss_20b.sh
  fi
fi

uv run python scripts/plot_training_log.py --log "$LOG_FILE" --out-dir artifacts/all
echo "paperhound run done"
