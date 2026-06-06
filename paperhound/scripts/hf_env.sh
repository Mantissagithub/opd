#!/usr/bin/env bash

load_hf_env() {
  local env_file="${ENV_FILE:-.env}"
  if [ -f "$env_file" ]; then
    set -a
    source "$env_file"
    set +a
  fi
}

start_training_log() {
  LOG_FILE="${LOG_FILE:-training.log}"
  mkdir -p "$(dirname "$LOG_FILE")"
  touch "$LOG_FILE"
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo "logging to $LOG_FILE"
}

require_hf_env() {
  if [ "${PUSH_TO_HF:-1}" != "1" ]; then
    return 0
  fi

  load_hf_env
  if [ -z "${HF_USERNAME:-}" ] || [ -z "${HF_TOKEN:-}" ]; then
    echo "PUSH_TO_HF=1 needs HF_USERNAME and HF_TOKEN in ${ENV_FILE:-.env}" >&2
    exit 1
  fi
}
