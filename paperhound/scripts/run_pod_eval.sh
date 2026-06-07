#!/usr/bin/env bash
set -euo pipefail

# one-shot cited-chunks eval for both sft checkpoints, meant to run on a rented gpu.
# needs HF_TOKEN, HF_USERNAME, OPENROUTER_API_KEY in env (or paperhound/.env).
# pipeline per model: hf download verl ckpt -> verl.model_merger to hf -> inference -> llm judge.

cd "$(dirname "$0")/.."   # paperhound/

if [ -f .env ]; then set -a; source .env; set +a; fi
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
export HF_HUB_ENABLE_HF_TRANSFER=1

DATA_DIR=data/paperhound
OUT_DIR=eval_out
mkdir -p "$OUT_DIR" merged ckpts

echo "### deps"
uv pip install -q hf_transfer outlines openai pydantic

echo "### build val split (ar5iv haystack)"
if [ ! -f "$DATA_DIR/val.parquet" ]; then
  uv run python scripts/prepare_verl_data.py --local-dir "$DATA_DIR"
fi

run_one () {
  local repo="$1" tag="$2" base="${3:-}"   # base set => load base in native dtype + lora adapter (gpt-oss mxfp4)
  echo "=================================================================="
  echo "### $tag :: $repo"
  echo "=================================================================="
  local ckpt="ckpts/$tag" merged="merged/$tag"

  if [ ! -d "$ckpt/global_step_40" ]; then
    uv run huggingface-cli download "$repo" --local-dir "$ckpt" >/dev/null
  fi

  if [ ! -f "$merged/config.json" ]; then
    echo "### merge lora -> hf ($tag)"
    uv run python -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "$ckpt/global_step_40" \
      --target_dir "$merged"
  fi

  # for gpt-oss the merger drops the frozen mxfp4 experts, so load the base in
  # mxfp4 and apply the saved lora adapter (attention-only) on top.
  local model_args
  if [ -n "$base" ]; then
    model_args=(-m "$base" --adapter "$merged/lora_adapter")
  else
    model_args=(-m "$merged")
  fi

  echo "### inference ($tag)"
  uv run python scripts/eval_cited_chunks.py \
    "${model_args[@]}" \
    --val-file "$DATA_DIR/val.parquet" \
    -o "$OUT_DIR/${tag}_generations.jsonl"

  echo "### judge ($tag)"
  uv run python scripts/llm_judge.py \
    -i "$OUT_DIR/${tag}_generations.jsonl" \
    -o "$OUT_DIR/${tag}_judged.jsonl" \
    | tee "$OUT_DIR/${tag}_judge_summary.txt"
}

# fast model first — validates the whole pipeline before the costly 20b run.
run_one "Pradheep1647/smollm2-135m-instruct-paper-cited-chunks-v1-sft-lr2e-5-ep8-lora32a64-seq4096-mbs8" "smollm2-135m"
run_one "Pradheep1647/gpt-oss-20b-paper-cited-chunks-v1-sft-lr8e-6-ep4-lora16a32-seq2048-mbs1" "gpt-oss-20b" "openai/gpt-oss-20b"

echo "### done. outputs in $OUT_DIR"
ls -la "$OUT_DIR"
