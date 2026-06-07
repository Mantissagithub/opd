# paperhound

verl + SGLang scripts for [`paperbd/paper-cited-chunks-v1`](https://huggingface.co/datasets/paperbd/paper-cited-chunks-v1).

the dataset card matters here:

- it has 200 rows and only exposes the `test` split.
- columns are `arxiv_id`, `query`, and `cited_chunks`.
- the paper text haystack is not included. the rows are positive cited chunks only, so this setup trains a model to emit the cited chunks for a paper/query pair. it is not full retrieval training unless you add downloaded arxiv paper chunks as negatives.

## models

| slot | model |
|---|---|
| small ~130M | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| big | `openai/gpt-oss-20b` |

## setup

```bash
cd paperhound
uv sync
```

`scripts/hf_env.sh` loads `HF_USERNAME` and `HF_TOKEN` from `.env` before training starts. the scripts default to pushing after training; set `PUSH_TO_HF=0` only if you want a local-only run.

repo names are generated as:

```text
<base_model>-<dataset>-<training_hyperparams>
```

examples:

- `smollm2-135m-instruct-paper-cited-chunks-v1-sft-lr2e-5-ep8-lora32a64-seq2048-mbs8`
- `gpt-oss-20b-paper-cited-chunks-v1-ppo-sglang-lr5e-7-ep8-kl0-001-rolloutn4-tp4-mbs1`

set `HF_PRIVATE=1` if the target model repos should be private.

## prepare data

```bash
uv run python scripts/prepare_verl_data.py --local-dir data/paperhound
```

this creates `train.parquet` and `val.parquet` in verl's expected shape:

- `prompt`: chat-template prompt for PPO/rollouts.
- `messages`: full user/assistant conversation for SFT.
- `reward_model.ground_truth`: JSON list of gold cited chunks.
- `extra_info`: arxiv id, split, row index, chunk count.

## SFT

each script trains, writes to `checkpoints/...`, creates a model card, then uploads that folder to Hugging Face.

small model:

```bash
NPROC_PER_NODE=1 uv run bash scripts/train_sft_130m.sh
```

gpt-oss-20b:

```bash
NPROC_PER_NODE=8 uv run bash scripts/train_sft_gpt_oss_20b.sh
```

## PPO with SGLang rollouts

the PPO scripts start from the SFT checkpoint by default, train with verl + SGLang rollouts, then push the final checkpoint folder to Hugging Face.

small model:

```bash
N_GPUS=1 uv run bash scripts/train_ppo_sglang_130m.sh
```

gpt-oss-20b:

```bash
N_GPUS=8 TP_SIZE=4 uv run bash scripts/train_ppo_sglang_gpt_oss_20b.sh
```

the PPO reward is rule-based: parse the model's JSON, compare `cited_chunks` to the gold chunks, then score recall/F1 plus a small JSON-format bonus.

## one-shot rented-gpu run

```bash
uv run bash scripts/train_all.sh
```

this runs data prep, both SFT jobs, both PPO/SGLang jobs, appends everything to `training.log`, writes plots under `artifacts/`, and pushes each checkpoint to Hugging Face.

for a 2x L40S pod:

```bash
BIG_NPROC_PER_NODE=2 BIG_N_GPUS=2 BIG_TP_SIZE=2 uv run bash scripts/train_all.sh
```

for a single 80GB pod:

```bash
BIG_NPROC_PER_NODE=1 BIG_N_GPUS=1 BIG_TP_SIZE=1 uv run bash scripts/train_all.sh
```

## evaluation

eval adapts [avbiswas/finetuning_recipes](https://github.com/avbiswas/finetuning_recipes) to the cited-chunks task: run each checkpoint over the held-out `val.parquet` split, then score the generations with an LLM judge.

```bash
# inference -> generations.jsonl (use --adapter for the gpt-oss base+adapter path)
uv run python scripts/eval_cited_chunks.py -m merged/smollm2-135m \
  --val-file data/paperhound/val.parquet -o eval_out/smollm2-135m_generations.jsonl

# judge -> scores (needs OPENROUTER_API_KEY)
uv run python scripts/llm_judge.py -i eval_out/smollm2-135m_generations.jsonl \
  -o eval_out/smollm2-135m_judged.jsonl
```

`scripts/run_pod_eval.sh` orchestrates the whole thing on a rented gpu (download verl ckpt -> merge -> infer -> judge) for both models. the judge is OpenRouter `deepseek/deepseek-v4-pro`, scoring 1-5 on faithfulness, answer_correctness, relevance, completeness.

results on the 40-row val split (2026-06-07, on-demand A6000):

| checkpoint | overall | faithfulness | answer_corr. | relevance | completeness |
|---|---|---|---|---|---|
| `smollm2-135m` | 1.36 | 1.77 | 1.18 | 1.40 | 1.10 |
| `gpt-oss-20b` | 3.20 | 4.53 | 2.66 | 3.55 | 2.05 |

(20B scored over 38/40 — 2 judge replies were unparseable and skipped.) the merged/loadable artifacts are pushed to HF: 🤗[`smollm2-135m-...-merged`](https://huggingface.co/Pradheep1647/smollm2-135m-instruct-paper-cited-chunks-v1-sft-lr2e-5-ep8-lora32a64-seq4096-mbs8-merged) (full model) and 🤗[`gpt-oss-20b-...-adapter`](https://huggingface.co/Pradheep1647/gpt-oss-20b-paper-cited-chunks-v1-sft-lr8e-6-ep4-lora16a32-seq2048-mbs1-adapter) (LoRA adapter only — the verl merge drops the frozen MXFP4 experts, so load it on top of base `openai/gpt-oss-20b`).

## notes

`gpt-oss-20b` needs real multi-GPU memory for training. the script uses LoRA, activation checkpointing, FSDP offload, and SGLang tensor-parallel rollout, but it is still a cloud/H100-class run, not a laptop run.

because the dataset only has positives, RL can overfit fast. for a real retrieval agent, extend `prepare_verl_data.py` to download arxiv papers, chunk them, and include hard negatives in `extra_info`.
