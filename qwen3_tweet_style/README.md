# qwen3_tweet_style

teaching two Qwen3 base models to draft X/Twitter posts in my own voice, by
fine-tuning them on [`Pradheep1647/tweet-style-dataset`](https://huggingface.co/datasets/Pradheep1647/tweet-style-dataset)
(967 train / 85 val / 86 test, `instruction` → `response`).

| model | base | adapter |
|---|---|---|
| 4B  | `Qwen/Qwen3-4B-Base`       | [![hf](https://img.shields.io/badge/🤗-qwen3--tweet--style--4b-yellow)](https://huggingface.co/Pradheep1647/qwen3-tweet-style-4b) |
| 30B-A3B | `Qwen/Qwen3-30B-A3B-Base`  | [![hf](https://img.shields.io/badge/🤗-qwen3--tweet--style--30b--a3b-yellow)](https://huggingface.co/Pradheep1647/qwen3-tweet-style-30b-a3b) |

> there is no `Qwen3-32B-Base` on the hub (qwen3's dense base models stop at 14B),
> so the big slot uses `Qwen3-30B-A3B-Base` — a 30B moe base, ~3B active per token.

## the point

this is the **SFT warm-up**, not the main event — it just teaches the base models
the `instruction → response` format and my voice, giving on-policy distillation a
sane starting policy to improve from. the OPD work builds on these adapters.

the only SFT details that bit:

- **4B → QLoRA** (4-bit nf4 base + bf16 lora).
- **30B-A3B → bf16 LoRA on attention only** — bitsandbytes can't quantize the moe's
  fused experts (loads ~61GB bf16 regardless), so qlora buys nothing; adapting just
  the attention keeps it under one 80GB gpu.

trained on a rented A100 80GB (prime intellect): 4B ~7 min, 30B-A3B ~46 min.

## setup

```bash
cd qwen3_tweet_style
uv sync
```

## train (TUI)

```bash
uv run python tui.py
```

asks once for **HF username**, **HF token**, and **OpenRouter API key**, then gives
a menu to train the 4B or 30B (auto-pushes the adapter to the hub) or run the
benchmark. everything the child process prints is streamed live and appended to
`training.log`.

run a script directly if you'd rather:

```bash
HF_USERNAME=... HF_TOKEN=... uv run python train_qwen3_4b.py --push
```

## benchmark

```bash
OPENROUTER_API_KEY=... HF_USERNAME=... uv run python eval_benchmark.py
```

scores both models on the held-out `test` split three ways, in full bf16 precision:

- **ROUGE-L** and **BLEU** against the reference tweet — cheap n-gram overlap, a
  rough proxy since style isn't really an exact-match problem.
- **LLM-as-judge** — an OpenRouter model (`openai/gpt-4o` by default,
  `--judge-model` to swap) rates 1–10 how well the generated tweet matches the
  reference in voice/tone. this is the metric that actually captures "does it sound
  like me". ([judging LLMs with LLMs, Zheng et al., 2023](https://arxiv.org/abs/2306.05685))

`--limit N` for a quick pass, `--no-judge` to skip the API calls.

## results

greedy decoding on all 86 test rows, judge = `openai/gpt-4o`.

| model | ROUGE-L | BLEU | judge (1–10) |
|---|---|---|---|
| Qwen3-4B tweet-style       | 0.4137 | 18.87 | 7.85 |
| Qwen3-30B-A3B tweet-style  | **0.4304** | **19.21** | **7.92** |

![benchmark](artifacts/benchmark.png)

the 30B-A3B is a bit ahead on all three, but the gap is small — the 4B already
picks up the voice well, and for a short-form style task it's the more practical
model to actually serve.

## training curves

![training vs validation loss](artifacts/loss_curves.png)

the 4B's train loss dives toward ~1.0 while its eval loss bottoms out at epoch 2 and
ticks back up at epoch 3 — mild overfitting on a tiny dataset. the 30B-A3B's eval
loss keeps drifting down (1.89 → 1.82 → 1.82), so it generalises a little better.
one epoch (or some early stopping) would probably be the sweet spot for the 4B.

![training dynamics](artifacts/train_dynamics.png)

the 4B (higher lr, smaller model) pushes token accuracy and grad norm up faster —
the same eagerness that shows up as overfitting above.

curves are parsed straight from the trainer logs:

```bash
uv run python plot_curves.py \
  --log "Qwen3-4B=artifacts/train_4b.log" \
  --log "Qwen3-30B-A3B=artifacts/train_30b_a3b.log"
```
