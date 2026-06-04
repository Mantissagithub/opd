# qwen3_tweet_style

teaching two Qwen3 base models to draft X/Twitter posts in my own voice, by
fine-tuning them on [`Pradheep1647/tweet-style-dataset`](https://huggingface.co/datasets/Pradheep1647/tweet-style-dataset)
(967 train / 85 val / 86 test, `instruction` → `response`).

| model | base | hub repo (`<model>-<ds>-<parameters>`) |
|---|---|---|
| 4B  | `Qwen/Qwen3-4B-Base`  | `<user>/qwen3-tweet-style-4b` |
| 32B | `Qwen/Qwen3-32B-Base` | `<user>/qwen3-tweet-style-32b` |

## what's going on here

The base models are pretrained-only (no instruct tuning), so we run plain
supervised fine-tuning to align them to the `instruction → response` format and,
more importantly, to my writing style.

- **QLoRA** — instead of touching the full weights we freeze a 4-bit (nf4)
  quantized base and train small low-rank adapters on top. This is what makes the
  32B trainable on a single rented GPU. See the LoRA paper
  ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)) and QLoRA
  ([Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)).
- **completion-only loss** — we feed the data to `trl`'s `SFTTrainer` as
  prompt/completion pairs so the loss only lands on the tweet, not on the
  instruction text we wrote. ([trl SFT docs](https://huggingface.co/docs/trl/sft_trainer))
- the dataset itself was built with on-policy distillation from a larger teacher,
  which is the broader theme of this repo
  ([GKD, Agarwal et al., 2023](https://arxiv.org/abs/2306.13649)) — here we're just
  doing the student SFT step on the collected data.

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
a menu to train the 4B or 32B (auto-pushes the adapter to the hub) or run the
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

scores both models on the held-out `test` split three ways:

- **ROUGE-L** and **BLEU** against the reference tweet — cheap n-gram overlap, a
  rough proxy since style isn't really an exact-match problem.
- **LLM-as-judge** — an OpenRouter model (`openai/gpt-4o` by default,
  `--judge-model` to swap) rates 1–10 how well the generated tweet matches the
  reference in voice/tone. this is the metric that actually captures "does it sound
  like me". ([judging LLMs with LLMs, Zheng et al., 2023](https://arxiv.org/abs/2306.05685))

`--limit N` for a quick pass, `--no-judge` to skip the API calls.

## results

| model | ROUGE-L | BLEU | judge (1–10) |
|---|---|---|---|
| Qwen3-4B tweet-style  | _tbd_ | _tbd_ | _tbd_ |
| Qwen3-32B tweet-style | _tbd_ | _tbd_ | _tbd_ |
