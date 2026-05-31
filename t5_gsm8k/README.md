# t5 gkd on gsm8k

[![huggingface](https://img.shields.io/badge/🤗%20model-Pradheep1647%2Fflan--t5--small--gsm8k--77m--gkd--jsd--lr3e4--b16--u4k--lam05--beta05-yellow)](https://huggingface.co/Pradheep1647/flan-t5-small-gsm8k-77m-gkd-jsd-lr3e4-b16-u4k-lam05-beta05)

implementing generalised knowledge distillation (gkd) from [this paper](https://arxiv.org/abs/2306.13649) on the gsm8k math reasoning dataset.

the idea is to distill a large teacher (flan-t5-large or a remote 20b model) into a smaller student (flan-t5-small) using a mix of supervised kl and on-policy kl loss. the student samples its own generations, asks the teacher to score them, and trains on that signal — which is the on-policy part that makes gkd different from standard kd.

## setup

```bash
uv sync
```

## scripts

- `train.py` — main training loop. hits a remote teacher api for logprobs, runs gkd loss
- `train_laptop_t5_gkd.py` — local gkd trainer for flan-t5-small with a local hf teacher
- `teacher_inference.py` — fastapi server for the teacher model, meant to run on a separate gpu
- `eval_compare.py` — compares flan-t5 small/base/large and the trained student on gsm8k
- `plot_training_metrics.py` — renders a bar chart for eval results and a training-metrics panel from tensorboard events

## results

ran eval on the full gsm8k test set after training the small model for 4k update steps.

```
model            accuracy
---------------  --------
flan-t5-small    0.0227
flan-t5-base     0.0318
flan-t5-large    0.0538
trained student  0.0243
```

![gsm8k accuracy bar chart](artifacts/gsm8k_accuracy_bar.png)

the trained student (flan-t5-small after gkd) does slightly better than the untrained small baseline (0.0243 vs 0.0227), but it is still far behind flan-t5-large. that is consistent with the task difficulty and the size gap between the student and teacher.

## training curves

the main run logged these scalar series to tensorboard:

- `train/loss_update`
- `train/lr`
- `train/on_policy_fraction`
- `train/grad_norm`
- `eval/accuracy`

![training metrics](artifacts/training_metrics.png)

## reproduce the plots

```bash
uv run python plot_training_metrics.py \
  --accuracy 'FLAN T5 Small=0.0227' \
  --accuracy 'FLAN T5 Base=0.0318' \
  --accuracy 'FLAN T5 Large=0.0538' \
  --accuracy 'Trained Student=0.0243'
```
