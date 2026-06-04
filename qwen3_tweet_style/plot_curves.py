import argparse
import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent

# trainer prints metrics as python-ish dicts with quoted numbers, one per logging step
DICT_RE = re.compile(r"\{'(?:loss|eval_loss)':.*?\}")

def parse_log(path):
    train, evals = [], []
    text = Path(path).read_text(errors="ignore")
    for m in DICT_RE.findall(text):
        try:
            d = ast.literal_eval(m)
        except (ValueError, SyntaxError):
            continue
        d = {k: float(v) for k, v in d.items() if _is_num(v)}
        if "loss" in d:
            train.append(d)
        elif "eval_loss" in d:
            evals.append(d)
    return train, evals

def _is_num(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False

def main():
    p = argparse.ArgumentParser()
    # --log LABEL=path, repeatable
    p.add_argument("--log", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--out", default=str(SCRIPT_DIR / "artifacts" / "training_curves.png"))
    args = p.parse_args()

    runs = []
    for spec in args.log:
        label, path = spec.split("=", 1)
        runs.append((label, *parse_log(path)))

    panels = [
        ("loss", "loss", "train loss"),
        ("learning_rate", "lr", "learning rate"),
        ("grad_norm", "grad norm", "grad norm"),
        ("mean_token_accuracy", "token acc", "mean token accuracy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, ylabel, title) in zip(axes.flat, panels):
        for label, train, evals in runs:
            xs = [d["epoch"] for d in train if key in d]
            ys = [d[key] for d in train if key in d]
            line, = ax.plot(xs, ys, marker=".", label=label)
            # overlay eval loss on the loss panel
            if key == "loss" and evals:
                ax.plot([d["epoch"] for d in evals], [d["eval_loss"] for d in evals],
                        marker="o", linestyle="--", color=line.get_color(), label=f"{label} (eval)")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("qwen3 tweet-style — training curves", fontsize=13)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")

if __name__ == "__main__":
    main()
