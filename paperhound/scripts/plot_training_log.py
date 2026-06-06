import argparse
import ast
import csv
import re
from pathlib import Path


METRIC_KEYS = (
    "loss",
    "train_loss",
    "eval_loss",
    "reward",
    "mean_reward",
    "kl",
    "entropy",
    "learning_rate",
    "lr",
)


def coerce_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_dict_line(line):
    start = line.find("{")
    end = line.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        parsed = ast.literal_eval(line[start : end + 1])
    except (SyntaxError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_key_values(line):
    values = {}
    for key, raw in re.findall(r"([a-zA-Z_][a-zA-Z0-9_./-]*)=([-+]?\d*\.?\d+(?:e[-+]?\d+)?)", line):
        name = key.split("/")[-1].replace("-", "_")
        if name in METRIC_KEYS or any(name.endswith(metric) for metric in METRIC_KEYS):
            values[name] = float(raw)
    return values


def parse_log(path):
    rows = []
    step = 0
    for line in path.read_text(errors="ignore").splitlines():
        metrics = {}
        parsed = parse_dict_line(line)
        for key, value in parsed.items():
            name = str(key).split("/")[-1].replace("-", "_")
            number = coerce_float(value)
            if number is not None and (name in METRIC_KEYS or any(name.endswith(metric) for metric in METRIC_KEYS)):
                metrics[name] = number
        metrics.update(parse_key_values(line))
        if not metrics:
            continue
        step += 1
        metrics.setdefault("step", step)
        rows.append(metrics)
    return rows


def write_csv(rows, path):
    keys = ["step"] + sorted({key for row in rows for key in row if key != "step"})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, out_path):
    import matplotlib.pyplot as plt

    metrics = [key for key in sorted({key for row in rows for key in row}) if key != "step"]
    if not metrics:
        return False

    cols = 2
    rows_n = (len(metrics) + 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(12, max(4, rows_n * 3)), squeeze=False)
    axes_flat = [axis for pair in axes for axis in pair]

    for axis, metric in zip(axes_flat, metrics):
        xs = [row["step"] for row in rows if metric in row]
        ys = [row[metric] for row in rows if metric in row]
        axis.plot(xs, ys, linewidth=1.8)
        axis.set_title(metric)
        axis.set_xlabel("logged metric index")
        axis.grid(alpha=0.25)

    for axis in axes_flat[len(metrics) :]:
        axis.axis("off")

    fig.suptitle("paperhound training metrics", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def plot_final_bars(rows, out_path):
    import matplotlib.pyplot as plt

    final = {}
    for row in rows:
        for key, value in row.items():
            if key != "step":
                final[key] = value
    final = {key: value for key, value in final.items() if key in {"loss", "train_loss", "eval_loss", "reward", "mean_reward"}}
    if not final:
        return False

    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(final.keys(), final.values(), color=["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c"][: len(final)])
    axis.set_title("final logged metrics")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="training.log")
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = parse_log(log_path)
    if not rows:
        print(f"no plottable metrics found in {log_path}")
        return

    write_csv(rows, out_dir / "training_metrics.csv")
    made_curve = plot(rows, out_dir / "training_metrics.png")
    made_bar = plot_final_bars(rows, out_dir / "final_metrics_bar.png")
    print(f"wrote {out_dir / 'training_metrics.csv'}")
    if made_curve:
        print(f"wrote {out_dir / 'training_metrics.png'}")
    if made_bar:
        print(f"wrote {out_dir / 'final_metrics_bar.png'}")


if __name__ == "__main__":
    main()
