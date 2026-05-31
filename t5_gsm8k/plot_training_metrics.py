import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = SCRIPT_DIR / "runs" / "t5_small_gsm8k_gkd_true_laptop"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts"

BG = "#fbfaf7"
PANEL = "#ffffff"
TEXT = "#201a17"
MUTED = "#6b625d"
GRID = "#ded8d1"
AXIS = "#8e857f"
COLORS = ["#2f6fed", "#ec6f66", "#2c9f70", "#9a57c5", "#d98a2b"]


def parse_args():
    parser = argparse.ArgumentParser(description="Render T5 GSM8K result plots from TensorBoard events.")
    parser.add_argument("--event-file", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--accuracy",
        action="append",
        default=[],
        help='Entry like "FLAN T5 Small=0.0227". Repeat for each model.',
    )
    return parser.parse_args()


def resolve_event_file(path_str):
    if path_str:
        path = Path(path_str)
        return path if path.is_absolute() else SCRIPT_DIR / path

    event_files = sorted(DEFAULT_RUN_DIR.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {DEFAULT_RUN_DIR}")
    return event_files[-1]


def parse_accuracy_entries(entries):
    if not entries:
        return []

    parsed = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid --accuracy value: {entry}")
        name, value = entry.split("=", 1)
        parsed.append((name.strip(), float(value.strip())))
    return parsed


def load_scalars(event_file):
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    scalars = {}
    for tag in accumulator.Tags()["scalars"]:
        events = accumulator.Scalars(tag)
        scalars[tag] = [(event.step, event.value) for event in events]
    return scalars


def get_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_text(draw, xy, text, font, fill=TEXT, anchor=None):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def format_float(value):
    if value >= 1:
        return f"{value:.2f}"
    if value >= 0.1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def draw_line_chart(draw, rect, series, title, color):
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle(rect, radius=18, fill=PANEL)

    title_font = get_font(22, bold=True)
    body_font = get_font(16)
    draw_text(draw, (x0 + 24, y0 + 20), title, title_font)

    chart_left = x0 + 60
    chart_top = y0 + 62
    chart_right = x1 - 28
    chart_bottom = y1 - 54

    draw.line((chart_left, chart_top, chart_left, chart_bottom), fill=AXIS, width=2)
    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=AXIS, width=2)

    if not series:
        draw_text(draw, (x0 + 24, y0 + 68), "No data", body_font, MUTED)
        return

    steps = [step for step, _ in series]
    values = [value for _, value in series]
    min_y = min(values)
    max_y = max(values)
    if max_y == min_y:
        min_y -= 1
        max_y += 1

    y_ticks = 4
    for i in range(y_ticks + 1):
        ratio = i / y_ticks
        y = chart_bottom - ratio * (chart_bottom - chart_top)
        value = min_y + ratio * (max_y - min_y)
        draw.line((chart_left, y, chart_right, y), fill=GRID, width=1)
        draw_text(draw, (chart_left - 10, y), format_float(value), body_font, MUTED, anchor="ra")

    for i, ratio in enumerate([0.0, 0.5, 1.0]):
        step_value = steps[0] + ratio * (steps[-1] - steps[0])
        x = chart_left + ratio * (chart_right - chart_left)
        draw.line((x, chart_top, x, chart_bottom), fill=GRID, width=1)
        draw_text(draw, (x, chart_bottom + 12), str(int(step_value)), body_font, MUTED, anchor="ma")

    def map_x(step):
        if steps[-1] == steps[0]:
            return chart_left
        return chart_left + (step - steps[0]) / (steps[-1] - steps[0]) * (chart_right - chart_left)

    def map_y(value):
        return chart_bottom - (value - min_y) / (max_y - min_y) * (chart_bottom - chart_top)

    points = [(map_x(step), map_y(value)) for step, value in series]
    draw.line(points, fill=color, width=4, joint="curve")

    last_x, last_y = points[-1]
    draw.ellipse((last_x - 5, last_y - 5, last_x + 5, last_y + 5), fill=color)
    draw_text(draw, (chart_right, chart_top - 8), f"last {format_float(values[-1])}", body_font, MUTED, anchor="ra")


def draw_summary_panel(draw, rect, event_file, scalars):
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle(rect, radius=18, fill=PANEL)
    title_font = get_font(22, bold=True)
    body_font = get_font(17)
    mono_font = get_font(16)
    draw_text(draw, (x0 + 24, y0 + 20), "Run Summary", title_font)

    eval_acc = scalars.get("eval/accuracy", [])
    loss = scalars.get("train/loss_update", [])
    grad_norm = scalars.get("train/grad_norm", [])
    lines = [
        f"event file: {event_file.name}",
        f"updates: {loss[-1][0] if loss else 'n/a'}",
        f"last loss: {format_float(loss[-1][1]) if loss else 'n/a'}",
        f"last eval acc: {format_float(eval_acc[-1][1]) if eval_acc else 'n/a'}",
        f"last grad norm: {format_float(grad_norm[-1][1]) if grad_norm else 'n/a'}",
    ]
    y = y0 + 62
    for line in lines:
        draw_text(draw, (x0 + 24, y), line, mono_font if ":" in line else body_font, TEXT)
        y += 34


def create_training_metrics_plot(scalars, event_file, output_path):
    image = Image.new("RGB", (1600, 1080), BG)
    draw = ImageDraw.Draw(image)
    title_font = get_font(34, bold=True)
    subtitle_font = get_font(18)
    draw_text(draw, (60, 42), "T5 GSM8K Training Metrics", title_font)
    draw_text(draw, (60, 86), "Generalized Knowledge Distillation run from TensorBoard scalars", subtitle_font, MUTED)

    margin_x = 52
    top = 132
    gap = 24
    panel_w = (1600 - 2 * margin_x - 2 * gap) // 3
    panel_h = 412

    charts = [
        ("train/loss_update", "Loss Per Update", COLORS[0]),
        ("eval/accuracy", "Eval Accuracy", COLORS[1]),
        ("train/lr", "Learning Rate", COLORS[2]),
        ("train/on_policy_fraction", "On-Policy Fraction", COLORS[3]),
        ("train/grad_norm", "Gradient Norm", COLORS[4]),
    ]

    for idx, (tag, title, color) in enumerate(charts):
        row = idx // 3
        col = idx % 3
        rect = (
            margin_x + col * (panel_w + gap),
            top + row * (panel_h + gap),
            margin_x + col * (panel_w + gap) + panel_w,
            top + row * (panel_h + gap) + panel_h,
        )
        draw_line_chart(draw, rect, scalars.get(tag, []), title, color)

    summary_rect = (
        margin_x + 2 * (panel_w + gap),
        top + panel_h + gap,
        margin_x + 2 * (panel_w + gap) + panel_w,
        top + 2 * panel_h + gap,
    )
    draw_summary_panel(draw, summary_rect, event_file, scalars)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def create_accuracy_bar_chart(results, output_path):
    image = Image.new("RGB", (1400, 820), BG)
    draw = ImageDraw.Draw(image)
    title_font = get_font(34, bold=True)
    subtitle_font = get_font(18)
    label_font = get_font(20, bold=True)
    body_font = get_font(16)

    draw_text(draw, (60, 42), "GSM8K Accuracy Comparison", title_font)
    draw_text(draw, (60, 86), "Full test split exact final-answer accuracy", subtitle_font, MUTED)

    x0, y0, x1, y1 = 90, 160, 1320, 720
    draw.rounded_rectangle((x0 - 20, y0 - 20, x1 + 20, y1 + 20), radius=18, fill=PANEL)
    draw.line((x0, y0, x0, y1), fill=AXIS, width=2)
    draw.line((x0, y1, x1, y1), fill=AXIS, width=2)

    max_value = max(value for _, value in results)
    chart_max = max(max_value * 1.15, 0.06)
    for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y1 - ratio * (y1 - y0)
        value = ratio * chart_max
        draw.line((x0, y, x1, y), fill=GRID, width=1)
        draw_text(draw, (x0 - 10, y), f"{value:.3f}", body_font, MUTED, anchor="ra")

    count = len(results)
    slot = (x1 - x0) / count
    bar_w = slot * 0.56
    for idx, (name, value) in enumerate(results):
        center = x0 + slot * idx + slot / 2
        bar_h = (value / chart_max) * (y1 - y0)
        left = center - bar_w / 2
        top = y1 - bar_h
        right = center + bar_w / 2
        draw.rounded_rectangle((left, top, right, y1), radius=14, fill=COLORS[idx % len(COLORS)])
        draw_text(draw, (center, top - 16), f"{value:.4f}", label_font, TEXT, anchor="ms")
        draw.multiline_text(
            (center, y1 + 18),
            name.replace(" ", "\n"),
            font=body_font,
            fill=TEXT,
            anchor="ma",
            align="center",
            spacing=2,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main():
    args = parse_args()
    event_file = resolve_event_file(args.event_file)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR / output_dir

    scalars = load_scalars(event_file)
    accuracy_results = parse_accuracy_entries(args.accuracy)

    if accuracy_results:
        create_accuracy_bar_chart(accuracy_results, output_dir / "gsm8k_accuracy_bar.png")
    create_training_metrics_plot(scalars, event_file, output_dir / "training_metrics.png")

    print(f"wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
