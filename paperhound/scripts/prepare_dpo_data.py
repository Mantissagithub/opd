import argparse
import json
from pathlib import Path

from datasets import load_dataset


DATASET_ID = "paperbd/paper_preference_150K-v1"

# the hub dataset already ships the trl conversational-dpo shape:
#   prompt / chosen / rejected, each a list of {role, content}
# so prep is mostly a local snapshot + optional subsample for faster runs.
COLUMNS = ["prompt", "chosen", "rejected"]


def take(split, n, seed):
    if n and n < len(split):
        return split.shuffle(seed=seed).select(range(n))
    return split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", default="data/paperhound_dpo")
    parser.add_argument("--seed", type=int, default=1647)
    # subsample knobs -- offline dpo on the full 120k pairs is slow on a 20b,
    # cap these for a fast pass and raise them when you want the full run.
    parser.add_argument("--max-train", type=int, default=0, help="0 = keep all 120k")
    parser.add_argument("--max-test", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.max_train, args.max_test = 64, 16

    ds = load_dataset(DATASET_ID)
    train = take(ds["train"], args.max_train, args.seed).select_columns(COLUMNS)
    test = take(ds["test"], args.max_test, args.seed).select_columns(COLUMNS)

    out_dir = Path(args.local_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(str(out_dir / "train.parquet"))
    test.to_parquet(str(out_dir / "test.parquet"))

    meta = {
        "dataset_id": DATASET_ID,
        "train_rows": len(train),
        "test_rows": len(test),
        "seed": args.seed,
        "format": "trl-conversational-dpo (prompt/chosen/rejected)",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
