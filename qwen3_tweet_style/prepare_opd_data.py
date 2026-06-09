import argparse
import json
from pathlib import Path

from datasets import Dataset, load_dataset

# verl applies the tokenizer chat template to `prompt` (a messages list) at rollout. the
# merged student/teacher tokenizers carry a passthrough template (see merge_adapter.py) that
# renders a single user message as the exact SFT format:
#   "### Instruction:\n{content}\n### Response:\n"
# so we just emit the raw instruction as one user message here.
DATA_SOURCE = "tweet-style-opd"


def make_record(instruction, idx, split):
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": instruction}],
        "ability": "tweet_style",
        # distillation needs no ground-truth reward; keep a stub so verl's schema is happy.
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {"split": split, "index": idx, "instruction": instruction},
    }


def load_instructions(args):
    if args.jsonl:
        return [json.loads(line)["instruction"] for line in Path(args.jsonl).read_text().splitlines() if line.strip()]
    ds = load_dataset(args.hf_dataset, split="train")
    return [r for r in ds["instruction"] if r and r.strip()]


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--jsonl", help="local opd_prompts.jsonl from gen_opd_prompts.py")
    src.add_argument("--hf-dataset", help="hub dataset id with an `instruction` column")
    p.add_argument("--local-dir", default="data/opd")
    p.add_argument("--val-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=1647)
    args = p.parse_args()

    instructions = load_instructions(args)
    import random

    random.Random(args.seed).shuffle(instructions)
    if not 0 < args.val_size < len(instructions):
        raise ValueError("--val-size must be between 1 and len(instructions)-1")

    val = instructions[: args.val_size]
    train = instructions[args.val_size :]

    out = Path(args.local_dir)
    out.mkdir(parents=True, exist_ok=True)
    Dataset.from_list([make_record(t, i, "train") for i, t in enumerate(train)]).to_parquet(str(out / "train.parquet"))
    Dataset.from_list([make_record(t, i, "val") for i, t in enumerate(val)]).to_parquet(str(out / "val.parquet"))
    print(f"wrote {len(train)} train + {len(val)} val records to {out}")


if __name__ == "__main__":
    main()
