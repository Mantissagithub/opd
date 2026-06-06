import argparse
import json
import random
from pathlib import Path

from datasets import Dataset, load_dataset


DATASET_ID = "paperbd/paper-cited-chunks-v1"

SYSTEM_PROMPT = (
    "you are a paper-grounded citation model. given an arxiv id and a retrieval "
    "query, return only the paper chunks that directly support the answer. "
    "output valid json with this shape: {\"cited_chunks\": [\"...\"]}."
)


def build_user_prompt(arxiv_id: str, query: str) -> str:
    return f"arxiv_id: {arxiv_id}\nquery: {query}\n\nreturn the cited chunks as json."


def build_target(cited_chunks: list[str]) -> str:
    return json.dumps({"cited_chunks": cited_chunks}, ensure_ascii=True, indent=2)


def make_record(example: dict, idx: int, split: str) -> dict:
    arxiv_id = example["arxiv_id"]
    query = example["query"]
    cited_chunks = list(example["cited_chunks"])
    target = build_target(cited_chunks)

    return {
        "data_source": DATASET_ID,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(arxiv_id, query)},
        ],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(arxiv_id, query)},
            {"role": "assistant", "content": target},
        ],
        "response": target,
        "ability": "citation_grounding",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(cited_chunks, ensure_ascii=True),
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "arxiv_id": arxiv_id,
            "num_cited_chunks": len(cited_chunks),
        },
    }


def split_rows(rows: list[dict], val_size: int, seed: int) -> tuple[list[dict], list[dict]]:
    order = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(order)
    val_ids = set(order[:val_size])
    train_rows = [row for idx, row in enumerate(rows) if idx not in val_ids]
    val_rows = [row for idx, row in enumerate(rows) if idx in val_ids]
    return train_rows, val_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", default="data/paperhound")
    parser.add_argument("--val-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1647)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    raw = load_dataset(DATASET_ID, split="test")
    rows = [raw[i] for i in range(len(raw))]
    if args.smoke:
        rows = rows[:12]
        args.val_size = min(4, len(rows) // 3)

    if not 0 < args.val_size < len(rows):
        raise ValueError("--val-size must be between 1 and len(dataset)-1")

    train_raw, val_raw = split_rows(rows, args.val_size, args.seed)
    train = [make_record(row, idx, "train") for idx, row in enumerate(train_raw)]
    val = [make_record(row, idx, "val") for idx, row in enumerate(val_raw)]

    out_dir = Path(args.local_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train).to_parquet(str(out_dir / "train.parquet"))
    Dataset.from_list(val).to_parquet(str(out_dir / "val.parquet"))

    meta = {
        "dataset_id": DATASET_ID,
        "source_split": "test",
        "train_rows": len(train),
        "val_rows": len(val),
        "seed": args.seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
