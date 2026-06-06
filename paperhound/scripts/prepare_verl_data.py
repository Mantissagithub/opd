import argparse
import html
import json
import random
import re
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from datasets import Dataset, load_dataset


DATASET_ID = "paperbd/paper-cited-chunks-v1"
AR5IV_URL = "https://ar5iv.labs.arxiv.org/html/{arxiv_id}"

SYSTEM_PROMPT = (
    "you are a paper-grounded citation model. you are given an arxiv id, a retrieval "
    "query, and the paper split into numbered chunks. return only the chunks that "
    "directly support the answer, copied verbatim. output valid json with this shape: "
    "{\"cited_chunks\": [\"...\"]}."
)


class ParagraphExtractor(HTMLParser):
    # pull text out of ar5iv <p> blocks, skipping math/script/style/table subtrees
    SKIP = {"math", "script", "style", "table"}

    def __init__(self) -> None:
        super().__init__()
        self.paras: list[str] = []
        self._buf: list[str] = []
        self._in_p = 0
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "p":
            self._in_p += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == "p" and self._in_p:
            self._in_p -= 1
            if self._in_p == 0:
                text = " ".join("".join(self._buf).split())
                if text:
                    self.paras.append(html.unescape(text))
                self._buf = []

    def handle_data(self, data):
        if self._in_p and not self._skip:
            self._buf.append(data)


def fetch_paper_paragraphs(arxiv_id: str, cache_dir: Path, timeout: int = 30) -> list[str]:
    cache = cache_dir / f"{slug_id(arxiv_id)}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    url = AR5IV_URL.format(arxiv_id=arxiv_id)
    req = urllib.request.Request(url, headers={"User-Agent": "paperhound/0.1"})
    paras: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "ignore")
        parser = ParagraphExtractor()
        parser.feed(raw)
        paras = [p for p in parser.paras if len(p) > 40]
    except Exception as exc:  # network / missing ar5iv render -> negative cache
        print(f"  fetch failed {arxiv_id}: {exc}")
        paras = []

    cache.write_text(json.dumps(paras))
    time.sleep(0.4)
    return paras


def slug_id(arxiv_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", arxiv_id)


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def build_haystack(
    cited_chunks: list[str],
    paragraphs: list[str],
    rng: random.Random,
    max_distractors: int,
    char_budget: int,
) -> tuple[list[str], int]:
    needle_norm = [normalize(c) for c in cited_chunks]

    distractors = []
    for para in paragraphs:
        n = normalize(para)
        if any(nn in n or n in nn for nn in needle_norm):
            continue
        distractors.append(para)
    rng.shuffle(distractors)

    chosen: list[str] = []
    total = sum(len(c) for c in cited_chunks)
    for d in distractors:
        if len(chosen) >= max_distractors:
            break
        if total + len(d) > char_budget:
            continue
        chosen.append(d)
        total += len(d)

    haystack = list(cited_chunks) + chosen
    rng.shuffle(haystack)
    return haystack, len(chosen)


def build_user_prompt(arxiv_id: str, query: str, haystack: list[str]) -> str:
    lines = [f"arxiv_id: {arxiv_id}", f"query: {query}", "", "paper chunks:"]
    for i, chunk in enumerate(haystack):
        lines.append(f"[{i}] {chunk}")
    lines.append("")
    lines.append("return the cited chunks as json.")
    return "\n".join(lines)


def build_target(cited_chunks: list[str]) -> str:
    return json.dumps({"cited_chunks": cited_chunks}, ensure_ascii=True, indent=2)


def make_record(example: dict, idx: int, split: str, args: argparse.Namespace, rng: random.Random, cache_dir: Path) -> dict:
    arxiv_id = example["arxiv_id"]
    query = example["query"]
    cited_chunks = list(example["cited_chunks"])

    paragraphs = fetch_paper_paragraphs(arxiv_id, cache_dir)
    haystack, num_distractors = build_haystack(
        cited_chunks, paragraphs, rng, args.max_distractors, args.haystack_char_budget
    )

    user_prompt = build_user_prompt(arxiv_id, query, haystack)
    target = build_target(cited_chunks)

    return {
        "data_source": DATASET_ID,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
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
            "num_distractors": num_distractors,
            "num_haystack": len(haystack),
            "paper_fetched": bool(paragraphs),
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
    parser.add_argument("--max-distractors", type=int, default=40)
    parser.add_argument("--haystack-char-budget", type=int, default=8000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    raw = load_dataset(DATASET_ID, split="test")
    rows = [raw[i] for i in range(len(raw))]
    if args.smoke:
        rows = rows[:12]
        args.val_size = min(4, len(rows) // 3)

    if not 0 < args.val_size < len(rows):
        raise ValueError("--val-size must be between 1 and len(dataset)-1")

    out_dir = Path(args.local_dir)
    cache_dir = out_dir / "papers_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_raw, val_raw = split_rows(rows, args.val_size, args.seed)
    rng = random.Random(args.seed)
    print(f"building {len(train_raw)} train + {len(val_raw)} val records (fetching papers from ar5iv)...")
    train = [make_record(row, idx, "train", args, rng, cache_dir) for idx, row in enumerate(train_raw)]
    val = [make_record(row, idx, "val", args, rng, cache_dir) for idx, row in enumerate(val_raw)]

    Dataset.from_list(train).to_parquet(str(out_dir / "train.parquet"))
    Dataset.from_list(val).to_parquet(str(out_dir / "val.parquet"))

    fetched = sum(1 for r in train + val if r["extra_info"]["paper_fetched"])
    meta = {
        "dataset_id": DATASET_ID,
        "source_split": "test",
        "train_rows": len(train),
        "val_rows": len(val),
        "seed": args.seed,
        "max_distractors": args.max_distractors,
        "haystack_char_budget": args.haystack_char_budget,
        "papers_fetched": fetched,
        "papers_total": len(train) + len(val),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
