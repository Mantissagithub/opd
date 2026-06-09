import argparse
import json
import os
import re
from pathlib import Path

from datasets import Dataset, load_dataset
from openai import OpenAI
from tqdm.auto import tqdm

# prompts-only synthetic set for on-policy distillation. the student self-generates
# the tweets during rollout, so we only need a big, diverse pool of instructions.
# seeded from the real dataset's instructions so the new ones stay on-topic for my voice.
# (kept import-light on purpose: only openai + datasets, no torch, so it runs fast locally.)
SEED_DATASET = "Pradheep1647/tweet-style-dataset"
OUT_REPO = "tweet-style-opd-prompts"

GEN_SYSTEM = (
    "You write instructions for a tweet-drafting model. Each instruction asks the model "
    "to write a single short X/Twitter post about some topic, idea, reaction, or moment. "
    "Vary the topic, intent (hot take, shower thought, announcement, joke, question, "
    "observation, rant, hype, etc.), and phrasing widely. Keep each instruction one line, "
    "self-contained, and free of the tweet text itself. Return ONLY a JSON array of strings."
)


def build_gen_prompt(seeds, n):
    examples = "\n".join(f"- {s}" for s in seeds)
    return (
        f"Here are example instructions in the style/voice we target:\n{examples}\n\n"
        f"Write {n} NEW, distinct instructions in the same spirit but covering different "
        f"topics and intents. JSON array of {n} strings only."
    )


def norm(text):
    return re.sub(r"\s+", " ", text.lower()).strip().strip(".")


def parse_array(text):
    # model usually returns a clean json array; fall back to bracket slice then to lines.
    text = text.strip()
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group(0))
            return [str(x).strip() for x in arr if str(x).strip()]
        except json.JSONDecodeError:
            pass
    return [re.sub(r"^[-*\d.)\s]+", "", ln).strip() for ln in text.splitlines() if len(ln.strip()) > 10]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deepseek/deepseek-v4-flash")
    p.add_argument("--target", type=int, default=1500, help="number of unique prompts to produce")
    p.add_argument("--per-call", type=int, default=25)
    p.add_argument("--seeds-per-call", type=int, default=8)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--out", default="data/opd_prompts.jsonl")
    p.add_argument("--push", action="store_true")
    p.add_argument("--hf-username", default=os.environ.get("HF_USERNAME"))
    args = p.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

    seed_ds = load_dataset(SEED_DATASET, split="train")
    seed_instructions = [r for r in seed_ds["instruction"] if r and r.strip()]

    import concurrent.futures as cf
    import random

    seen = {norm(s) for s in seed_instructions}  # don't regenerate the seeds themselves
    prompts = []

    def one_call(_):
        seeds = random.sample(seed_instructions, min(args.seeds_per_call, len(seed_instructions)))
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": GEN_SYSTEM},
                    {"role": "user", "content": build_gen_prompt(seeds, args.per_call)},
                ],
                temperature=args.temperature,
            )
            return parse_array(resp.choices[0].message.content or "")
        except Exception as e:
            print(f"  call failed ({type(e).__name__}: {e})", flush=True)
            return []

    print(f"generating ~{args.target} prompts with {args.model} (concurrency={args.concurrency}) ...", flush=True)
    stale = 0
    calls = 0
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        pbar = tqdm(total=args.target, desc="prompts", unit="prompt", dynamic_ncols=True)
        while len(prompts) < args.target and stale < 8:
            before = len(prompts)
            for cands in ex.map(one_call, range(args.concurrency)):
                calls += 1
                for cand in cands:
                    n = norm(cand)
                    if n and n not in seen and 10 < len(cand) < 400:
                        seen.add(n)
                        prompts.append(cand)
            added = len(prompts) - before
            stale = stale + 1 if added == 0 else 0
            pbar.update(min(added, args.target - pbar.n))
            pbar.set_postfix(calls=calls, dupes=f"{stale}/8", kept=len(prompts))
        pbar.close()

    prompts = prompts[: args.target]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for instr in prompts:
            f.write(json.dumps({"instruction": instr}) + "\n")
    print(f"wrote {len(prompts)} prompts to {out_path}")

    if args.push:
        if not args.hf_username:
            raise RuntimeError("--push needs HF_USERNAME")
        repo_id = f"{args.hf_username}/{OUT_REPO}"
        Dataset.from_list([{"instruction": i} for i in prompts]).push_to_hub(
            repo_id, token=os.environ.get("HF_TOKEN")
        )
        print(f"pushed dataset to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
