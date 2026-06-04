import argparse
import os
import re
from pathlib import Path

import torch
from datasets import load_dataset
from openai import OpenAI
from peft import PeftModel
from rouge_score import rouge_scorer
from sacrebleu import sentence_bleu
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import sft

SCRIPT_DIR = Path(__file__).resolve().parent

MODELS = [
    {"name": "Qwen3-4B tweet-style", "base": "Qwen/Qwen3-4B-Base", "repo": "qwen3-tweet-style-4b"},
    {"name": "Qwen3-32B tweet-style", "base": "Qwen/Qwen3-32B-Base", "repo": "qwen3-tweet-style-32b"},
]

JUDGE_SYSTEM = (
    "You are scoring how well a generated tweet matches a reference tweet in voice, "
    "tone, brevity, and style. Reply with ONLY an integer from 1 (no match) to 10 "
    "(indistinguishable style)."
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-username", default=os.environ.get("HF_USERNAME"))
    p.add_argument("--judge-model", default="openai/gpt-4o")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--no-judge", action="store_true")
    return p.parse_args()

def resolve_adapter(repo, hf_username):
    # prefer a local training dir, otherwise pull the adapter off the hub
    local = SCRIPT_DIR / repo
    if local.exists():
        return str(local)
    if hf_username:
        return f"{hf_username}/{repo}"
    raise FileNotFoundError(f"no local '{repo}' dir and no HF username for hub fallback")

def load_model(base, adapter_ref):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_ref)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter_ref)
    model.eval()
    return model, tokenizer

def generate(model, tokenizer, instruction, max_new_tokens):
    prompt = sft.build_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    # strip the prompt, keep only what the model generated
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def judge_score(client, judge_model, instruction, reference, prediction):
    user = f"Instruction: {instruction}\n\nReference tweet: {reference}\n\nGenerated tweet: {prediction}\n\nScore 1-10:"
    resp = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
        max_tokens=4,
        temperature=0.0,
    )
    match = re.search(r"\d+", resp.choices[0].message.content or "")
    return min(max(int(match.group()), 1), 10) if match else None

def eval_model(spec, test_ds, args, judge_client):
    adapter_ref = resolve_adapter(spec["repo"], args.hf_username)
    model, tokenizer = load_model(spec["base"], adapter_ref)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rouge_sum = bleu_sum = 0.0
    judge_sum = judge_n = 0

    for row in tqdm(test_ds, desc=spec["name"], leave=False):
        pred = generate(model, tokenizer, row["instruction"], args.max_new_tokens)
        ref = row["response"]
        rouge_sum += scorer.score(ref, pred)["rougeL"].fmeasure
        bleu_sum += sentence_bleu(pred, [ref]).score
        if judge_client is not None:
            score = judge_score(judge_client, args.judge_model, row["instruction"], ref, pred)
            if score is not None:
                judge_sum += score
                judge_n += 1

    n = len(test_ds)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"rougeL": rouge_sum / n, "bleu": bleu_sum / n, "judge": (judge_sum / judge_n) if judge_n else None}

def main():
    args = parse_args()
    split = "test" if args.limit is None else f"test[:{args.limit}]"
    test_ds = load_dataset(sft.DATASET_ID, split=split)

    judge_client = None
    if not args.no_judge:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set (pass --no-judge to skip)")
        judge_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

    results = []
    for spec in MODELS:
        try:
            results.append((spec["name"], eval_model(spec, test_ds, args, judge_client)))
        except FileNotFoundError as e:
            print(f"skipping {spec['name']}: {e}")

    name_w = max((len(n) for n, _ in results), default=10)
    print()
    print(f"{'Model':<{name_w}}  ROUGE-L   BLEU    Judge")
    print(f"{'-' * name_w}  -------  ------  ------")
    for name, m in results:
        judge = f"{m['judge']:.2f}" if m["judge"] is not None else "  n/a"
        print(f"{name:<{name_w}}  {m['rougeL']:.4f}  {m['bleu']:6.2f}  {judge:>5}")
    print()

if __name__ == "__main__":
    main()
