import argparse
import json

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config


# held-out preference eval for the dpo model: for each (prompt, chosen, rejected) pair,
# score sequence logprob of chosen vs rejected. base model = adapter off, dpo = adapter on.
# reports accuracy (% pairs where chosen is preferred) and mean logp margin.


def seq_logp(model, tok, prompt_msgs, completion_msgs, device, max_length):
    prompt_ids = tok.apply_chat_template(prompt_msgs, add_generation_prompt=True, tokenize=True)
    full_ids = tok.apply_chat_template(list(prompt_msgs) + list(completion_msgs), tokenize=True)
    full_ids = full_ids[:max_length]
    ids = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        logits = model(ids).logits.float()
    logprobs = torch.log_softmax(logits[0, :-1], dim=-1)
    targets = ids[0, 1:]
    token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    start = min(len(prompt_ids), len(full_ids)) - 1  # score only the completion tokens
    return token_lp[start:].sum().item()


def evaluate(model, tok, rows, device, max_length, adapter_on, label):
    if hasattr(model, "set_adapter"):
        model.enable_adapter_layers() if adapter_on else model.disable_adapter_layers()
    wins, margin = 0, 0.0
    for r in rows:
        lc = seq_logp(model, tok, r["prompt"], r["chosen"], device, max_length)
        lr = seq_logp(model, tok, r["prompt"], r["rejected"], device, max_length)
        wins += int(lc > lr)
        margin += lc - lr
    n = len(rows)
    out = {"phase": label, "n": n, "accuracy": wins / n, "mean_margin": margin / n}
    print("[eval] " + json.dumps(out))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="openai/gpt-oss-20b")
    p.add_argument("--adapter", required=True)
    p.add_argument("--data-dir", default="data/paperhound_dpo")
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--output-file", default="eval_out/dpo_preference.json")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, attn_implementation="eager", dtype=torch.bfloat16,
        use_cache=False, quantization_config=Mxfp4Config(dequantize=True), device_map="cuda",
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    device = next(model.parameters()).device

    ds = load_dataset("parquet", data_files=f"{args.data_dir}/test.parquet", split="train")
    rows = [ds[i] for i in range(min(args.num_samples, len(ds)))]

    before = evaluate(model, tok, rows, device, args.max_length, adapter_on=False, label="model")
    after = evaluate(model, tok, rows, device, args.max_length, adapter_on=True, label="model+dpo")

    import os
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump({"model": before, "model+dpo": after}, f, indent=2)
    print("[eval_summary] " + json.dumps({"model": before, "model+dpo": after}))


if __name__ == "__main__":
    main()
