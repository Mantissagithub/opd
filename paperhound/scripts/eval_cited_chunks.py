import argparse
import json
import os

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# adapts avbiswas/finetuning_recipes instruction_tuning/inference.py to the
# paper-cited-chunks task: feed the held-out val prompts, emit a jsonl the repo's
# llm_judge.py can score (id, question, response, ground_truth).

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", "-m", required=True)
parser.add_argument("--adapter", default=None, help="peft lora adapter dir to load on top of model-path")
parser.add_argument("--val-file", default="data/paperhound/val.parquet")
parser.add_argument("--num-samples", "-n", type=int, default=None)
parser.add_argument("--batch-size", "-bs", type=int, default=4)
parser.add_argument("--max-new-tokens", type=int, default=1024)
parser.add_argument("--output-file", "-o", required=True)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)
if args.adapter:
    # gpt-oss base stays mxfp4; lora only touches the bf16 attention projections
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()

df = pd.read_parquet(args.val_file)
if args.num_samples:
    df = df.iloc[: args.num_samples]
rows = df.to_dict("records")

os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)


def to_messages(prompt):
    # parquet stores prompt as a list/ndarray of {role, content}
    return [{"role": m["role"], "content": m["content"]} for m in prompt]


def gold_text(row):
    # reward_model.ground_truth is a json-encoded list of gold chunks
    gt = row["reward_model"]["ground_truth"]
    chunks = json.loads(gt)
    return json.dumps({"cited_chunks": chunks}, ensure_ascii=True)


total = len(rows)
completed = 0
with open(args.output_file, "w") as f:
    for start in range(0, total, args.batch_size):
        batch = rows[start : start + args.batch_size]
        messages = [to_messages(r["prompt"]) for r in batch]
        questions = [m[-1]["content"] for m in messages]
        prompts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = encoded["input_ids"].shape[1]
        for j, r in enumerate(batch):
            generated = tokenizer.decode(
                output_ids[j][input_len:], skip_special_tokens=True
            )
            record = {
                "id": start + j,
                "question": questions[j],
                "response": generated,
                "ground_truth": gold_text(r),
            }
            f.write(json.dumps(record) + "\n")
            print(f"[{start + j + 1}/{total}] {generated[:80].replace(chr(10), ' ')}...")

        completed += len(batch)
        print(f"  batch done — {completed}/{total}")

print(f"\nsaved {total} results to {args.output_file}")
