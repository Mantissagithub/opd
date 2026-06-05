import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Compare baseline T5 models and a trained GSM8K checkpoint.")
    parser.add_argument("--small-model-name", default="google/flan-t5-small")
    parser.add_argument("--base-model-name", default="google/flan-t5-base")
    parser.add_argument("--large-model-name", default="google/flan-t5-large")
    parser.add_argument("--xl-model-name", default="google/flan-t5-xl")
    parser.add_argument("--xxl-model-name", default="google/flan-t5-xxl")
    parser.add_argument("--include-xl", action="store_true")
    parser.add_argument("--include-xxl", action="store_true")
    parser.add_argument("--trained-dir", default="t5_small_gsm8k_gkd_true_laptop")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--results-out", default=None, help="optional path to write accuracies as json")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


def resolve_model_ref(ref):
    path = Path(ref)
    if not path.is_absolute():
        candidate = SCRIPT_DIR / path
        if candidate.exists():
            return str(candidate)
    return ref


def make_prompt(question):
    return f"question: {question}\nanswer:"


def extract_answer(text):
    if "####" in text:
        return text.split("####")[-1].strip()
    parts = text.strip().replace(",", "").split()
    for token in reversed(parts):
        cleaned = token.strip().rstrip(".")
        if cleaned and cleaned.replace("-", "", 1).replace(".", "", 1).isdigit():
            return cleaned
    return text.strip()


def run_eval(model, tokenizer, dataset, args):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad(), tqdm(total=len(dataset), desc="eval", dynamic_ncols=True, leave=False) as progress:
        for start in range(0, len(dataset), args.micro_batch):
            rows = [dataset[idx] for idx in range(start, min(start + args.micro_batch, len(dataset)))]
            prompts = [make_prompt(row["question"]) for row in rows]
            gold_answers = [extract_answer(row["answer"]) for row in rows]

            encoded = tokenizer(
                prompts,
                truncation=True,
                padding=True,
                max_length=args.max_input_length,
                return_tensors="pt",
            )
            generated_ids = model.generate(
                input_ids=encoded["input_ids"].to(args.device),
                attention_mask=encoded["attention_mask"].to(args.device),
                max_new_tokens=args.max_new_tokens,
            )

            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for pred, gold in zip(predictions, gold_answers):
                if extract_answer(pred) == gold:
                    correct += 1
                total += 1
            progress.update(len(rows))
            progress.set_postfix(acc=f"{correct / max(total, 1):.4f}")

    return correct / max(total, 1)


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(resolve_model_ref(args.small_model_name))
    split = "test" if args.limit is None else f"test[:{args.limit}]"
    eval_dataset = load_dataset("openai/gsm8k", "main", split=split)

    models = [
        ("FLAN T5 Small", resolve_model_ref(args.small_model_name)),
        ("FLAN T5 Base", resolve_model_ref(args.base_model_name)),
        ("FLAN T5 Large", resolve_model_ref(args.large_model_name)),
    ]
    if args.include_xl:
        models.append(("FLAN T5 XL", resolve_model_ref(args.xl_model_name)))
    if args.include_xxl:
        models.append(("FLAN T5 XXL", resolve_model_ref(args.xxl_model_name)))
    models.append(("Trained Student", resolve_model_ref(args.trained_dir)))

    # fp32 xl/xxl won't fit, and bf16 is fine for t5 inference (fp16 overflows)
    dtype = torch.bfloat16 if torch.cuda.is_available() else None

    results = []
    for name, model_ref in models:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_ref, dtype=dtype).to(args.device)
        accuracy = run_eval(model, tokenizer, eval_dataset, args)
        results.append((name, accuracy))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    name_width = max(len(name) for name, _ in results)
    print()
    print(f"{'Model':<{name_width}}  Accuracy")
    print(f"{'-' * name_width}  --------")
    for name, accuracy in results:
        print(f"{name:<{name_width}}  {accuracy:.4f}")
    print()

    if args.results_out:
        Path(args.results_out).write_text(json.dumps(dict(results), indent=2))
        print(f"wrote results to {args.results_out}")


if __name__ == "__main__":
    main()
