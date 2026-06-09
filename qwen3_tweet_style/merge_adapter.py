import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# the 30b-a3b adapter only touches attention; peft's moe expert-key remap is a no-op for us
# and trips a peft<->transformers api skew, so neutralize it (same guard as eval_benchmark.py).
# defensive: the submodule only exists on some peft/transformers combos — skip if absent.
try:
    import peft.utils.transformers_weight_conversion as _twc

    _twc.build_peft_weight_mapping = lambda *a, **k: {}
except Exception:
    pass

# passthrough chat template that reproduces sft.build_prompt exactly:
#   "### Instruction:\n{content}\n### Response:\n"
# so verl's apply_chat_template at rollout matches how the warm-up models were trained.
SFT_CHAT_TEMPLATE = (
    "{% for message in messages %}{% if message['role'] == 'user' %}"
    "### Instruction:\n{{ message['content'] }}\n### Response:\n"
    "{% endif %}{% endfor %}"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="base model id, e.g. Qwen/Qwen3-4B-Base")
    p.add_argument("--adapter", required=True, help="lora adapter id/dir, e.g. Pradheep1647/qwen3-tweet-style-4b")
    p.add_argument("--out", required=True, help="output dir for the merged full model")
    args = p.parse_args()

    print(f"loading base {args.base} ...")
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cpu")
    print(f"applying + merging adapter {args.adapter} ...")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = SFT_CHAT_TEMPLATE
    tokenizer.save_pretrained(args.out)
    print(f"merged model + sft chat template saved to {args.out}")


if __name__ == "__main__":
    main()
