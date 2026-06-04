import argparse

import sft

# qwen3 has no 32b base, so the big slot uses the 30b-a3b moe base (3b active).
# smaller per-device batch + bigger grad-accum to fit in 4-bit. hub: qwen3-tweet-style-30b-a3b
CFG = {
    "base_model": "Qwen/Qwen3-30B-A3B-Base",
    "repo_name": "qwen3-tweet-style-30b-a3b",
    "batch_size": 1,
    "grad_accum": 16,
    "epochs": 3,
    "lr": 1e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "max_seq_len": 512,
    # the moe experts won't 4-bit quantize, so train the bf16 base with plain lora.
    # adapt attention only to keep ~61gb model + training under 80gb.
    "load_4bit": False,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    CFG["push"] = args.push
    sft.run_sft(CFG)
