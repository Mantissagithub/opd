import argparse

import sft

# same setup as the 4b, just a smaller per-device batch + bigger grad-accum so the
# 32b base fits in 4-bit. hub repo: qwen3-tweet-style-32b
CFG = {
    "base_model": "Qwen/Qwen3-32B-Base",
    "repo_name": "qwen3-tweet-style-32b",
    "batch_size": 1,
    "grad_accum": 16,
    "epochs": 3,
    "lr": 1e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "max_seq_len": 512,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    CFG["push"] = args.push
    sft.run_sft(CFG)
