import argparse

import sft

# hub repo follows <model>-<ds>-<parameters>: qwen3-tweet-style-4b
CFG = {
    "base_model": "Qwen/Qwen3-4B-Base",
    "repo_name": "qwen3-tweet-style-4b",
    "batch_size": 4,
    "grad_accum": 4,
    "epochs": 3,
    "lr": 2e-4,
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
