import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config, TrainerCallback
from trl import DPOConfig, DPOTrainer


# offline dpo on a static preference set (paperbd/paper_preference_150K-v1).
# works for both paperhound models (lora on attention q/k/v/o):
#   - gpt-oss-20b: hf Trainer refuses to train an MXFP4 model, so dequantize experts to
#     bf16 (--dequantize, default) + attn_implementation=eager (no sdpa kernel). ~40gb/80gb,
#     fits since peft only saves the adapter (no full gather).
#   - smollm2-135m: --no-dequantize --attn sdpa, trivially small.
# speed: peft model is its own reference (adapter disabled), so no 2nd copy of the base.


class PhaseMetrics(TrainerCallback):
    # log eval metrics tagged by phase so the model->(+dpo) story is plottable:
    #   phase=model      -> ref == policy, accuracies ~0.5, margin ~0
    #   phase=model+dpo  -> after training, accuracies up, margin > 0
    def __init__(self, trainer):
        self.trainer = trainer

    def _emit(self, phase):
        m = self.trainer.evaluate()
        flat = {k.split("/")[-1]: v for k, v in m.items() if "rewards/" in k or k.endswith("loss")}
        print(f"[phase={phase}] " + json.dumps(flat))
        return flat

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.before = self._emit("model")

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.after = self._emit("model+dpo")
            print("[dpo_phase_summary] " + json.dumps({"model": self.before, "model+dpo": self.after}))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/paperhound_dpo")
    p.add_argument("--model-path", default="openai/gpt-oss-20b")
    p.add_argument("--output-dir", default="checkpoints/paperhound-gpt-oss-20b-dpo")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=1536)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--attn", default="eager")  # gpt-oss needs eager; smollm2 is fine on sdpa
    p.add_argument("--dequantize", action=argparse.BooleanOptionalAction, default=True,
                   help="dequantize MXFP4 experts (gpt-oss); use --no-dequantize for non-quantized models")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # for gpt-oss: dequantize MXFP4 experts -> bf16 so the hf Trainer will train it (kernels
    # must be installed). non-quantized models (smollm2) load straight with --no-dequantize.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation=args.attn,
        dtype=torch.bfloat16,
        use_cache=False,
        quantization_config=Mxfp4Config(dequantize=True) if args.dequantize else None,
    )

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    cfg = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        loss_type="sigmoid",
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch,
        per_device_eval_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        max_length=args.max_length,
        truncation_mode="keep_end",  # keep the end of the prompt (paper tail + query)
        # speed knobs. precompute_ref_log_probs is OFF: for 1 epoch it just adds a
        # separate full ref pass (~same total compute) and delays training start; with
        # peft the ref is the adapter-disabled base computed inline, no 2nd 20b copy.
        precompute_ref_log_probs=False,
        optim="adamw_torch_fused",
        dataset_num_proc=os.cpu_count(),
        remove_unused_columns=False,
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="epoch",
        report_to="none",
    )

    train = load_dataset("parquet", data_files=f"{args.data_dir}/train.parquet", split="train")
    eval_ds = load_dataset("parquet", data_files=f"{args.data_dir}/test.parquet", split="train")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # peft -> reference is the adapter-disabled base, no 2nd 20b copy
        args=cfg,
        train_dataset=train,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.add_callback(PhaseMetrics(trainer))

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
