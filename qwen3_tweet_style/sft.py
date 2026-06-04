import os

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

DATASET_ID = "Pradheep1647/tweet-style-dataset"

INSTRUCTION_PREFIX = "### Instruction:\n"
RESPONSE_TEMPLATE = "\n### Response:\n"

def build_prompt(instruction):
    return f"{INSTRUCTION_PREFIX}{instruction}{RESPONSE_TEMPLATE}"

def run_sft(cfg):
    hf_username = os.environ.get("HF_USERNAME")
    hf_token = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg.get("load_4bit", True):
        # qlora: 4-bit nf4 base + bf16 lora on top. works for dense models.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=bnb_config, device_map="auto")
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
    else:
        # moe experts don't 4-bit quantize, so load bf16 and do plain lora instead.
        model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=torch.bfloat16, device_map="auto")
        model.config.use_cache = False
        model.enable_input_require_grads()  # needed for grad checkpointing + lora

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
    )

    # feed it as prompt/completion so trl masks the instruction and only trains on the response
    def to_prompt_completion(batch):
        return {
            "prompt": [build_prompt(instr) for instr in batch["instruction"]],
            "completion": list(batch["response"]),
        }

    dataset = load_dataset(DATASET_ID)
    train_ds = dataset["train"].map(to_prompt_completion, batched=True, remove_columns=dataset["train"].column_names)
    eval_ds = dataset["validation"].map(to_prompt_completion, batched=True, remove_columns=dataset["validation"].column_names)

    sft_config = SFTConfig(
        output_dir=cfg["repo_name"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        max_length=cfg["max_seq_len"],
        completion_only_loss=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(cfg["repo_name"])
    tokenizer.save_pretrained(cfg["repo_name"])
    print(f"training done, adapter saved to {cfg['repo_name']}")

    if cfg.get("push"):
        if not (hf_username and hf_token):
            raise RuntimeError("--push needs HF_USERNAME and HF_TOKEN in the env")
        repo_id = f"{hf_username}/{cfg['repo_name']}"
        trainer.model.push_to_hub(repo_id, token=hf_token)
        tokenizer.push_to_hub(repo_id, token=hf_token)
        print(f"pushed adapter to https://huggingface.co/{repo_id}")
