# in ref of this paper: https://arxiv.org/abs/2306.13649

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import requests

TEACHER_URL = "http://100.113.133.12:8000"

# hyperparameters from the paper
H = {
    "training_steps": 40000,
    "batch_size": 32,
    "dropout": 0.05,
    "lr": 0.0003,
    "warmup_steps": 2000,
    "cooldown_begin": 30000,
    "cooldown_end": 40000,
    "max_input_length": 512,
    "max_output_length": 320,
    "teacher_temp": 0.1,
    "student_data_fraction": 0.5,
    "eval_every": 1000,
}

def load_gsm8k():
    return load_dataset("gsm8k", split="train")

class Gsm8kDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, tokenizer):
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    enc = tokenizer(questions, truncation=True, padding=True, max_length=H["max_input_length"], return_tensors="pt")
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "questions": questions, "answers": answers}

def get_teacher_logprobs(prompts):
    # hits the teacher api for each prompt, gets back generated text + per-token logprobs
    import time
    results = []
    for p in prompts:
        try:
            response = client.chat.completions.create(
                model="deepseek/deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": p},
                ],
                logprobs=True,
                max_tokens=H["max_output_length"],
            )
            choice = response.choices[0]
            text = choice.message.content or ""
            logprobs_list = []
            if choice.logprobs:
                logprobs_list = [
                    {"token": lp.token, "logprob": lp.logprob}
                    for lp in choice.logprobs.content
                ]
            results.append((text, logprobs_list))
        except Exception as e:
            print(f"teacher api call failed: {e}")
            results.append(("", []))
        time.sleep(0.1)
    return results

def compute_forward_kl(student_logits, target_ids, teacher_lps_list, temperature, vocab_size):
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_targets = target_ids[:, 1:].contiguous()
    student_logprobs = torch.log_softmax(shift_logits / temperature, dim=-1)

    total = 0.0
    count = 0

    for b in range(shift_targets.size(0)):
        t_lps = teacher_lps_list[b]
        for pos in range(min(shift_targets.size(1), len(t_lps))):
            tid = shift_targets[b, pos]
            if tid == 0 or tid >= vocab_size:
                continue
            teacher_logprob = t_lps[pos]["logprob"]
            w = torch.exp(torch.tensor(teacher_logprob / temperature, device=student_logprobs.device))
            total = total - w * student_logprobs[b, pos, tid]
            count += 1

    return total / max(count, 1)

def evaluate(model, tokenizer, dataloader, num_samples=200):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            gen_ids = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=H["max_output_length"],
            )
            for i in range(len(batch["questions"])):
                gen_text = tokenizer.decode(gen_ids[i], skip_special_tokens=True)
                gt_answer = batch["answers"][i].split("####")[-1].strip() if "####" in batch["answers"][i] else batch["answers"][i]
                if gt_answer and gen_text.strip().endswith(gt_answer):
                    correct += 1
                total += 1
                if total >= num_samples:
                    break
            if total >= num_samples:
                break
    model.train()
    return correct / max(total, 1)

def train():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    model.config.dropout_rate = H["dropout"]
    vocab_size = model.config.vocab_size

    dataset = Gsm8kDataset(load_gsm8k())
    dataloader = DataLoader(
        dataset, batch_size=H["batch_size"], shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=H["lr"])

    def lr_lambda(step):
        if step < H["warmup_steps"]:
            return step / H["warmup_steps"]
        elif step >= H["cooldown_end"]:
            return 0.0
        elif step >= H["cooldown_begin"]:
            return 1.0 - (step - H["cooldown_begin"]) / (H["cooldown_end"] - H["cooldown_begin"])
        else:
            return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step = 0
    train_iter = iter(dataloader)
    lambd = H["student_data_fraction"]

    while step < H["training_steps"]:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(dataloader)
            batch = next(train_iter)

        questions = batch["questions"]
        bs = len(questions)
        on_policy_mask = torch.rand(bs) < lambd
        sup_mask = ~on_policy_mask
        loss_total = 0.0

        # off-policy half — y comes from the teacher, we minimize kl between teacher and student on those
        if sup_mask.any():
            sup_questions = [questions[i] for i in range(bs) if sup_mask[i]]
            teacher_prompts = [f"Question: {q}\nAnswer:" for q in sup_questions]
            teacher_results = get_teacher_logprobs(teacher_prompts)
            teacher_texts = [r[0] for r in teacher_results]
            teacher_lps = [r[1] for r in teacher_results]

            target_enc = tokenizer(teacher_texts, truncation=True, padding=True,
                                   max_length=H["max_output_length"], return_tensors="pt")

            outputs = model(
                input_ids=batch["input_ids"][sup_mask],
                attention_mask=batch["attention_mask"][sup_mask],
                labels=target_enc["input_ids"],
            )

            loss_sup = compute_forward_kl(
                outputs.logits, target_enc["input_ids"],
                teacher_lps, H["teacher_temp"], vocab_size
            )
            loss_total = loss_total + (1 - lambd) * loss_sup

        # on-policy half — student generates its own y, teacher scores it, we train on that
        if on_policy_mask.any():
            on_questions = [questions[i] for i in range(bs) if on_policy_mask[i]]

            with torch.no_grad():
                student_gen_ids = model.generate(
                    batch["input_ids"][on_policy_mask],
                    attention_mask=batch["attention_mask"][on_policy_mask],
                    max_new_tokens=H["max_output_length"],
                    do_sample=True, temperature=1.0,
                )

            student_gen_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in student_gen_ids]
            on_prompts = [f"Question: {q}\nAnswer: {t}" for q, t in zip(on_questions, student_gen_texts)]
            teacher_results_on = get_teacher_logprobs(on_prompts)
            teacher_lps_on = [r[1] for r in teacher_results_on]

            target_enc_on = tokenizer(student_gen_texts, truncation=True, padding=True,
                                      max_length=H["max_output_length"], return_tensors="pt")

            outputs_on = model(
                input_ids=batch["input_ids"][on_policy_mask],
                attention_mask=batch["attention_mask"][on_policy_mask],
                labels=target_enc_on["input_ids"],
            )

            loss_on = compute_forward_kl(
                outputs_on.logits, target_enc_on["input_ids"],
                teacher_lps_on, H["teacher_temp"], vocab_size
            )
            loss_total = loss_total + lambd * loss_on

        if not sup_mask.any() and not on_policy_mask.any():
            step += 1
            continue

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"step {step} | loss {loss_total.item():.4f} | lr {lr_now:.6f} | "
                  f"sup {sup_mask.sum().item()}/{bs} | onpol {on_policy_mask.sum().item()}/{bs}")

        if step % H["eval_every"] == 0 and step > 0:
            acc = evaluate(model, tokenizer, dataloader)
            print(f"step {step} | eval accuracy {acc:.4f}")

        step += 1

    model.save_pretrained("t5_gsm8k_gkd")
    tokenizer.save_pretrained("t5_gsm8k_gkd")
    print("training done, model saved to t5_gsm8k_gkd")

if __name__ == "__main__":
    train()
