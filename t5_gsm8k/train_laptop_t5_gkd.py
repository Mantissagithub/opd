import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi, login as hf_login
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SAVE_DIR = SCRIPT_DIR / "t5_small_gsm8k_gkd_true_laptop"
DEFAULT_LOG_DIR = SCRIPT_DIR / "runs" / "t5_small_gsm8k_gkd_true_laptop"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a T5 student with Generalized Knowledge Distillation.")
    parser.add_argument("--student-name", default="google/flan-t5-small")
    parser.add_argument("--teacher-name", default="google/flan-t5-large")
    parser.add_argument("--fallback-teacher-name", default="google/flan-t5-base")
    parser.add_argument("--save-dir", default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--training-steps", type=int, default=4000, help="Optimizer updates.")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--cooldown-begin", type=int, default=3000)
    parser.add_argument("--cooldown-end", type=int, default=4000)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-output-length", type=int, default=320)
    parser.add_argument("--generation-max-new-tokens", type=int, default=192)
    parser.add_argument("--student-data-fraction", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature for on-policy outputs.")
    parser.add_argument("--distill-temperature", type=float, default=1.0, help="Temperature inside the divergence.")
    parser.add_argument("--divergence", choices=["jsd", "forward_kl", "reverse_kl"], default="jsd")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-limit", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo-id")
    parser.add_argument("--hub-token")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    args.save_dir = resolve_path(args.save_dir)
    args.log_dir = resolve_path(args.log_dir)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.use_fp16_teacher = args.device.type == "cuda"
    args.gradient_checkpointing = args.gradient_checkpointing or args.device.type == "cuda"

    if args.smoke:
        args.training_steps = 2
        args.micro_batch_size = 1
        args.grad_accum_steps = 1
        args.warmup_steps = 1
        args.cooldown_begin = 1
        args.cooldown_end = 2
        args.eval_every = 1
        args.log_every = 1
        args.eval_limit = min(args.eval_limit, 8)
        args.save_dir = resolve_path(f"{args.save_dir}_smoke")
        args.log_dir = resolve_path(f"{args.log_dir}_smoke")

    args.generation_max_new_tokens = min(args.generation_max_new_tokens, args.max_output_length)
    args.student_data_fraction = min(max(args.student_data_fraction, 0.0), 1.0)
    args.beta = min(max(args.beta, 0.0), 1.0)

    if args.cooldown_begin > args.cooldown_end:
        raise ValueError("cooldown_begin must be <= cooldown_end")
    if args.cooldown_end > args.training_steps:
        raise ValueError("cooldown_end must be <= training_steps")
    if args.push_to_hub and not args.hub_repo_id:
        raise ValueError("--hub-repo-id is required with --push-to-hub")

    return args


def resolve_path(path_str):
    path = Path(path_str)
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def preview(text, limit=96):
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def next_rows(dataset, order, cursor, batch_size, rng):
    if cursor + batch_size > len(order):
        rng.shuffle(order)
        cursor = 0
    batch_indices = order[cursor : cursor + batch_size]
    rows = [dataset[int(idx)] for idx in batch_indices]
    return rows, cursor + batch_size


def encode_prompts(prompts, tokenizer, args):
    encoded = tokenizer(
        prompts,
        truncation=True,
        padding=True,
        max_length=args.max_input_length,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"].to(args.device),
        "attention_mask": encoded["attention_mask"].to(args.device),
    }


def encode_targets(targets, tokenizer, args):
    encoded = tokenizer(
        targets,
        truncation=True,
        padding=True,
        max_length=args.max_output_length,
        return_tensors="pt",
    )
    labels = encoded["input_ids"].to(args.device)
    labels_for_model = labels.clone()
    labels_for_model[labels_for_model == tokenizer.pad_token_id] = -100
    return labels, labels_for_model


def batch_from_rows(rows, tokenizer, args, targets=None):
    prompts = [make_prompt(row["question"]) for row in rows]
    targets = targets or [row["answer"].strip() for row in rows]
    prompt_batch = encode_prompts(prompts, tokenizer, args)
    labels, labels_for_model = encode_targets(targets, tokenizer, args)
    return {
        "prompts": prompts,
        "targets": targets,
        "input_ids": prompt_batch["input_ids"],
        "attention_mask": prompt_batch["attention_mask"],
        "labels": labels,
        "labels_for_model": labels_for_model,
    }


def set_model_dropout(model, dropout):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout


def generate_student_targets(student, tokenizer, input_ids, attention_mask, args):
    was_training = student.training
    use_cache = getattr(student.config, "use_cache", True)
    student.eval()
    student.config.use_cache = True
    with torch.no_grad():
        generated_ids = student.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            top_k=0,
            temperature=args.temperature,
            max_new_tokens=args.generation_max_new_tokens,
        )
    student.config.use_cache = use_cache
    if was_training:
        student.train()
    return [text.strip() or "no answer" for text in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]


def masked_reduce(loss_tensor, labels_for_model):
    mask = labels_for_model != -100
    if not mask.any():
        return loss_tensor.new_tensor(0.0)
    return loss_tensor[mask].sum() / mask.sum()


def forward_kl_loss(student_logits, teacher_logits, labels_for_model, temperature):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    per_token = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(dim=-1)
    return masked_reduce(per_token, labels_for_model)


def reverse_kl_loss(student_logits, teacher_logits, labels_for_model, temperature):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    per_token = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True).sum(dim=-1)
    return masked_reduce(per_token, labels_for_model)


def generalized_jsd_loss(student_logits, teacher_logits, labels_for_model, beta, temperature):
    if beta <= 0.0:
        return forward_kl_loss(student_logits, teacher_logits, labels_for_model, temperature)
    if beta >= 1.0:
        return reverse_kl_loss(student_logits, teacher_logits, labels_for_model, temperature)

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    beta_tensor = student_log_probs.new_tensor(beta)
    mixture_log_probs = torch.logsumexp(
        torch.stack(
            [
                student_log_probs + torch.log(beta_tensor),
                teacher_log_probs + torch.log1p(-beta_tensor),
            ]
        ),
        dim=0,
    )
    kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(dim=-1)
    kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True).sum(dim=-1)
    per_token = beta_tensor * kl_teacher + (1.0 - beta_tensor) * kl_student
    return masked_reduce(per_token, labels_for_model)


def compute_gkd_loss(student_logits, teacher_logits, labels_for_model, args):
    if args.divergence == "forward_kl":
        return forward_kl_loss(student_logits, teacher_logits, labels_for_model, args.distill_temperature)
    if args.divergence == "reverse_kl":
        return reverse_kl_loss(student_logits, teacher_logits, labels_for_model, args.distill_temperature)
    return generalized_jsd_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels_for_model=labels_for_model,
        beta=args.beta,
        temperature=args.distill_temperature,
    )


def evaluate(model, tokenizer, dataset, args):
    was_training = model.training
    use_cache = getattr(model.config, "use_cache", True)
    model.eval()
    model.config.use_cache = True
    correct = 0
    total = 0
    limit = min(len(dataset), args.eval_limit)

    with torch.no_grad():
        for start in range(0, limit, args.eval_batch_size):
            rows = [dataset[idx] for idx in range(start, min(start + args.eval_batch_size, limit))]
            prompts = [make_prompt(row["question"]) for row in rows]
            gold_answers = [extract_answer(row["answer"]) for row in rows]
            prompt_batch = encode_prompts(prompts, tokenizer, args)
            generated_ids = model.generate(
                input_ids=prompt_batch["input_ids"],
                attention_mask=prompt_batch["attention_mask"],
                max_new_tokens=args.eval_max_new_tokens,
            )
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for pred, gold in zip(predictions, gold_answers):
                if extract_answer(pred) == gold:
                    correct += 1
                total += 1

    model.config.use_cache = use_cache
    if was_training:
        model.train()
    return correct / max(total, 1)


def lr_lambda(step, args):
    if step < args.warmup_steps:
        return step / max(args.warmup_steps, 1)
    if step >= args.cooldown_end:
        return 0.0
    if step >= args.cooldown_begin:
        span = args.cooldown_end - args.cooldown_begin
        return 1.0 - (step - args.cooldown_begin) / max(span, 1)
    return 1.0


def load_seq2seq_model(name_or_path, device, torch_dtype=None):
    kwargs = {}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForSeq2SeqLM.from_pretrained(name_or_path, **kwargs)
    return model.to(device)


def load_teacher_with_fallback(args):
    preferred_dtype = torch.float16 if args.use_fp16_teacher else None
    candidate_names = [args.teacher_name]
    if args.fallback_teacher_name and args.fallback_teacher_name != args.teacher_name:
        candidate_names.append(args.fallback_teacher_name)

    last_error = None
    for i, candidate in enumerate(candidate_names):
        try:
            teacher = load_seq2seq_model(candidate, args.device, preferred_dtype)
            if preferred_dtype is not None:
                teacher = teacher.half()
            return teacher, candidate
        except RuntimeError as exc:
            last_error = exc
            if "out of memory" not in str(exc).lower() or i == len(candidate_names) - 1:
                raise
            print(f"[teacher] load OOM for {candidate}; retrying with {candidate_names[i + 1]}.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise last_error


def maybe_probe_teacher(teacher, tokenizer, dataset, args, loaded_teacher_name):
    rows = [dataset[idx] for idx in range(min(args.micro_batch_size, len(dataset)))]
    sample_batch = batch_from_rows(rows, tokenizer, args)
    try:
        with torch.no_grad():
            teacher(
                input_ids=sample_batch["input_ids"],
                attention_mask=sample_batch["attention_mask"],
                labels=sample_batch["labels_for_model"],
            )
        return teacher, loaded_teacher_name
    except RuntimeError as exc:
        can_fallback = (
            "out of memory" in str(exc).lower()
            and args.fallback_teacher_name
            and loaded_teacher_name != args.fallback_teacher_name
        )
        if not can_fallback:
            raise
        print(f"[teacher] probe OOM for {loaded_teacher_name}; retrying with {args.fallback_teacher_name}.")
        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fallback_teacher = load_seq2seq_model(
            args.fallback_teacher_name,
            args.device,
            torch.float16 if args.use_fp16_teacher else None,
        )
        if args.use_fp16_teacher:
            fallback_teacher = fallback_teacher.half()
        return fallback_teacher, args.fallback_teacher_name


def format_setup_table(args, loaded_teacher_name):
    rows = [
        ("student", args.student_name),
        ("teacher", loaded_teacher_name),
        ("device", str(args.device)),
        ("divergence", args.divergence),
        ("beta", f"{args.beta:.2f}"),
        ("lambda", f"{args.student_data_fraction:.2f}"),
        ("updates", str(args.training_steps)),
        ("micro batch", str(args.micro_batch_size)),
        ("grad accum", str(args.grad_accum_steps)),
        ("effective batch", str(args.micro_batch_size * args.grad_accum_steps)),
        ("save dir", str(args.save_dir)),
    ]
    width = max(len(key) for key, _ in rows)
    return "\n".join(f"{key:<{width}} : {value}" for key, value in rows)


def maybe_push_to_hub(student, tokenizer, args):
    if not args.push_to_hub:
        return

    if args.hub_token:
        hf_login(token=args.hub_token, add_to_git_credential=False)
    api = HfApi()
    api.create_repo(repo_id=args.hub_repo_id, private=False, exist_ok=True)
    api.upload_folder(
        repo_id=args.hub_repo_id,
        folder_path=str(args.save_dir),
        commit_message="Train true GKD student",
    )
    print(f"[hub] pushed to https://huggingface.co/{args.hub_repo_id}")


def choose_mode(global_micro_step, args, rng):
    if args.smoke and global_micro_step == 0:
        return "supervised"
    if args.smoke and global_micro_step == 1:
        return "on_policy"
    return "on_policy" if rng.random() <= args.student_data_fraction else "supervised"


def gpu_memory_gb():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def main():
    args = parse_args()
    set_seed(args.seed)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    if args.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.student_name)
    train_dataset = load_dataset("openai/gsm8k", "main", split="train")
    eval_dataset = load_dataset("openai/gsm8k", "main", split="test")

    student = load_seq2seq_model(args.student_name, args.device)
    student.config.dropout_rate = args.dropout
    set_model_dropout(student, args.dropout)
    student.config.use_cache = False
    if args.gradient_checkpointing and hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()

    teacher, loaded_teacher_name = load_teacher_with_fallback(args)
    teacher, loaded_teacher_name = maybe_probe_teacher(teacher, tokenizer, train_dataset, args, loaded_teacher_name)
    teacher.eval()
    teacher.config.use_cache = False
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    if student.config.vocab_size != teacher.config.vocab_size:
        raise ValueError("Student and teacher must share the same vocabulary size for token-level GKD.")

    print("\n[GKD setup]")
    print(format_setup_table(args, loaded_teacher_name))
    print()

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_lambda(step, args))
    writer = SummaryWriter(log_dir=str(args.log_dir))

    train_order = list(range(len(train_dataset)))
    rng = random.Random(args.seed)
    rng.shuffle(train_order)
    cursor = 0
    global_micro_step = 0
    window_loss = 0.0
    window_micro_steps = 0
    last_targets = []
    last_mode_counts = {"supervised": 0, "on_policy": 0}
    last_eval_accuracy = None

    optimizer.zero_grad(set_to_none=True)
    student.train()

    progress = tqdm(
        total=args.training_steps,
        desc="train",
        dynamic_ncols=True,
        smoothing=0.05,
        leave=True,
    )
    try:
        for update_step in range(1, args.training_steps + 1):
            mode_counts = {"supervised": 0, "on_policy": 0}
            update_loss_sum = 0.0
            grad_norm = None

            for _ in range(args.grad_accum_steps):
                rows, cursor = next_rows(train_dataset, train_order, cursor, args.micro_batch_size, rng)
                mode = choose_mode(global_micro_step, args, rng)
                mode_counts[mode] += 1
                global_micro_step += 1

                if mode == "on_policy":
                    prompts = [make_prompt(row["question"]) for row in rows]
                    prompt_batch = encode_prompts(prompts, tokenizer, args)
                    sampled_targets = generate_student_targets(
                        student,
                        tokenizer,
                        prompt_batch["input_ids"],
                        prompt_batch["attention_mask"],
                        args,
                    )
                    batch = batch_from_rows(rows, tokenizer, args, targets=sampled_targets)
                else:
                    batch = batch_from_rows(rows, tokenizer, args)

                student_outputs = student(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels_for_model"],
                )
                with torch.no_grad():
                    teacher_outputs = teacher(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels_for_model"],
                    )

                raw_loss = compute_gkd_loss(
                    student_logits=student_outputs.logits,
                    teacher_logits=teacher_outputs.logits.float(),
                    labels_for_model=batch["labels_for_model"],
                    args=args,
                )
                (raw_loss / args.grad_accum_steps).backward()

                loss_value = raw_loss.detach().item()
                update_loss_sum += loss_value
                window_loss += loss_value
                window_micro_steps += 1
                last_targets = batch["targets"]

            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            for parameter in student.parameters():
                if parameter.grad is not None:
                    grad_norm = grad_norm or 0.0
                    grad_norm += parameter.grad.detach().norm().item() ** 2
            if grad_norm is not None:
                grad_norm = grad_norm ** 0.5

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            last_mode_counts = mode_counts
            mean_update_loss = update_loss_sum / max(args.grad_accum_steps, 1)
            writer.add_scalar("train/loss_update", mean_update_loss, update_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], update_step)
            writer.add_scalar("train/on_policy_fraction", mode_counts["on_policy"] / args.grad_accum_steps, update_step)
            if grad_norm is not None:
                writer.add_scalar("train/grad_norm", grad_norm, update_step)

            gpu_mem = gpu_memory_gb()
            postfix = {
                "loss": f"{mean_update_loss:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "sup": mode_counts["supervised"],
                "onp": mode_counts["on_policy"],
            }
            if gpu_mem is not None:
                postfix["gpu"] = f"{gpu_mem:.2f}G"
            progress.set_postfix(postfix)
            progress.update(1)

            if update_step % args.log_every == 0 or update_step == 1:
                mean_window_loss = window_loss / max(window_micro_steps, 1)
                progress.write(
                    f"[train] step={update_step}/{args.training_steps} "
                    f"window_loss={mean_window_loss:.4f} "
                    f"sample={preview(last_targets[0]) if last_targets else 'n/a'}"
                )
                window_loss = 0.0
                window_micro_steps = 0

            if update_step % args.eval_every == 0 or (args.smoke and update_step == args.training_steps):
                last_eval_accuracy = evaluate(student, tokenizer, eval_dataset, args)
                writer.add_scalar("eval/accuracy", last_eval_accuracy, update_step)
                progress.write(f"[eval] step={update_step} accuracy={last_eval_accuracy:.4f}")
    finally:
        progress.close()

    student.config.tie_word_embeddings = False
    student.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    writer.close()

    print("\n[done]")
    print(f"saved model      : {args.save_dir}")
    print(f"teacher used     : {loaded_teacher_name}")
    print(
        "last mode mix    : "
        f"supervised={last_mode_counts['supervised']} "
        f"on_policy={last_mode_counts['on_policy']}"
    )
    print(f"last eval acc    : {last_eval_accuracy if last_eval_accuracy is not None else 'not run'}")

    maybe_push_to_hub(student, tokenizer, args)


if __name__ == "__main__":
    main()
