import argparse
import os
import re
from pathlib import Path

from huggingface_hub import HfApi


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def slug(text: str) -> str:
    text = text.split("/")[-1]
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def write_model_card(model_dir: Path, repo_id: str, args: argparse.Namespace) -> None:
    card = f"""---
base_model: {args.base_model}
datasets:
- {args.dataset_id}
library_name: verl
tags:
- sglang
- paperhound
- citation-grounding
---

# {repo_id.split("/")[-1]}

trained with verl for paper-query citation chunk grounding.

- base model: `{args.base_model}`
- dataset: `{args.dataset_id}`
- training hyperparams: `{args.hyperparams}`
- local source folder: `paperhound`

the dataset contains positive cited chunks, not the full arxiv paper haystack, so this model is trained to emit known supporting chunks for a paper/query pair.
"""
    (model_dir / "README.md").write_text(card)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset-id", default="paperbd/paper-cited-chunks-v1")
    parser.add_argument("--hyperparams", required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    load_dotenv(Path(args.env_file))
    username = os.environ.get("HF_USERNAME")
    token = os.environ.get("HF_TOKEN")
    if not username or not token:
        raise RuntimeError(f"HF_USERNAME and HF_TOKEN must be set in env or {args.env_file}")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model dir not found: {model_dir}")

    repo_name = "-".join(
        [
            slug(args.base_model),
            slug(args.dataset_id),
            slug(args.hyperparams),
        ]
    )
    repo_id = f"{username}/{repo_name}"
    write_model_card(model_dir, repo_id, args)

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(model_dir),
        commit_message=f"Upload {repo_name}",
    )
    print(f"pushed model to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
