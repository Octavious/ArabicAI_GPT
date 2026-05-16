"""
Upload Arabic AI TinyStories GPT checkpoint to Hugging Face Hub.

Run from this folder (model-repo/) after you have:
  - checkpoint.pt in ../../  (episode-06 root), or pass --checkpoint
  - tinystories_tokenizer.json in ../../
  - huggingface-cli login  (or HF_TOKEN set)

Example:
  python upload_model.py --repo-id luayas1977/arabicai-tinystories-gpt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo

# Paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
EPISODE_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_CHECKPOINT = EPISODE_ROOT / "checkpoint.pt"
DEFAULT_TOKENIZER = EPISODE_ROOT / "tinystories_tokenizer.json"
DEFAULT_MODEL_PY = EPISODE_ROOT / "model.py"
CONFIG_JSON = SCRIPT_DIR / "config.json"
MODEL_CARD = SCRIPT_DIR / "README.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload TinyStories GPT to Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hub repo id, e.g. luayas1977/arabicai-tinystories-gpt",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to checkpoint.pt (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER,
        help=f"Path to tokenizer JSON (default: {DEFAULT_TOKENIZER})",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private model repo",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files only; do not upload",
    )
    args = parser.parse_args()

    for path, label in [
        (args.checkpoint, "checkpoint.pt"),
        (args.tokenizer, "tinystories_tokenizer.json"),
        (CONFIG_JSON, "config.json"),
        (MODEL_CARD, "README.md (model card)"),
        (DEFAULT_MODEL_PY, "model.py"),
    ]:
        if not path.exists():
            print(f"ERROR: Missing {label}: {path}", file=sys.stderr)
            sys.exit(1)

    # Quick sanity check on checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    for key in ("model", "config", "step", "val_loss"):
        if key not in ckpt:
            print(f"WARNING: checkpoint missing key '{key}'")

    step = ckpt.get("step", "?")
    val_loss = ckpt.get("val_loss", "?")
    print(f"  step={step}, val_loss={val_loss}")

    if args.dry_run:
        print("Dry run OK — all files present.")
        return

    api = HfApi()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print(f"Creating repo: {args.repo_id} (private={args.private})")
    create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
        token=token,
    )

    staging = SCRIPT_DIR / "_upload_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    try:
        # Hub filename model.pt — full checkpoint dict for easy reload
        shutil.copy2(args.checkpoint, staging / "model.pt")
        shutil.copy2(args.tokenizer, staging / "tinystories_tokenizer.json")
        shutil.copy2(CONFIG_JSON, staging / "config.json")
        shutil.copy2(MODEL_CARD, staging / "README.md")
        shutil.copy2(DEFAULT_MODEL_PY, staging / "model.py")

        # Optional metadata sidecar
        meta = {
            "step": ckpt.get("step"),
            "val_loss": float(ckpt["val_loss"]) if ckpt.get("val_loss") is not None else None,
        }
        (staging / "training_info.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print(f"Uploading folder to {args.repo_id} ...")
        api.upload_folder(
            folder_path=str(staging),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Upload TinyStories GPT checkpoint and tokenizer",
            token=token,
        )
        print(f"Done: https://huggingface.co/{args.repo_id}")
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
