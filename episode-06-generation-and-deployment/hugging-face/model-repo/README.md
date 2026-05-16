---
license: mit
tags:
  - gpt
  - pytorch
  - text-generation
  - tinystories
  - from-scratch
language:
  - en
library_name: pytorch
---

# Arabic AI TinyStories GPT (12M parameters)

A small decoder-only GPT trained from scratch on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). Built line-by-line in PyTorch as part of the **Build a GPT From Scratch on a 6 GB GPU** series — no `transformers` model classes.

## Model description

| Property | Value |
|----------|-------|
| Parameters | ~12.4M |
| Layers | 6 |
| Heads | 6 |
| Embedding dim | 384 |
| Context length | 256 tokens |
| Vocabulary | 4,096 (custom BPE) |
| Training data | TinyStories |
| Training steps | ~25,000 |
| Validation loss | ~1.7 (varies by run) |

The model writes short children's stories in English. It was trained with a custom tokenizer (`tinystories_tokenizer.json`) and a hand-written `GPT` class in pure PyTorch.

## Files in this repository

| File | Description |
|------|-------------|
| `model.pt` | Full training checkpoint (`model`, `config`, `step`, `val_loss`) |
| `config.json` | Architecture hyperparameters |
| `tinystories_tokenizer.json` | BPE tokenizer from Episode 2 |
| `model.py` | `GPT` and `GPTConfig` source (for loading locally) |

## Usage

### Install dependencies

```bash
pip install torch tokenizers huggingface_hub
```

### Download and generate

```python
import json
import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO_ID = "luayas1977/arabicai-tinystories-gpt"

# Download artifacts
ckpt_path = hf_hub_download(repo_id=REPO_ID, filename="model.pt")
tok_path = hf_hub_download(repo_id=REPO_ID, filename="tinystories_tokenizer.json")
model_py = hf_hub_download(repo_id=REPO_ID, filename="model.py")

# Load architecture (download model.py into cwd or add to path)
import importlib.util
spec = importlib.util.spec_from_file_location("gpt_model", model_py)
gpt_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpt_model)
GPT = gpt_model.GPT

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

model = GPT(checkpoint["config"]).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

tokenizer = Tokenizer.from_file(tok_path)
prompt = "Once upon a time"
prompt_ids = tokenizer.encode(prompt).ids
idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

eos_id = tokenizer.token_to_id("<|endoftext|>")
with torch.no_grad():
    out = model.generate(idx, max_new_tokens=200, temperature=0.8, top_k=40, eos_token_id=eos_id)

text = tokenizer.decode(out[0].tolist())
print(text)
```

### Try it in the browser

**[Arabic AI TinyStories GPT Playground](https://huggingface.co/spaces/luayas1977/arabicai-tinystories-gpt)** — interactive Gradio demo with temperature, top-k, and max-tokens controls.

Model repo: [luayas1977/arabicai-tinystories-gpt](https://huggingface.co/luayas1977/arabicai-tinystories-gpt)

## Training details

- **Optimizer:** AdamW (lr 3e-4, weight decay 0.1)
- **Schedule:** Linear warmup (500 steps) + cosine decay
- **Batch size:** 32 sequences × 256 tokens
- **Hardware:** Consumer GPU (~6 GB VRAM)
- **Runtime:** ~2 hours for 25,000 steps

## Limitations

- Small model trained only on TinyStories — not suitable for general knowledge, code, or adult topics
- English only
- Context limited to 256 tokens
- May repeat story openers or run past a natural ending if EOS was not learned strongly (depends on data prep)

## Citation / series

Part of **Build a GPT From Scratch on a 6 GB GPU** — [YouTube playlist](https://www.youtube.com/playlist?list=PLvmpljk9TE2v4LbSrQxmnbOKkLa0Mw1oD).
