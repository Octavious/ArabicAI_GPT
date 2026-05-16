---
title: Arabic AI TinyStories GPT Playground
emoji: 📖
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
python_version: "3.11"
models:
  - luayas1977/arabicai-tinystories-gpt
pinned: false
license: mit
---

Interactive demo for the **TinyStories GPT** model from the *Build a GPT From Scratch on a 6 GB GPU* series.

## Space configuration

After creating this Space on Hugging Face, set a **Repository variable**:

| Name | Value | Example |
|------|-------|---------|
| `HF_MODEL_REPO` | Your uploaded model repo id | `luayas1977/arabicai-tinystories-gpt` |

The app downloads `model.pt` and `tinystories_tokenizer.json` from that repo at startup.

## Files

| File | Role |
|------|------|
| `app.py` | Gradio UI and inference |
| `model.py` | GPT architecture |
| `inference_utils.py` | Story-boundary trimming fallback |
| `requirements.txt` | Python dependencies |

See the parent [hugging-face guide](../README.md) for upload and deployment steps.
