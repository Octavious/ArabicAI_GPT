"""
Gradio Space — TinyStories GPT playground.

Set the Space secret or environment variable HF_MODEL_REPO to your Hub model id,
e.g. your-username/tinystories-gpt
"""

from __future__ import annotations

import os

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from inference_utils import truncate_at_story_restart
from model import GPT

# Change this after upload, or set HF_MODEL_REPO in Space Settings → Variables
DEFAULT_REPO = "luayas1977/arabicai-tinystories-gpt"
MODEL_REPO = os.environ.get("HF_MODEL_REPO", DEFAULT_REPO)

_model = None
_tokenizer = None
_eos_id: int | None = None
_device = "cpu"
_checkpoint_info: dict = {}


def load_model() -> str:
    """Download weights from Hub and load into memory. Called once at startup."""
    global _model, _tokenizer, _eos_id, _device, _checkpoint_info

    if not MODEL_REPO or "YOUR_USERNAME" in MODEL_REPO:
        return (
            "Set **HF_MODEL_REPO** in Space Settings → Repository variables "
            f"(e.g. `luayas1977/arabicai-tinystories-gpt`). Current value: `{MODEL_REPO}`"
        )

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = hf_hub_download(repo_id=MODEL_REPO, filename="model.pt")
    tok_path = hf_hub_download(repo_id=MODEL_REPO, filename="tinystories_tokenizer.json")

    ckpt = torch.load(ckpt_path, map_location=_device, weights_only=False)
    _model = GPT(ckpt["config"]).to(_device)
    _model.load_state_dict(ckpt["model"])
    _model.eval()

    _tokenizer = Tokenizer.from_file(tok_path)
    _eos_id = _tokenizer.token_to_id("<|endoftext|>")

    _checkpoint_info = {
        "step": ckpt.get("step", 0),
        "val_loss": ckpt.get("val_loss", 0.0),
    }

    return (
        f"Loaded **{MODEL_REPO}** on `{_device}` — "
        f"step {_checkpoint_info['step']:,}, "
        f"val loss {_checkpoint_info['val_loss']:.4f}"
    )


def generate_story(
    prompt: str,
    temperature: float,
    top_k: int,
    max_tokens: int,
    stop_at_eos: bool,
) -> str:
    if _model is None or _tokenizer is None:
        return "Model not loaded. Check HF_MODEL_REPO and Space logs."

    prompt = (prompt or "Once upon a time").strip()
    top_k_val = int(top_k) if top_k and int(top_k) > 0 else None
    eos_id = _eos_id if stop_at_eos else None

    prompt_ids = _tokenizer.encode(prompt).ids
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=_device)

    with torch.no_grad():
        out = _model.generate(
            idx,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_k=top_k_val,
            eos_token_id=eos_id,
        )

    full_ids = out[0].tolist()
    if eos_id is not None and full_ids and full_ids[-1] == eos_id:
        full_ids = full_ids[:-1]

    prompt_text = _tokenizer.decode(prompt_ids)
    generated_only = _tokenizer.decode(full_ids[len(prompt_ids) :])

    if stop_at_eos:
        generated_only, _ = truncate_at_story_restart(generated_only)

    return prompt_text + generated_only


# ── Gradio UI ─────────────────────────────────────────────────────────────────

load_status = load_model()

with gr.Blocks(title="Arabic AI TinyStories GPT") as demo:
    gr.Markdown(
        "# Arabic AI TinyStories GPT · Playground\n"
        "12M-parameter GPT trained from scratch on TinyStories. "
        "Part of *Build a GPT From Scratch on a 6 GB GPU*."
    )
    gr.Markdown(load_status)

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                value="Once upon a time",
                lines=2,
            )
            temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
            top_k = gr.Slider(0, 200, value=40, step=1, label="Top-K (0 = off)")
            max_tokens = gr.Slider(20, 500, value=200, step=10, label="Max tokens")
            stop_at_eos = gr.Checkbox(value=True, label="Stop at story end")
            btn = gr.Button("Generate story", variant="primary")
        with gr.Column(scale=2):
            output = gr.Textbox(label="Generated story", lines=16)

    examples = gr.Examples(
        examples=[
            ["Once upon a time", 0.8, 40, 200, True],
            ["Lily found a", 0.8, 40, 200, True],
            ["The dragon was", 1.0, 40, 150, True],
        ],
        inputs=[prompt, temperature, top_k, max_tokens, stop_at_eos],
    )

    btn.click(
        generate_story,
        inputs=[prompt, temperature, top_k, max_tokens, stop_at_eos],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()
