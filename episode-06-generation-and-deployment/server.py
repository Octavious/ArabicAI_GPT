"""
═══════════════════════════════════════════════════════════════════════════════
 server.py — Tiny Flask Server Wrapping the Trained GPT Model
 Built in Episode 06 of "Build a GPT From Scratch on a 6 GB GPU"
 YouTube: https://www.youtube.com/playlist?list=PLvmpljk9TE2v4LbSrQxmnbOKkLa0Mw1oD
═══════════════════════════════════════════════════════════════════════════════

 What this is:
 A small web server that loads checkpoint.pt and exposes a /generate endpoint.
 Browser opens generate_ui.html, types a prompt, gets a generated story back.

 What you need in the same folder:
   - model.py                       (architecture, from Episode 4)
   - tinystories_tokenizer.json     (tokenizer, from Episode 2)
   - checkpoint.pt                  (trained weights, from Episode 5)
   - generate_ui.html               (the web UI, built in this episode)

 Run it:
   python server.py

 Then open http://localhost:5000 in your browser.
"""

import os
import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tokenizers import Tokenizer

from model import GPT
from inference_utils import truncate_at_story_restart


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_PATH = "checkpoint.pt"
TOKENIZER_PATH  = "tinystories_tokenizer.json"
HOST = "127.0.0.1"
PORT = 5000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)   # allow the browser to call us even if the HTML is opened separately

# Global state — model and tokenizer stay loaded in memory across requests
_model = None
_tokenizer = None
_checkpoint_info = None      # {'step': ..., 'val_loss': ...} for the loaded ckpt
_eos_id = None               # the <|endoftext|> token ID (cached at startup)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_model():
    """Load checkpoint.pt and prepare the model for inference."""
    global _model, _checkpoint_info

    print(f"Loading checkpoint from {CHECKPOINT_PATH} ...")

    # weights_only=False because we save the config dataclass alongside the
    # weights — that's safe here since we trust our own checkpoint file.
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

    # Rebuild the model using the same config that was used to train it
    config = ckpt['config']
    model = GPT(config).to(DEVICE)

    # Load the trained weights. The key is 'model' to match train.py's save format.
    model.load_state_dict(ckpt['model'])

    # eval() disables dropout — important for clean generation
    model.eval()

    _model = model
    _checkpoint_info = {
        'step':     ckpt.get('step',     0),
        'val_loss': ckpt.get('val_loss', 0.0),
    }
    print(f"  step:     {_checkpoint_info['step']:,}")
    print(f"  val_loss: {_checkpoint_info['val_loss']:.4f}")
    print(f"  device:   {DEVICE}")


def load_tokenizer():
    """Load the BPE tokenizer and cache the EOS token ID."""
    global _tokenizer, _eos_id

    print(f"Loading tokenizer from {TOKENIZER_PATH} ...")
    _tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    # Cache the end-of-story token. In our tokenizer, <|endoftext|> = ID 0.
    # The model will use this to know when to stop generating.
    _eos_id = _tokenizer.token_to_id("<|endoftext|>")
    print(f"  vocab size:    {_tokenizer.get_vocab_size():,}")
    print(f"  <|endoftext|>: ID {_eos_id}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def serve_ui():
    """Serve generate_ui.html when the user opens the server root."""
    return send_from_directory(".", "generate_ui.html")


@app.route("/info", methods=["GET"])
def info():
    """Return basic info about the loaded model — used by the UI on startup."""
    return jsonify({
        'step':       _checkpoint_info['step'],
        'val_loss':   _checkpoint_info['val_loss'],
        'device':     DEVICE,
        'vocab_size': _tokenizer.get_vocab_size(),
        'has_eos':    _eos_id is not None,
    })


@app.route("/generate", methods=["POST"])
def generate():
    """
    Generate text from a prompt and return it.

    Expected JSON body:
        {
            "prompt":         "Once upon a time",
            "max_new_tokens": 200,        (optional, default 200)
            "temperature":    0.8,        (optional, default 0.8)
            "top_k":          40,         (optional, default 40, null=disabled)
            "stop_at_eos":    true        (optional, default true)
        }
    """
    data = request.get_json(force=True)

    # Parse and validate inputs with sensible defaults
    prompt         = data.get('prompt', 'Once upon a time')
    max_new_tokens = int(data.get('max_new_tokens', 200))
    temperature    = float(data.get('temperature', 0.8))
    top_k          = data.get('top_k', 40)
    top_k          = int(top_k) if top_k and int(top_k) > 0 else None
    stop_at_eos    = data.get('stop_at_eos', True)

    # Look up the EOS token ID if the caller asked to stop at end-of-story
    eos_id = _eos_id if stop_at_eos else None

    # Encode the prompt to token IDs and add a batch dimension: (1, T)
    prompt_ids = _tokenizer.encode(prompt).ids
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    # Run the autoregressive generation loop
    with torch.no_grad():
        output_ids = _model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_id,
        )

    # Convert the output tensor back to a Python list
    full_ids = output_ids[0].tolist()

    # If generation stopped at EOS, strip the trailing EOS token —
    # it's an invisible control marker, no point showing it
    stopped_early = False
    if eos_id is not None and full_ids and full_ids[-1] == eos_id:
        full_ids = full_ids[:-1]
        stopped_early = True

    # Decode three useful pieces and return them all
    prompt_text    = _tokenizer.decode(prompt_ids)
    generated_only = _tokenizer.decode(full_ids[len(prompt_ids):])

    # Training data does not insert <|endoftext|> between stories, so the model
    # rarely samples EOS. Fall back to cutting before a repeated story opener.
    if stop_at_eos:
        generated_only, heuristic_stop = truncate_at_story_restart(generated_only)
        stopped_early = stopped_early or heuristic_stop

    full_text = prompt_text + generated_only

    return jsonify({
        'prompt':         prompt_text,
        'generated':      generated_only,
        'full_text':      full_text,
        'num_tokens':     len(full_ids) - len(prompt_ids),
        'stopped_early':  stopped_early,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(" GPT Inference Server")
    print("=" * 60)
    print(f"Device: {DEVICE}\n")

    # Verify required files exist before we try to load them
    for path in (CHECKPOINT_PATH, TOKENIZER_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                f"Make sure model.py, tinystories_tokenizer.json, and "
                f"checkpoint.pt are all in the current directory."
            )

    load_tokenizer()
    load_model()

    print(f"\nServer ready. Open http://{HOST}:{PORT} in your browser.\n")
    app.run(host=HOST, port=PORT, debug=False)


if __name__ == "__main__":
    main()
