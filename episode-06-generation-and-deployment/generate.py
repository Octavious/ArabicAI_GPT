"""
═══════════════════════════════════════════════════════════════════════════════
 generate.py — Generate Text From Your Trained Model
 Built in Episode 05/06 of "Build a GPT From Scratch on a 6 GB GPU"
 YouTube: https://www.youtube.com/playlist?list=PLvmpljk9TE2v4LbSrQxmnbOKkLa0Mw1oD
═══════════════════════════════════════════════════════════════════════════════

 The simplest possible generator. Loads checkpoint.pt, takes a prompt, prints
 a generated story.

 What you need in the same folder:
   - model.py                       (the architecture, from Episode 4)
   - tinystories_tokenizer.json     (the tokenizer, from Episode 2)
   - checkpoint.pt                  (the trained weights, from Episode 5)

 Usage:
   python generate.py
   python generate.py "Lily found a"
   python generate.py "Once upon a time" --tokens 200 --temperature 0.8
"""

import argparse
import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_PATH = "checkpoint.pt"
TOKENIZER_PATH  = "tinystories_tokenizer.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ─────────────────────────────────────────────────────────────────────────
    # Parse command-line arguments
    # ─────────────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Generate a story from the trained model.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Once upon a time",
        help="The starting text for the model to continue."
    )
    parser.add_argument(
        "--tokens", type=int, default=200,
        help="How many new tokens to generate (default 200, ~50-100 words)."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature. 1.0=neutral, <1.0=safer, >1.0=more creative."
    )
    parser.add_argument(
        "--top-k", type=int, default=40,
        help="Only sample from the top-K most likely tokens at each step."
    )
    args = parser.parse_args()

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 — Load the tokenizer
    # ─────────────────────────────────────────────────────────────────────────
    # Same tokenizer file we used during training. Must match — if you trained
    # with one tokenizer and generate with another, every token ID means
    # something different and you'll get garbage.
    print(f"Loading tokenizer from {TOKENIZER_PATH} ...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 — Load the trained model from checkpoint.pt
    # ─────────────────────────────────────────────────────────────────────────
    # The checkpoint is a dictionary containing:
    #   'model'     — the trained weights (state_dict)
    #   'config'    — the GPTConfig used during training
    #   'step'      — how many training steps completed
    #   'val_loss'  — the validation loss at that point
    #   'optimizer' — the optimizer state (we don't need this for generation)
    print(f"Loading checkpoint from {CHECKPOINT_PATH} ...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

    # Build the model with the SAME config that was used during training.
    # If the config differs, the weight shapes won't match.
    config = checkpoint['config']
    model = GPT(config)

    # Load the trained weights into the model
    model.load_state_dict(checkpoint['model'])

    # Move to GPU (or CPU) and switch to evaluation mode.
    # eval() disables dropout — important for clean generation.
    model.to(DEVICE)
    model.eval()

    # Report what we loaded
    print(f"  trained for {checkpoint['step']:,} steps")
    print(f"  val loss:   {checkpoint['val_loss']:.4f}")
    print(f"  device:     {DEVICE}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 — Encode the prompt
    # ─────────────────────────────────────────────────────────────────────────
    # Turn the prompt text into a tensor of token IDs.
    # The shape needs an extra batch dimension: (1, T)
    prompt_ids = tokenizer.encode(args.prompt).ids
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 — Generate new tokens
    # ─────────────────────────────────────────────────────────────────────────
    # The model.generate() method runs the autoregressive loop:
    # for each step, run the model forward, sample one token from the
    # output distribution, append it, repeat.
    print(f"\nGenerating {args.tokens} tokens "
          f"(temperature={args.temperature}, top_k={args.top_k}) ...\n")
    print("=" * 60)

    with torch.no_grad():
        output_ids = model.generate(
            idx,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5 — Decode the output back to text
    # ─────────────────────────────────────────────────────────────────────────
    # output_ids has shape (1, prompt_len + tokens). Take the first row,
    # convert to a Python list of ints, then decode.
    output_text = tokenizer.decode(output_ids[0].tolist())

    print(output_text)
    print("=" * 60)


if __name__ == '__main__':
    main()
