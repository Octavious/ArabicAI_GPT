"""
═══════════════════════════════════════════════════════════════════════════════
 train.py — Training the Model
 Built in Episode 05 of "Build a GPT From Scratch on a 6 GB GPU"
 YouTube: https://www.youtube.com/playlist?list=PLvmpljk9TE2v4LbSrQxmnbOKkLa0Mw1oD
═══════════════════════════════════════════════════════════════════════════════

 Episode 4 built the model. It works, but it knows nothing — its 12 million
 parameters are random noise, and its loss is ~8.3 (as confused as possible).

 This file teaches it. The core is a loop that repeats ~25,000 times:

   1. get a batch of real text from train.bin
   2. forward pass  — the model predicts the next token at every position
   3. compute loss  — how wrong were those predictions?
   4. backward pass — compute how each parameter contributed to the error
   5. update        — nudge every parameter slightly toward "less wrong"
   6. repeat

 Around that loop sits the scaffolding: a learning rate schedule, periodic
 validation checks, and checkpoint saving.

 Expected outcome: loss drops from ~8.3 to ~1.7 over about 2 hours on a
 GTX 1660 SUPER. Gibberish becomes coherent children's stories.
"""

import os
import time
import math
import torch

from model import GPT, GPTConfig
from data_loader import DataLoader


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING CONFIGURATION
# Every knob you can turn, in one place.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR     = "."                  # folder with train.bin and validation.bin
CHECKPOINT   = "checkpoint.pt"       # where to save model snapshots

# ── How long to train ────────────────────────────────────────────────────────
MAX_STEPS    = 25000                # total training steps
BATCH_SIZE   = 32                   # sequences per batch
BLOCK_SIZE   = 256                  # tokens per sequence (must match the model)

# ── Optimizer settings ───────────────────────────────────────────────────────
LEARNING_RATE = 3e-4                # the peak learning rate
WEIGHT_DECAY  = 0.1                 # L2-style regularization (AdamW)
BETA1         = 0.9                 # AdamW momentum term
BETA2         = 0.95                # AdamW variance term
GRAD_CLIP     = 1.0                 # clip gradients to this max norm

# ── Learning rate schedule ───────────────────────────────────────────────────
WARMUP_STEPS  = 1000                # ramp LR up from 0 over these steps
MIN_LR        = 3e-5                # LR never decays below this (10% of peak)

# ── How often to check in ────────────────────────────────────────────────────
EVAL_INTERVAL = 500                 # run validation + log a status line every N steps
EVAL_BATCHES  = 50                  # average over this many batches when evaluating

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULE
# Warmup, then cosine decay. The LR is not constant during training.
# ═══════════════════════════════════════════════════════════════════════════════

def get_learning_rate(step):
    """
    Compute the learning rate for a given step.

    The schedule has two phases:

      Phase 1 — Warmup (steps 0 to WARMUP_STEPS):
        LR ramps linearly from 0 up to the peak LEARNING_RATE.
        Why? At the start, the model's parameters are random. Large updates
        on random parameters can destabilize training. A gentle warmup lets
        the model find its footing first.

      Phase 2 — Cosine decay (WARMUP_STEPS to MAX_STEPS):
        LR follows a cosine curve down from LEARNING_RATE to MIN_LR.
        Why? Early in training, big steps make fast progress. Later, the model
        is near a good solution and needs small, careful steps to settle in.
        The cosine shape gives a smooth transition between the two.

    The result looks like:  /\\___
                            up   slow decline
    """
    # Phase 1 — linear warmup
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS

    # Past the end of training — just hold at the minimum
    if step > MAX_STEPS:
        return MIN_LR

    # Phase 2 — cosine decay
    # `progress` goes from 0.0 (start of decay) to 1.0 (end of training)
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    # cosine_factor goes smoothly from 1.0 down to 0.0
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    # interpolate between MIN_LR and LEARNING_RATE using that factor
    return MIN_LR + cosine_factor * (LEARNING_RATE - MIN_LR)


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# Measure loss on data the model never trains on.
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def estimate_loss(model, loader):
    """
    Run the model on several batches from BOTH splits and average the loss.

    Why both splits?
      - train loss tells us how well the model fits the data it sees
      - val loss tells us how well it GENERALIZES to data it doesn't

    If train loss keeps dropping but val loss stops (or rises), the model is
    overfitting — memorizing the training data instead of learning patterns.

    @torch.no_grad() disables gradient tracking — we're only measuring here,
    not learning, so we don't need the backward-pass machinery. It's faster
    and uses less memory.
    """
    model.eval()   # switch to evaluation mode (disables dropout)

    results = {}
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_BATCHES)
        for i in range(EVAL_BATCHES):
            x, y = loader.get_batch(split)
            _, loss = model(x, y)
            losses[i] = loss.item()
        results[split] = losses.mean().item()

    model.train()  # switch back to training mode (re-enables dropout)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# THE MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(" Training a GPT from scratch")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # ── Set up data ───────────────────────────────────────────────────────────
    print("\nLoading data...")
    loader = DataLoader(
        data_dir=DATA_DIR,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    # ── Build the model ───────────────────────────────────────────────────────
    print("\nBuilding model...")
    config = GPTConfig()
    model = GPT(config)
    model.to(DEVICE)
    print(f"  Parameters: {model.get_num_params():,}")

    # ── Create the optimizer ──────────────────────────────────────────────────
    # AdamW is the standard optimizer for transformers. It does two things:
    #   - Adam part: keeps a running average of gradients (momentum) and
    #     their variance, so each parameter gets its own adaptive step size
    #   - W part: applies weight decay correctly (pulls weights toward zero,
    #     a form of regularization that reduces overfitting)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,            # this gets overridden each step by our schedule
        weight_decay=WEIGHT_DECAY,
        betas=(BETA1, BETA2),
    )

    # ── Training state ────────────────────────────────────────────────────────
    best_val_loss = float('inf')     # track the best validation loss so far
    start_time = time.time()

    # Per-step timing: we accumulate step durations between eval intervals
    # so we can report the average ms/step on each log line.
    interval_start = time.time()

    print(f"\nStarting training for {MAX_STEPS:,} steps...")
    print("-" * 80)

    # ─────────────────────────────────────────────────────────────────────────
    # THE LOOP
    # ─────────────────────────────────────────────────────────────────────────
    for step in range(MAX_STEPS):

        # ── Set the learning rate for this step ───────────────────────────────
        # We compute it fresh every step from our warmup+cosine schedule,
        # then write it into the optimizer.
        lr = get_learning_rate(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ── Step 1: Get a batch of training data ──────────────────────────────
        x, y = loader.get_batch('train')

        # ── Step 2: Forward pass ──────────────────────────────────────────────
        # The model predicts the next token at every position, and because we
        # passed `y`, it also computes the cross-entropy loss for us.
        logits, loss = model(x, y)

        # ── Step 3: Backward pass ─────────────────────────────────────────────
        # First clear any gradients left over from the previous step.
        # (PyTorch accumulates gradients by default — we don't want that here.)
        optimizer.zero_grad(set_to_none=True)

        # Then compute new gradients: for every parameter, how much did it
        # contribute to the loss? This is backpropagation.
        loss.backward()

        # ── Step 4: Clip gradients ────────────────────────────────────────────
        # Occasionally a bad batch produces huge gradients that would blow the
        # model's weights out of a good region. Clipping caps the total
        # gradient size at GRAD_CLIP, keeping training stable.
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # ── Step 5: Update the weights ────────────────────────────────────────
        # The optimizer nudges every parameter a small step in the direction
        # that reduces the loss. This is the actual "learning."
        optimizer.step()

        # ── Evaluation + logging: combined into one line per eval interval ────
        # Every EVAL_INTERVAL steps we run validation and print a status line.
        # The format keeps everything we want to track on a single row:
        #
        #   step  500 | train 5.1030 | val 3.7570 | lr 3.00e-04 | 380.6ms/step
        #
        # Per-step training loss is intentionally NOT printed every step — it's
        # noisy (each batch is different), and the EVAL_BATCHES-averaged
        # train loss in this line is far more meaningful.
        if (step + 1) % EVAL_INTERVAL == 0:
            # Compute average time per step over this interval
            interval_elapsed = time.time() - interval_start
            ms_per_step = (interval_elapsed / EVAL_INTERVAL) * 1000

            # Run validation
            losses = estimate_loss(model, loader)

            # One clean status line — everything that matters on one row
            print(f"step {step + 1:>6d} | "
                  f"train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | "
                  f"lr {lr:.2e} | "
                  f"{ms_per_step:.1f}ms/step")

            # ── Checkpointing: save the model if it's the best so far ──────────
            # We only save when validation loss improves. This way, if training
            # later goes bad, our checkpoint is still the best version.
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'step': step + 1,
                    'val_loss': best_val_loss,
                }
                torch.save(checkpoint, CHECKPOINT)
                print(f"       └─ saved checkpoint (val loss {best_val_loss:.4f})")

            # Reset the interval timer for the next window
            interval_start = time.time()

    # ── Done ──────────────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    print("=" * 60)
    print(f" Training complete.")
    print(f"   Total time:      {total_time/60:.1f} minutes")
    print(f"   Best val loss:   {best_val_loss:.4f}")
    print(f"   Checkpoint:      {CHECKPOINT}")
    print("=" * 60)


if __name__ == '__main__':
    main()
