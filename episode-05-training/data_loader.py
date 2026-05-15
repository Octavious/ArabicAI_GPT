"""
═══════════════════════════════════════════════════════════════════════════════
 data_loader.py — Feeding Data to the Model
 Built in Episode 05 of "Build a GPT From Scratch on a 6 GB GPU"
 YouTube: https://www.youtube.com/playlist?list=PLvmpljk9TE2v4LbSrQxmnbOKkLa0Mw1oD
═══════════════════════════════════════════════════════════════════════════════

 This file does one job: pull a random batch of training sequences out of the
 .bin files we created in Episode 2.

 A reminder of what we have on disk after Episode 2:
   - train.bin       — 485 million token IDs, stored as raw uint16
   - validation.bin  — a smaller held-out set the model never trains on

 The model can't read a 485-million-token file all at once — it would never
 fit in 6 GB of VRAM. Instead, every training step we grab a small random
 window: 32 sequences of 256 tokens each. That's one "batch."

 The key trick here is memory-mapping (np.memmap). Instead of loading the
 whole .bin file into RAM, we map it — the operating system pages in only
 the bytes we actually touch. This lets us treat a multi-gigabyte file as
 if it were a normal array, while using almost no memory.
"""

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════════════
# THE DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class DataLoader:
    """
    Loads random batches from the tokenized .bin files.

    One DataLoader instance handles both splits (train and validation).
    Call get_batch('train') or get_batch('val') each step.
    """

    def __init__(self, data_dir, block_size, batch_size, device):
        """
        Args:
            data_dir:   folder containing train.bin and validation.bin
            block_size: sequence length — how many tokens per sample (256 for us)
            batch_size: how many sequences per batch (32 for us)
            device:     'cuda' or 'cpu' — where to put the tensors
        """
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        # ── Memory-map both files ─────────────────────────────────────────────
        # np.memmap opens the file WITHOUT loading it into RAM.
        # The dtype must match exactly what we wrote in Episode 2: uint16.
        # (uint16 holds values 0..65535 — plenty for our 4,096-token vocab.)
        #
        # We re-create these memmaps fresh on every get_batch() call below,
        # not here. Why? A known memory-leak quirk: keeping one long-lived
        # memmap and reading from it repeatedly slowly grows memory usage.
        # Re-opening it each batch avoids that. We store only the paths here.
        self.train_path = f"{data_dir}/train.bin"
        self.val_path = f"{data_dir}/validation.bin"

        # Quick sanity check — report how many tokens each split has
        train_data = np.memmap(self.train_path, dtype=np.uint16, mode='r')
        val_data = np.memmap(self.val_path, dtype=np.uint16, mode='r')
        print(f"  train.bin:      {len(train_data):,} tokens")
        print(f"  validation.bin: {len(val_data):,} tokens")
        # Let these go out of scope — we'll re-open per batch.
        del train_data, val_data

    def get_batch(self, split):
        """
        Pull one random batch from the requested split.

        Args:
            split: 'train' or 'val'

        Returns:
            x: input  token IDs, shape (batch_size, block_size)
            y: target token IDs, shape (batch_size, block_size)
               y is x shifted one position to the left — see below.
        """
        # ── Step 1: Re-open the memory-map ────────────────────────────────────
        # Fresh memmap each call (see the note in __init__ about why).
        path = self.train_path if split == 'train' else self.val_path
        data = np.memmap(path, dtype=np.uint16, mode='r')

        # ── Step 2: Pick random starting positions ────────────────────────────
        # We need `batch_size` random windows. Each window is `block_size`
        # tokens long, plus one extra token for the target shift.
        #
        # The highest valid starting index is len(data) - block_size - 1,
        # so that even the last window doesn't run off the end of the file.
        max_start = len(data) - self.block_size - 1
        starts = torch.randint(low=0, high=max_start, size=(self.batch_size,))

        # ── Step 3: Build the input batch (x) and target batch (y) ────────────
        # For each random start position `i`:
        #   x  =  data[i      : i + block_size]       the input tokens
        #   y  =  data[i + 1  : i + block_size + 1]   the SAME tokens, shifted by 1
        #
        # Why is y just x shifted by one?
        # The model's job is "predict the next token." So for every position
        # in x, the correct answer is simply the token that comes right after.
        # Shifting x left by one position gives us exactly those answers.
        #
        # Example with a tiny sequence:
        #   data    = [ The , cat , sat , on , the , mat ]
        #   x       = [ The , cat , sat , on ]
        #   y       = [ cat , sat , on  , the ]
        #   meaning:  "after The → cat", "after cat → sat", etc.
        #
        # We convert each numpy slice to a torch tensor (int64, which is what
        # the embedding layer expects).
        x = torch.stack([
            torch.from_numpy(data[i : i + self.block_size].astype(np.int64))
            for i in starts
        ])
        y = torch.stack([
            torch.from_numpy(data[i + 1 : i + 1 + self.block_size].astype(np.int64))
            for i in starts
        ])

        # ── Step 4: Move to the GPU ───────────────────────────────────────────
        # If we're training on CUDA, ship the batch to the GPU.
        #
        # pin_memory + non_blocking is a small optimization: it lets the data
        # transfer overlap with computation. On CPU it's a harmless no-op path.
        if self.device == 'cuda':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)

        return x, y


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# Run this file directly to verify the .bin files load and batches come out
# the right shape.
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Adjust this path to wherever your .bin files actually live
    DATA_DIR = "."

    print("Loading data...")
    loader = DataLoader(
        data_dir=DATA_DIR,
        block_size=256,
        batch_size=32,
        device='cpu',   # use 'cpu' for this quick test
    )

    # Pull one batch from each split
    x, y = loader.get_batch('train')
    print(f"\nTrain batch:")
    print(f"  x shape: {x.shape}   (expect [32, 256])")
    print(f"  y shape: {y.shape}   (expect [32, 256])")
    print(f"  x dtype: {x.dtype}   (expect int64)")

    # Verify the shift relationship: y[0][:-1] should equal x[0][1:]
    shift_ok = torch.equal(x[0, 1:], y[0, :-1])
    print(f"\n  y is x shifted by 1: {shift_ok}   (expect True)")

    xv, yv = loader.get_batch('val')
    print(f"\nValidation batch:")
    print(f"  x shape: {xv.shape}   (expect [32, 256])")

    print("\nData loader test passed." if shift_ok else "\nWARNING: shift check failed.")
