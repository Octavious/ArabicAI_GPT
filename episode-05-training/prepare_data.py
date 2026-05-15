"""
═══════════════════════════════════════════════════════════════════════════════
 prepare_data.py — Converting the Dataset to Binary
 Built in Episode 05 of "Build a GPT From Scratch on a 6 GB GPU"
 YouTube: https://www.youtube.com/playlist?list=PLvmpljk9TE2v4LbSrQxmnbOKkLa0Mw1oD
═══════════════════════════════════════════════════════════════════════════════

 In Episode 2 we built and trained the BPE tokenizer (tinystories_tokenizer.json).
 This script is the step that actually USES it: we run every story in the
 TinyStories dataset through the tokenizer and save the resulting token IDs
 as two flat binary files.

   train.bin       — token IDs for the training split
   validation.bin  — token IDs for the held-out validation split

 Why save as raw binary instead of keeping the text?

   - Plain text would need re-tokenizing every single training run — slow
     and wasteful, since tokenization never changes.
   - A Python list or pickle would have to be fully loaded into RAM —
     485 million numbers won't fit comfortably.
   - A flat binary file of fixed-size integers (uint16, 2 bytes each) can
     be MEMORY-MAPPED. The training data loader can jump straight to any
     position and read a small slice without touching the rest of the file.

 That last property is exactly what makes the random-window sampling in
 data_loader.py possible. We choose the format here with that use in mind.

 This script only needs to be run ONCE. After it finishes, train.py reads
 the .bin files directly.

 Expected runtime: a few minutes (it's mostly tokenization, single-threaded).
"""

import os
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# The tokenizer we trained in Episode 2
TOKENIZER_PATH = "tinystories_tokenizer.json"

# Where to write the output files
TRAIN_BIN = "train.bin"
VAL_BIN   = "validation.bin"

# The Hugging Face dataset identifier
DATASET_NAME = "roneneldan/TinyStories"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(" Preparing TinyStories data")
    print("=" * 60)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 — Load the tokenizer
    # ─────────────────────────────────────────────────────────────────────────
    # This is the exact tokenizer file we produced in Episode 2.
    # It maps text → integer token IDs using the BPE merges it learned.
    print(f"\nLoading tokenizer from {TOKENIZER_PATH} ...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"  Tokenizer vocabulary size: {vocab_size:,}")

    # Safety check — our model and data loader assume uint16 storage.
    # uint16 can hold values 0..65,535. As long as the vocabulary is
    # smaller than that, every token ID fits in 2 bytes.
    assert vocab_size < 65536, \
        "Vocabulary too large for uint16 storage — would need uint32."

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 — Load the dataset
    # ─────────────────────────────────────────────────────────────────────────
    # load_dataset pulls TinyStories from the Hugging Face Hub (or from the
    # local cache if you've downloaded it before). It comes already split
    # into 'train' and 'validation'.
    print(f"\nLoading dataset '{DATASET_NAME}' ...")
    dataset = load_dataset(DATASET_NAME)
    print(f"  train split:      {len(dataset['train']):,} stories")
    print(f"  validation split: {len(dataset['validation']):,} stories")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 — Tokenize each split and write it to a .bin file
    # ─────────────────────────────────────────────────────────────────────────
    # We process the 'train' and 'validation' splits with the same logic.
    for split_name, out_path in [('train', TRAIN_BIN), ('validation', VAL_BIN)]:
        print(f"\nTokenizing '{split_name}' split → {out_path}")

        split = dataset[split_name]

        # We collect all token IDs from all stories into one big list,
        # then write the whole thing out as a single flat array.
        #
        # Why one flat array, with all stories concatenated end-to-end?
        # During training we pull random 256-token windows. A window might
        # span the boundary between two stories — and that's fine. The model
        # learns "this story ended, a new one began" as just another pattern.
        # Treating the whole split as one long stream keeps the data loader
        # dead simple.
        all_ids = []

        # tqdm gives us a progress bar — tokenizing 2M stories takes a few minutes
        for story in tqdm(split, desc=f"  {split_name}"):
            text = story['text']

            # Run the text through the tokenizer → list of integer IDs
            encoded = tokenizer.encode(text)
            ids = encoded.ids

            # Append this story's tokens to the running list
            all_ids.extend(ids)

        # ── Convert the Python list to a numpy array of uint16 ────────────────
        # uint16 = 2 bytes per token. This matches what data_loader.py expects
        # when it memory-maps the file back in.
        arr = np.array(all_ids, dtype=np.uint16)

        # ── Write the array to disk as raw bytes ──────────────────────────────
        # .tofile() dumps the array's raw bytes — no headers, no metadata,
        # just a flat sequence of uint16 values. That's the simplest possible
        # format, and it's exactly what np.memmap reads back.
        arr.tofile(out_path)

        # ── Report ────────────────────────────────────────────────────────────
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  wrote {len(arr):,} tokens  ({size_mb:.1f} MB)")

    # ─────────────────────────────────────────────────────────────────────────
    # Done
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" Data preparation complete.")
    print(f"   {TRAIN_BIN} and {VAL_BIN} are ready.")
    print("   You can now run train.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
