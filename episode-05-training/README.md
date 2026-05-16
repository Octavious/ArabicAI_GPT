# Episode 05 — Training the Model

The fifth episode in the [Build a GPT From Scratch on a 6 GB GPU](../README.md) series.

**▶ [Watch the video](https://www.youtube.com/playlist?list=PLvmpljk9TE2v4LbSrQxmnbOKkLa0Mw1oD)**

---

## What This Episode Is

Episode 4 built the model. It works — but it knows nothing. Its 12 million parameters are random noise, and its loss sits at ~8.3, meaning it's as confused as it's possible to be: every one of the 4,096 vocabulary tokens looks equally likely as the next word.

This episode teaches it. By the end, the loss has dropped to ~1.7 and the same model — same architecture, same parameter count — writes coherent children's stories.

The mechanism is a loop that repeats about 25,000 times:

```
get a batch → forward pass → compute loss → backward pass → update weights → repeat
```

Everything else in this episode is scaffolding around that loop.

---

## Prerequisites

- [Episode 04](../episode-04-building-the-model/) finished — you need `model.py`
- [Episode 02](../episode-02-data-and-tokenizer/) finished — you need `tinystories_tokenizer.json`
- The TinyStories dataset accessible via Hugging Face `datasets` (the script downloads it)

Note: in Episode 2 we built and trained the tokenizer, but we didn't yet run the dataset through it. That conversion happens here, as the opening step of this episode — see `prepare_data.py` below. After that step finishes, you'll have `train.bin` and `validation.bin`, and everything else in this episode reads those.

All the files (`model.py`, `tinystories_tokenizer.json`, and this episode's three files) should sit in the same folder.

---

## Files in This Episode

| File | What it is |
|---|---|
| `prepare_data.py` | Runs the Episode 2 tokenizer over TinyStories, writes `train.bin` and `validation.bin` |
| `data_loader.py` | Pulls random batches from the `.bin` files |
| `train.py` | The full training script — the loop, the optimizer, the schedule, checkpointing |
| `episode_05_slides.html` | Slide deck used during the recording |

The episode flows in that order: **prepare → load → train.**

---

## A Note on Recording This Episode

Training takes about 2 hours. You can't film 2 hours of a scrolling progress bar.

The standard approach: **run the training first**, capture the terminal logs and the saved checkpoints, then record the episode around that captured material. Film yourself writing and explaining `train.py` and `data_loader.py` live, start the run on camera, then cut to pre-recorded time-lapse footage and the loss curves.

The scrolling loss numbers and the periodic eval lines are exactly the footage Episode 6 ("Watching It Learn") will need — so capture generously.

---

## Python Dependencies

This episode introduces three new Python packages on top of `torch` from Episode 1:

```powershell
uv pip install datasets tokenizers tqdm
```

| Package | Used in | What for |
|---|---|---|
| `datasets` | `prepare_data.py` | Pulls TinyStories from the Hugging Face Hub via `load_dataset` |
| `tokenizers` | `prepare_data.py` | Loads and applies the BPE tokenizer from Episode 2 |
| `tqdm` | `prepare_data.py` | Progress bar during tokenization (~2 million stories) |

Only `prepare_data.py` uses these. Once the `.bin` files exist, `data_loader.py` and `train.py` only need `torch` and `numpy`.

---

## prepare_data.py — Converting the Dataset to Binary

This is the opening step of the episode, and it only needs to run once.

In Episode 2 we built and trained the BPE tokenizer. `prepare_data.py` is the step that actually *uses* it: it runs every story in TinyStories through the tokenizer and saves the resulting token IDs as two flat binary files — `train.bin` and `validation.bin`.

### What the script does

```python
# 1. Load the Episode 2 tokenizer
tokenizer = Tokenizer.from_file("tinystories_tokenizer.json")

# 2. Load the dataset from Hugging Face
dataset = load_dataset("roneneldan/TinyStories")

# 3. For each split: tokenize every story, concatenate, write as uint16
for split_name, out_path in [('train', 'train.bin'), ('validation', 'validation.bin')]:
    all_ids = []
    for story in dataset[split_name]:
        all_ids.extend(tokenizer.encode(story['text']).ids)
    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(out_path)
```

### Why save as raw binary?

| Format | Problem |
|---|---|
| Plain text | Would need re-tokenizing every training run — slow and wasteful |
| Python list / pickle | Must load all 485M numbers into RAM — won't fit comfortably |
| **Raw `.bin` (uint16)** | **Memory-mappable — read tiny random slices without loading the file** |

That last property is exactly what makes the random-window sampling in `data_loader.py` possible. Because the data is a flat array of fixed-size integers on disk, `np.memmap` can jump straight to any position and read 256 tokens without touching the rest of the file. The format is chosen here with that exact use in mind.

### Why concatenate all stories into one stream

The script collects all token IDs from all stories into one big flat array, end to end. During training, a random 256-token window might span the boundary between two stories — and that's fine. The model learns "this story ended, a new one began" as just another pattern. Treating the whole split as one long stream keeps the data loader simple.

### Running it

```powershell
python prepare_data.py
```

Expected output:

```
============================================================
 Preparing TinyStories data
============================================================

Loading tokenizer from tinystories_tokenizer.json ...
  Tokenizer vocabulary size: 4,096

Loading dataset 'roneneldan/TinyStories' ...
  train split:      2,119,719 stories
  validation split: 21,990 stories

Tokenizing 'train' split → train.bin
  train: 100%|████████████████| 2119719/2119719
  wrote 485,XXX,XXX tokens  (925.X MB)

Tokenizing 'validation' split → validation.bin
  validation: 100%|███████████| 21990/21990
  wrote 4,XXX,XXX tokens  (9.X MB)

============================================================
 Data preparation complete.
============================================================
```

Runtime is a few minutes — it's mostly single-threaded tokenization. Once it finishes, you never need to run it again; `train.py` reads the `.bin` files directly.

---



## data_loader.py — Feeding the Model

The model can't read a 485-million-token file all at once. Every training step, we grab a small random window: 32 sequences of 256 tokens each. That's one batch.

### Memory-mapping

The key technique is `np.memmap`. Instead of loading the multi-gigabyte `.bin` file into RAM, we *map* it — the operating system pages in only the bytes we actually touch. We can treat a huge file like a normal array while using almost no memory.

```python
data = np.memmap(path, dtype=np.uint16, mode='r')
```

The `dtype=np.uint16` must match exactly what we wrote in Episode 2. `uint16` holds values 0–65,535, plenty for our 4,096-token vocabulary, at 2 bytes per token.

### Why y is x shifted by one

The model's job is "predict the next token." For every position in the input, the correct answer is simply the token that comes right after it. So the target `y` is just the input `x` shifted left by one position:

```
data  = [ The , cat , sat , on , the , mat ]
x     = [ The , cat , sat , on ]
y     = [ cat , sat , on  , the ]
```

Read it as: "after `The` → `cat`", "after `cat` → `sat`", and so on. Every position in `x` is paired with its correct next token in `y`.

### Why we re-open the memmap every batch

`data_loader.py` re-creates the `np.memmap` on every `get_batch()` call rather than keeping one long-lived handle. This avoids a known memory-leak quirk: repeatedly reading from a single long-lived memmap slowly grows memory usage over a long training run. Re-opening it each batch sidesteps the problem entirely.

---

## train.py — The Training Loop

### The five steps of one training step

Inside the loop, each iteration does exactly five things:

1. **Get a batch** — `loader.get_batch('train')` returns one random 32×256 batch of input/target pairs.

2. **Forward pass** — `model(x, y)` runs the input through all 9 steps from Episode 3 and, because we passed targets, also computes the cross-entropy loss.

3. **Backward pass** — `loss.backward()` computes, for every one of the 12 million parameters, how much it contributed to the error. This is backpropagation. We call `optimizer.zero_grad()` first because PyTorch accumulates gradients by default and we want a fresh start each step.

4. **Clip gradients** — `clip_grad_norm_` caps the total gradient size. Occasionally a bad batch produces enormous gradients that would knock the model's weights out of a good region; clipping keeps training stable.

5. **Update the weights** — `optimizer.step()` nudges every parameter a small step in the direction that reduces loss. This is the actual learning.

### The optimizer: AdamW

AdamW is the standard optimizer for transformers. It does two jobs:

- **The Adam part** keeps a running average of each parameter's gradient (momentum) and its variance, so every parameter gets its own adaptive step size. Parameters with consistently large gradients take smaller, more careful steps.
- **The W part** applies weight decay correctly — gently pulling weights toward zero, a form of regularization that reduces overfitting.

### The learning rate schedule

The learning rate is **not constant** during training. It follows a two-phase schedule:

**Phase 1 — Warmup (steps 0 to 1,000).** The LR ramps linearly from near-zero up to the peak (`3e-4`). At the very start, the model's parameters are random; large updates on random parameters can destabilize training. A gentle warmup lets the model find its footing.

**Phase 2 — Cosine decay (steps 1,000 to 25,000).** The LR follows a cosine curve down from the peak to a minimum (`3e-5`). Early in training, big steps make fast progress. Later, the model is near a good solution and needs small, careful steps to settle in. The cosine shape gives a smooth transition.

The curve looks like a quick ramp up, then a long gentle slope down.

| Step | Learning rate | Phase |
|---|---|---|
| 0 | ~3e-7 | Warmup start |
| 500 | ~1.5e-4 | Ramping up |
| 1,000 | 3e-4 | Peak — warmup done |
| 13,000 | ~1.65e-4 | Mid cosine decay |
| 25,000 | 3e-5 | Decay done |

### Validation and checkpointing

Every 500 steps, training pauses and runs `estimate_loss` — it measures average loss on both the train and validation splits, averaged over 50 batches each.

Why both splits?
- **Train loss** tells you how well the model fits the data it sees.
- **Validation loss** tells you how well it *generalizes* to data it never trains on.

If train loss keeps dropping but validation loss stalls or rises, the model is overfitting — memorizing the training data instead of learning patterns.

Each time validation loss improves, `train.py` saves a checkpoint. It only saves on improvement, so if training later goes bad, the saved checkpoint is still the best version the model ever reached.

---

## Running the Training

First, if you haven't already, prepare the data (one-time step):

```powershell
python prepare_data.py
```

Then start training:

```powershell
python train.py
```

What you'll see:

```
============================================================
 Training a GPT from scratch
============================================================
Device: cuda

Loading data...
  train.bin:      485,XXX,XXX tokens
  validation.bin: X,XXX,XXX tokens

Building model...
  Parameters: 12,199,680

Starting training for 25,000 steps...
--------------------------------------------------------------------------------
step    500 | train 5.1030 | val 3.7570 | lr 3.00e-04 | 380.6ms/step
       └─ saved checkpoint (val loss 3.7570)
step   1000 | train 3.4063 | val 3.0372 | lr 3.00e-04 | 324.0ms/step
       └─ saved checkpoint (val loss 3.0372)
step   1500 | train 2.8421 | val 2.6105 | lr 2.99e-04 | 318.7ms/step
       └─ saved checkpoint (val loss 2.6105)
...
```

One line per 500 steps, containing everything that matters: step number, average train loss (over 50 batches), validation loss (also averaged), current learning rate, and the average time per step over that interval. A `saved checkpoint` line appears below whenever val loss improved.

Per-step loss is intentionally not printed every step — each batch is different, so per-step numbers bounce around noisily. The 50-batch average that prints at each eval interval is far more meaningful, and one rich line per interval is much easier to read than 10 noisy lines.

Expected total time on a GTX 1660 SUPER: **about 2 hours** for 25,000 steps.

---

## What "Good" Looks Like

A healthy training run has a few recognizable signs:

**The loss drops fast at first, then slows.** The steepest decline is in the first ~2,000 steps — the model goes from "completely random" to "knows basic letter and word patterns" quickly. After that, progress is slower and steadier.

**Train and validation loss stay close together.** A small gap is normal and healthy. If the gap widens dramatically — train loss at 1.2 while validation sits at 2.5 — that's overfitting.

**No sustained upward trend.** Occasional small spikes are normal (a hard batch). A loss that climbs and stays up means something is wrong — usually the learning rate is too high.

The target numbers for our setup:

| Milestone | Approx. loss | What the model can do |
|---|---|---|
| Step 0 | ~8.3 | Nothing — pure noise |
| Step 1,000 | ~4.5 | Letter clusters, common short words |
| Step 5,000 | ~2.6 | Real words, rough word order |
| Step 15,000 | ~1.9 | Mostly-correct grammar, simple sentences |
| Step 25,000 | ~1.7 | Coherent short stories |

---

## Underfitting and Overfitting

Two concepts every machine-learning practitioner needs to recognize on sight.

**The short version:**

| Term | What it means, in plain words |
|---|---|
| **Underfitting** | The model is **too dumb** for the task. It hasn't learned enough yet. Loss is high on both training and validation data. |
| **Overfitting** | The model **memorized** the training data instead of learning real patterns. Loss is low on training data but high on data it hasn't seen. |
| **Healthy** | The model is **learning real patterns**. Loss drops on both training and validation, and the two stay close together. |

A simple analogy:

- **Underfit student** → didn't study. Fails the practice test *and* the real test.
- **Overfit student** → memorized the practice test word-for-word. Aces the practice test but fails the real test because the real questions are slightly different.
- **Healthy student** → actually learned the material. Does well on both.

The rest of this section explains each in more detail.

### Underfitting — the model hasn't learned enough yet

Both the training loss and the validation loss are still high. The model is failing to capture the patterns in the data — not because the data is bad, but because the model hasn't yet built the internal representations it needs.

A quick analogy: a student who hasn't studied. They do badly on the practice exam *and* badly on the real exam — both for the same reason: they just don't know the material yet.

**Signs:** both train and val loss are high. Outputs are still mostly nonsense.
**Cure:** train longer, use a bigger model, or use more diverse data.

In our run, **underfitting is the entire first ~2,000 steps**. Both losses are above 5. The model hasn't learned enough yet. We don't worry — that's the expected starting point. We just keep training.

### Overfitting — the model memorized the training data

The training loss keeps dropping, but the validation loss stops dropping or starts climbing. The model is no longer learning *general patterns* — it's memorizing the specific examples it sees in training, and that memorization doesn't help it on data it hasn't seen.

A quick analogy: a student who memorized the practice exam word for word. They ace the practice exam but fail the real exam — because the questions are slightly different and they never actually learned the underlying material.

**Signs:** train loss keeps dropping, val loss stalls or rises. The gap between the two grows over time.
**Cure:** stop training (early stopping), use less model capacity, more data, or stronger regularization (dropout, weight decay).

In our run, **overfitting is unlikely to be a problem** because TinyStories has 485 million tokens — far more than our 12M-parameter model can memorize in 25,000 steps. But on smaller datasets or with bigger models, it's the most common failure mode.

### What we want to see — the sweet spot

Both losses are dropping, and they stay close together. The model is genuinely learning patterns that work on data it has never seen. That's a healthy training run.

```
   Loss
    │  ╲
  8 │   ╲╲             ← Underfitting
    │     ╲╲             (both still high)
  6 │       ╲╲
    │         ╲╲╲
  4 │            ╲╲       ← Healthy learning
    │              ╲╲╲      (both dropping together)
  2 │                 ╲╲╲╲╲___
    │ ─── train  ─── val
    └──────────────────────────── steps
```

The two lines should descend together. If the val line peels away upward while the train line keeps falling — that's overfitting, and it's time to stop training.

---

## Common Questions

### "Why batch_size = 32 and not larger?"

Larger batches give smoother gradient estimates, but they also use more VRAM. On a 6 GB GPU, 32 sequences of 256 tokens is roughly the largest batch that fits comfortably alongside the model, its gradients, and the optimizer's momentum/variance buffers. This was worked out in the Episode 2 memory math.

### "Why does the loss bounce around instead of dropping smoothly?"

Each step sees a different random batch. Some batches are harder than others, so the per-step loss is noisy. The *trend* is what matters — over a few hundred steps the average clearly drops, even though individual steps jump up and down. This is why we also run the smoother `estimate_loss` every 500 steps.

### "What's gradient clipping actually doing?"

It measures the total size (norm) of all gradients combined. If that total exceeds `GRAD_CLIP` (1.0 for us), every gradient is scaled down proportionally so the total equals 1.0. The *direction* of the update is preserved; only the *size* is capped. This prevents a single unlucky batch from making a destructive jump.

### "Why save checkpoints only when validation improves?"

If we saved every 500 steps unconditionally, a checkpoint taken right after training went bad would overwrite a good one. Saving only on improvement guarantees that `checkpoint.pt` is always the best version the model ever reached.

### "Can I stop and resume training?"

The checkpoint saves the model weights, the optimizer state, and the step number — everything needed to resume. Adding resume logic is a small extension: load the checkpoint at startup and start the loop from the saved step. We keep `train.py` simple here and run it in one pass, but the saved state makes resuming straightforward if you want to add it.

---

## What's Next: Episode 06

In [Episode 06](../episode-06-watching-it-learn/), we don't write much new code. Instead, we *read* the training run.

Episode 6 takes the logs and checkpoints this episode produced and walks through:
- How to read a loss curve — what healthy looks like, what trouble looks like
- The train/validation gap and what it tells you
- Perplexity — a more interpretable cousin of the loss number
- The sample outputs at checkpoints 1,000 / 5,000 / 15,000 / 25,000 — watching gibberish become stories

It's the most underrated episode in any from-scratch LLM series, and it's where the 2 hours of training pays off.

---

## References

- [Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)](https://arxiv.org/abs/1412.6980) — the original Adam optimizer paper.
- [Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1711.05101) — the paper introducing AdamW, the "W" in our optimizer.
- [SGDR: Stochastic Gradient Descent with Warm Restarts (Loshchilov & Hutter, 2016)](https://arxiv.org/abs/1608.03983) — introduces the cosine learning rate schedule.
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) — the reference training loop this is based on.
