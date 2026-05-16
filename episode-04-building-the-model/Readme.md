# Episode 04 — Building the Model

The fourth episode in the [Build a GPT From Scratch on a 6 GB GPU](../README.md) series.

**▶ [Watch the video](https://www.youtube.com/playlist?list=PLvmpljk9TE2v4LbSrQxmnbOKkLa0Mw1oD)**

---

## What This Episode Is

Episode 3 was theory. This episode is code.

Every concept from Episode 3 — the students analogy, Q/K/V, expand-filter-compress, residuals — becomes a Python class. By the end of the episode, we have `model.py`: a file you can run, feed token IDs into, and get vocabulary scores out of.

The model doesn't know anything useful yet. It's randomly initialized — as confused as possible. That's exactly right. Episode 5 trains it.

---

## Prerequisites

[Episode 03](../episode-03-architecture-theory/) finished, especially the Shape Flow visualization. Every class in `model.py` maps directly to one step in that storyboard.

If any architectural concept feels unclear before writing code, go back to the Shape Flow and walk through the relevant step. The code is translation, not new theory.

---

## Files in This Episode

| File | What it is |
|---|---|
| `model.py` | The complete model — four classes, ~200 lines, every non-obvious line commented |
| `episode_04_slides.html` | Slide deck used during the recording — open in any browser |

---

## Quick Reminder: Head vs Block

A common source of confusion — they are two different things.

A **block** is one full round of "talk, then think": CausalSelfAttention followed by FeedForward, wrapped in LayerNorm and residuals. We stack **6 blocks**.

A **head** is a subdivision *inside* the attention step of one block. Instead of one big attention operation, we run **6 smaller ones in parallel**, each specializing on a different type of relationship.

```
1 Block
├── CausalSelfAttention  ← contains 6 heads running in parallel
│   ├── Head 1           ← learns one pattern (e.g. previous token)
│   ├── Head 2           ← learns another  (e.g. subject tracking)
│   └── ... × 6
└── FeedForward          ← no heads here — just expand/filter/compress
```

Our model: **6 blocks × 6 heads = 36 attention heads total**.

---

## The Four Classes

`model.py` builds up in this order — each class depends on the previous one.

### GPTConfig

```python
@dataclass
class GPTConfig:
    vocab_size: int   = 4096   # our BPE tokenizer
    block_size: int   = 256    # context window
    n_embd:     int   = 384    # embedding dimension
    n_layer:    int   = 6      # transformer blocks
    n_head:     int   = 6      # attention heads per block
    dropout:  float   = 0.1   # regularization rate
```

All hyperparameters in one dataclass. Change one number here and the entire model adapts. These values come from the memory math in Episode 2:

- `n_embd = 384` targets ~12M parameters on our 6 GB GPU
- `n_layer = 6` is deep enough for TinyStories, shallow enough to train in 2 hours
- `n_head = 6` gives `384 ÷ 6 = 64` dimensions per head — the practical sweet spot
- `block_size = 256` fits a short story in one context window
- `vocab_size = 4096` matches the BPE tokenizer we trained in Episode 2

### CausalSelfAttention — the "talk" phase

Each token forms a **Query** ("what am I looking for?"), a **Key** ("here's what I am"), and a **Value** ("here's what I'd share") — all derived from its current embedding through three learned linear projections.

The attention score between two tokens is the dot product of their Q and K. Tokens whose K best matches a given Q contribute more of their V to that token's output.

**Three details worth slowing down for when reading the code:**

**The reshape trick.** We compute Q, K, V for all 6 heads in one matrix multiply, then reshape:

```python
head_size = C // self.n_head                       # 384 // 6 = 64
q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
#   (B, T, 384) → (B, T, 6, 64) → (B, 6, T, 64)
```

After the transpose, the 6 heads sit next to the batch dimension. PyTorch treats them as 6 independent operations running in parallel.

**The causal mask.** One line prevents any token from attending to a future position:

```python
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
```

Future positions get `-inf` before softmax. After softmax, `-inf` becomes exactly `0.0` — contributing precisely nothing. This is what "causal" means: students can only read earlier seats.

**Scaling by √head_size.** Raw dot products grow in magnitude as head_size increases. Large values push softmax into its flat regions where gradients vanish. Dividing by `√64 = 8` keeps the variance stable.

```python
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
```

### FeedForward — the "think" phase

After attention, each token has gathered context from earlier positions. The feed-forward network processes each token independently to transform that gathered information into something more useful.

Three sub-steps:

```python
nn.Linear(n_embd, 4 * n_embd)   # Expand: 384 → 1,536
nn.GELU()                         # Filter: keep positive, smooth out negative
nn.Linear(4 * n_embd, n_embd)   # Compress: 1,536 → 384
```

**The most important line is `nn.GELU()`.** Without a non-linearity, the expand and compress layers would collapse into a single linear transformation — the whole block would add nothing useful. GELU is what makes the feed-forward capable of learning complex, non-linear relationships.

**56% of the model's parameters live here** — roughly twice as many as in attention. The feed-forward is where the model stores factual knowledge, not attention. Attention routes information; feed-forward transforms it.

### Block — one complete round

```python
def forward(self, x):
    x = x + self.attn(self.ln_1(x))   # talk, then add back
    x = x + self.ffn(self.ln_2(x))    # think, then add back
    return x
```

Two things to notice:

**LayerNorm before each sub-layer (pre-norm).** The original 2017 transformer paper applied LayerNorm after each sub-layer. GPT-2 switched to pre-norm because it trains more stably for deeper models. This is the GPT-2 style.

**Residual connections (`x = x + ...`).** The sub-layer only needs to learn an adjustment, not a complete transformation. The `+ x` shortcut lets gradients flow backward through all 6 blocks without fading. Without residuals, deep networks fail to train.

The most important property of a block: **input shape = output shape = (B, T, C)**. That's why we can stack 6 of them.

### GPT — the full model

The full forward pass in seven steps:

```python
tok = self.tok_emb(idx)                        # Step 2: token embeddings  (B, T, 384)
pos = self.pos_emb(torch.arange(T, ...))       # Step 3: position embeddings (T, 384)
x   = self.drop(tok + pos)                    # Steps 2+3: combined       (B, T, 384)
x   = self.blocks(x)                           # Step 4: 6 transformer blocks (B, T, 384)
x   = self.ln_f(x)                             # Step 5: final LayerNorm   (B, T, 384)
logits = self.lm_head(x)                       # Steps 6+7: project to vocab (B, T, 4096)
```

**One detail worth calling out — weight tying:**

```python
self.tok_emb.weight = self.lm_head.weight
```

The token embedding table (which maps IDs → meaning vectors) and the output projection (which maps meaning vectors → vocabulary scores) share the same matrix. Same dictionary, used in both directions:

- Input side: token ID → 384-number meaning vector
- Output side: 384-number vector → 4,096 vocabulary scores

This saves ~1.6M parameters and consistently improves performance. It works because the two operations are genuinely related — the same geometric space that encodes token meaning can be inverted to score tokens.

---

## Running the Code

### Install dependencies

```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

No other dependencies. `model.py` only uses PyTorch and Python's standard library.

### Run the sanity check

```powershell
python model.py
```

Expected output:

```
Parameters: 12,369,920
Input shape:  torch.Size([4, 64])
Logits shape: torch.Size([4, 64, 4096])
Loss:         8.3178  (expect ~8.32)
Sanity check passed.
```

### Why the loss is ~8.32

A randomly initialized model assigns roughly equal probability to all 4,096 tokens. The cross-entropy loss of a uniform distribution over 4,096 options is:

```
ln(4096) = 8.3178...
```

If you see a number close to 8.32, the model is working correctly — it's as confused as possible, which is exactly the right starting point for training. If the number is very different, something is wrong in the forward pass.

---

## Where the Parameters Live

```
python -c "
from model import GPT, GPTConfig
m = GPT(GPTConfig())
print(f'Total: {m.get_num_params():,}')
"
```

| Component | Parameters | Share |
|---|---|---|
| FeedForward expand + compress (×6 blocks) | 7,077,888 | 56% |
| Attention Q/K/V + output proj (×6 blocks) | 3,538,944 | 28% |
| Token embedding | 1,572,864 | 12% |
| LayerNorms + positional embedding | ~180,224 | ~1% |
| **Total** | **~12.4 M** | **100%** |

The common assumption is that "attention is the model." The numbers say otherwise: **feed-forward has twice as many parameters as attention**. Research has shown that specific facts are stored in feed-forward weights — attention does the routing, feed-forward does the computation and storage.

---

## Common Questions

### "Why bias=False on the linear layers?"

The LayerNorm layers that follow include their own learned bias (β), which effectively plays the same role. Adding a linear bias on top is redundant and wastes parameters.

### "Why -inf and not -1000 in the causal mask?"

After softmax, `-inf` becomes exactly `0.0` — the future position contributes precisely nothing. A large negative number like `-1000` produces a very small but nonzero weight, which is mathematically incorrect. `float('-inf')` is the right implementation.

### "What is .contiguous() doing?"

After `.transpose()`, the tensor's data is still in the original memory layout — only the metadata describing how to read it changes. `.view()` needs the data to be physically contiguous in memory to reshape it. `.contiguous()` forces a copy into the correct layout before the reshape.

### "Why small random initialization, not zeros?"

If all weights start at zero, every neuron in a layer computes the same output, receives the same gradient, and updates identically. The layer never learns diverse features — it's stuck. Small random values (std=0.02, matching GPT-2) break this symmetry from the start.

### "Why does get_num_params subtract positional embeddings?"

By convention, the "model size" figure typically excludes positional embeddings because they're a table lookup rather than a learned projection. Subtracting `pos_emb` (256 × 384 = 98,304 params) gives the number more commonly reported. Both numbers are valid — just be consistent when comparing.

---

## What's Next: Episode 05

In [Episode 05](../episode-05-training/), we train this model.

`train.py` will:
- Load `train.bin` using memory-mapped reads
- Run the forward pass, compute the loss, run backpropagation
- Update the model weights with AdamW
- Evaluate on `validation.bin` every 500 steps
- Save checkpoints so training can be resumed

By the end of Episode 05, the loss will have dropped from **8.32 → ~1.7**. The model will have gone from "as confused as possible" to writing coherent children's stories.

---

## References

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — the original transformer paper. The architecture here is a decoder-only simplification of that design.
- [Language Models are Unsupervised Multitask Learners — GPT-2 (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — introduces pre-norm and weight tying used here.
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) — the reference implementation this series is based on.
- [GELU Activation Function (Hendrycks & Gimpel, 2016)](https://arxiv.org/abs/1606.08415) — the paper introducing GELU.
- [Transformer Feed-Forward Layers Are Key-Value Memories (Geva et al., 2020)](https://arxiv.org/abs/2012.14913) — the paper showing factual knowledge is stored in feed-forward weights.
