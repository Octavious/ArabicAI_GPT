# Episode 03 — What Is a Transformer, Really?

The third episode in the [Build a GPT From Scratch on a 6 GB GPU](../README.md) series.

**▶ [Watch the video](https://www.youtube.com/playlist?list=PLvmpljk9TE2v4LbSrQxmnbOKkLa0Mw1oD)**

---

## What This Episode Is For

No code in this episode. Just intuition.

By the end, you should be able to draw the data flow through a transformer on a napkin and explain — to yourself or someone else — what each step does and why it's needed. That's the point. If you understand the architecture conceptually, the code in Episode 4 is just translation.

This episode is built around three interactive visualizations and one analogy. Both are linked below.

---

## Prerequisites

You need [Episode 02](../episode-02-data-and-tokenizer/) finished:
- A trained BPE tokenizer (`tinystories_tokenizer.json`)
- Two binary files of token IDs ready for training (`train.bin`, `validation.bin`)

You don't need to have written any model code yet. That's Episode 4's job.

---

## The Mental Model: Students in a Row

A row of students, each holding a card with a word on it. They don't know anything about each other's cards yet — they can only read their own. Over six rounds, they turn around, read each other's cards, think quietly about what they learned, and update their own notes. By the end, each student understands not just their own word, but where it fits in the full sentence.

Everything in this episode maps onto that picture:

| Model concept | Students analogy |
|---|---|
| A token | A student holding a card |
| Token ID | The number printed on the card |
| Token embedding | What the card *means* — a list of properties |
| Positional embedding | Each student knowing their seat number |
| A batch | 32 separate classrooms doing this in parallel |
| Attention | Students turning around to read each other's cards |
| Multi-head attention | Looking at different aspects simultaneously |
| Feed-forward | Each student thinking quietly about what they learned |
| LayerNorm | The teacher saying "settle down" to keep notes calm |
| Residual connection | Keeping your original notes while adding new ones |
| Stacking 6 blocks | Six rounds of "talk, then think" |
| Output projection | Each student writing what word they think comes next |
| Softmax | Turning rough scores into clean percentages |
| Loss | The teacher checking how confident the right answer was |
| Generation | Picking one student's guess and seating a new student |

Hold this picture in your head as you read.

---

## The Interactive Visualizations

These are the heart of the episode. You should open and play with each one.

### 01 — [Embedding Explorer](./01_embedding_explorer.html)

Pick any two words from a curated library and see them as vectors. Each word is plotted on twelve labeled dimensions (animal-ness, action-ness, emotion, positivity, etc.). The tool shows:

- The cosine similarity between the two words
- A dimension-by-dimension breakdown — each axis runs from −1 to +1 with zero in the middle
- The six closest neighbors of each word

**What to try:** compare `happy` vs `sad` (mirror images on positivity), `cat` vs `dog` (almost identical), `kitten` vs `tiger` (same animal-ness, opposite size and pet-ness).

The teaching moment: **words are vectors**, and meaning lives in the geometry. Similar words point in similar directions.

### 02 — [Shape Flow](./02_shape_flow.html)

A nine-step storyboard walking through how a tensor's shape evolves through the model. Click through with arrow keys or the navigation tabs.

Each step has its own deep-dive panel structure: a **purpose framing** at the top (what state are we in coming in, what does this step accomplish), then mechanism panels below.

**What to try:** start at Step 1 and walk through to Step 9. Each step builds on the previous. The Q/K/V breakdown in Step 4 and the expand/filter/compress breakdown of feed-forward are the most important panels in the whole episode.

### 03 — [Attention Visualizer](./03_attention_visualizer.html)

Type a sentence, click a token, watch what it attends to. Six hand-crafted attention heads are available, each with a different specialization:

- **Previous token** — the most common pattern in real models
- **Self-attention only** — preserving identity
- **Punctuation tracker** — clause boundaries
- **Pronoun resolution** — when "it" looks back to find what it refers to
- **Content words** — ignoring `the`, `a`, `of`
- **Sentence start** — the famous "BOS sink" pattern

Both a curved-arc visualization (above the tokens) and a heatmap matrix (below) show the same data. Causal masking is shown as visually striped cells in the upper triangle.

**What to try:** "The cat sat on the mat because it was tired" with the pronoun-resolution head. Click `it` and watch the attention curve back to `cat`.

The teaching moment: **attention is asymmetric**. Each token has a question; it pulls in information from whoever has the best answer. Not all tokens look at all tokens equally.

---

## The Nine-Step Pipeline

Here's the same flow, summarized for reference. Detailed walkthrough is in the third visualization above.

### Step 1 — Tokens `(B, T)`

The tokenizer converts our sentence into integer IDs. The students are seated, each holding a card with a number on it. Card 365 means "·Lily" (with a leading space) — but only because our tokenizer happened to assign that ID. The number itself carries no meaning yet.

We process **32 sequences in parallel** (batch dimension B), each with **256 tokens** (sequence dimension T). The GPU is built for this kind of parallelism — doing 32 at once isn't 32× slower, it's only 2-3× slower.

### Step 2 — Token Embeddings `(B, T, C)`

The model has a **giant dictionary** with 4,096 entries — one for every possible card. Each entry is a list of 384 properties describing what that card means.

Each student trades their card-number for its dictionary entry. After training, similar words have similar entries. "Lily" and "Tim" (both common TinyStories names) end up with vectors pointing in similar directions; "Lily" and "explosion" don't.

The shape grows: we added a new dimension **C = 384** for the meaning vector at each position.

### Step 3 — Positional Embeddings `(B, T, C)`

Without seat numbers, "the cat sat" and "sat the cat" look identical to the model. Attention treats its input as a *set*, not a sequence.

We make a **second dictionary**, this one keyed by seat number (0 through 255). Each seat has its own learned 384-number signature. Every student adds their seat's signature to their card's meaning.

We **add** rather than concatenate to keep the shape unchanged at `(B, T, C)`. With 384 dimensions, the model has plenty of room to use some for word-meaning and others for position information, even when blended together.

### Step 4 — Transformer Block × 6 `(B, T, C)`

The heart of the model. Each block does two things, in two phases.

**Phase 1 — "Talk" (attention).** Each student forms three things from their card:

- **Q (Query)** — "what kind of word would help me right now?"
- **K (Key)** — "here's what I am, in case anyone needs me"
- **V (Value)** — "here's the actual content I'd share"

The student matches their Q against everyone's K. Whoever matches best wins more attention. The student receives a **weighted blend of everyone's V**.

The asymmetry matters: each student decides *for themselves* what they need. Two students looking at the same row can pay attention to completely different things, because their questions are different.

**Phase 2 — "Think" (feed-forward).** After attention, each student has a card with bits of context mixed in. Now they transform those notes into something more useful, in three sub-steps:

1. **Expand** — stretch the 384-number card into a 1,536-number working space (4× wider). Imagine many learned "lenses" each highlighting different combinations.
2. **Filter** — apply a non-linearity (GELU). Keep positive signals, suppress negatives. *This is the most important step* — without it, the whole feed-forward collapses into a single linear transformation.
3. **Compress** — squeeze back down to 384 numbers, ready for the next round.

**Most of the model's parameters live in feed-forward** — roughly twice as many as in attention. A common misconception is that "attention is the model"; in fact, attention is the routing system, and the feed-forward is the actual computation and storage.

**Wrappers around each phase:**

- **LayerNorm** before each sub-layer, keeping notes at a calm volume
- **Residual connection** around each sub-layer — students keep their original notes and add new ones on top, instead of replacing

We do six rounds of this. By the end, each student's card reflects their full sentence context. The shape stays at `(B, T, C)` the whole time.

### Step 5 — Final LayerNorm `(B, T, C)`

After six rounds, the residuals have kept adding to each card. Some values are now uncomfortably large.

One last "settle down, everyone" moment. For each student's vector: subtract the mean, divide by the standard deviation, then apply two learned parameters (γ scale, β shift) so the model can re-stretch if useful.

### Step 6 — Output Projection `(B, T, V)`

Each student now writes their guess. A final linear layer maps the 384-number card into 4,096 raw scores — one for every word in our vocabulary. These scores are called **logits**.

Higher score = the model prefers that word as the next one.

This happens at every position in parallel. The model produces 256 sets of vocabulary scores per sequence. Every position predicts what comes after it, all at once.

**Weight-tying trick:** in our model, this output projection shares its weights with the token embedding table from Step 2. Same dictionary, used in both directions: input embedding turns IDs into meanings, output projection turns meanings back into ID-scores. This saves us ~1.6 million parameters.

### Step 7 — Softmax `(B, T, V)`

Logits are messy — some huge, some negative. Softmax converts them into clean **percentages**: every value between 0 and 1, all values summing to exactly 1.

Same shape, new meaning. What was a raw preference score is now a proper probability you can sample from or grade.

### Step 8 — Loss / Sample

The same probabilities, two different uses.

**During training**: for each prediction, look up the percentage the model assigned to the *correct* token. Take its negative log. If the model assigned high probability (say 0.9), the loss is low (0.1). If it assigned low probability (0.001), the loss is high (6.9). Average across all 32 × 256 = 8,192 predictions to get the batch loss.

**During generation**: we don't know the correct answer — that's the whole point. Sample one token from the last position's distribution, weighted by the probabilities. Three common strategies:

- **Greedy** — always pick the highest probability. Deterministic but boring.
- **Temperature** — sharpen or flatten the distribution before sampling.
- **Top-k** — only consider the k most likely tokens.

### Step 9 — Repeat

Generation is autoregressive. Pick a token, append it to the sequence, run the entire 9-step pipeline again to get the next token. Repeat 200 times to write a short story.

The pipeline you just walked through runs roughly 10,000 times per minute during training, and a few hundred times during a single generation.

---

## Three Concepts That Trip Everyone Up

These are the parts where most beginners stay confused. Worth pausing on.

### Why we *add* token and positional embeddings

When you first see "we add the position vector to the token vector," it feels wrong. Wouldn't that mix them up irretrievably?

In practice, no. Two reasons:

**Shape preservation.** Adding keeps the shape at `(B, T, C)`. Concatenating would double it to `(B, T, 2C)`, doubling the cost of every downstream operation.

**384 dimensions is plenty of room.** During training, the model learns to use some dimensions primarily for token information and others primarily for position. The "mixing" is a non-problem because the model has plenty of space to keep them functionally separate.

### Why attention is asymmetric (Q ≠ K ≠ V)

Many introductions to attention make it sound symmetric: "tokens compare themselves to other tokens." That's misleading.

The mechanism is **one-directional**. Each token forms its own *question* (Q) about what it needs. It looks at every other token's *answer-it-offers* (K). It receives a weighted blend of *content-to-share* (V).

A token's Q, K, and V are derived from the same starting card via three different learned matrices — `W_q`, `W_k`, `W_v`. The model learns over training to make each transformation specialized for its role.

This asymmetry is what makes attention powerful. Two tokens looking at the same row can pay attention to completely different things, because their questions are different.

### Why we need both attention *and* feed-forward

A common misconception: "attention is the magic, feed-forward is just glue."

Reality:

- **Attention** is the **communication system**. It decides which tokens share information with which other tokens. It's a routing decision.
- **Feed-forward** is the **computation**. It takes the routed information and transforms it — combining features, suppressing irrelevant signals, computing higher-level facts.

Interpretability research has shown that specific factual knowledge (like "Paris is the capital of France") is stored in *feed-forward weights*, not in attention. Most of the model's parameters live in the feed-forward — for our model, roughly 2× as many as in attention.

A useful one-liner: **attention is how tokens talk; feed-forward is how they think.**

---

## Numerical Tour: What Our Model Actually Looks Like

For the model we're building, here's every dimension you'll see:

| Symbol | Name | Value | Meaning |
|---|---|---|---|
| **B** | batch size | 32 | sequences processed in parallel |
| **T** | block size / context length | 256 | tokens per sequence |
| **C** / `n_embd` | embedding dimension | 384 | numbers per token vector |
| **n_layer** | number of blocks | 6 | rounds of "talk, then think" |
| **n_head** | attention heads per block | 6 | parallel attention specializations |
| **head_size** | size per attention head | 64 | C ÷ n_head = 384 ÷ 6 |
| **V** | vocabulary size | 4,096 | possible tokens |
| **dropout** | dropout probability | 0.1 | regularization rate |

These choices are deliberate. They cap the parameter count at ~12 M, which fits comfortably on our 6 GB GPU during training, and make attention scale reasonably (the `T²` term in attention memory is `256² = 65,536` numbers per head — tractable).

---

## Why "Multiple Heads" Helps

We haven't talked much about why attention has multiple heads in parallel.

A **single attention head** can only attend to one pattern at a time. If a head learns to track subjects, it can't also track punctuation in the same forward pass.

**Multiple heads** (six in our model) run in parallel, each with its own Q/K/V transformations. They each learn to specialize. After training, you'll typically find:

- One head focused on the previous token (local syntactic dependencies)
- One head tracking long-range references (pronoun resolution)
- One head focused on punctuation (clause boundaries)
- One head doing position-relative attention
- And so on

The outputs from all six heads are concatenated and combined through one final linear layer that learns which combinations are most useful.

This is exactly what the [Attention Visualizer](./02_attention_visualizer.html) shows — six different head specializations on the same sentence. Click through them and notice how dramatically different the attention patterns look.

---

## What's *Not* Covered (Yet)

This episode keeps things tractable by skipping a few details that we'll meet in Episode 4 when we write the code:

- **Causal masking** — the triangular mask that prevents tokens from attending to the future. Without it, training would leak the answer.
- **Scaled dot-product** — the `÷ √head_size` divisor in the attention formula. Without it, attention scores become too sharp at high dimensions.
- **Dropout** — the regularization that randomly zeros out 10% of activations during training.
- **Weight initialization** — why we start parameters near zero (small Gaussian noise) rather than at zero exactly.

These are all engineering details that matter for the code but don't change the conceptual picture. We meet them when we need them in Episode 4.

---

## Your Mental Checklist

Before moving to Episode 4, try to answer these *without looking back*:

1. What's in the input tensor at the start? What's in it after Step 2?
2. What problem does positional embedding solve?
3. What are Q, K, V — in plain language?
4. What does the feed-forward network do that attention can't?
5. Why do we use residual connections?
6. Why do we stack six blocks instead of one big one?
7. What's the difference between logits and probabilities?
8. What's the difference between training and generation, in terms of how we use the output?

If any of these feel shaky, go back to the [Shape Flow visualization](./03_shape_flow.html) and walk through the relevant step. Don't proceed to Episode 4 until they all click — the code will be much harder if the architecture isn't intuitive yet.

---

## What's Next: Episode 04

In [Episode 04](../episode-04-building-the-model/), we translate this entire mental model into PyTorch code. Every concept here gets a class:

- `MultiHeadAttention` — the Q/K/V mechanism in matrix form
- `FeedForward` — the expand/filter/compress pipeline
- `Block` — combining the two with LayerNorm and residuals
- `GPT` — the full model with embeddings, blocks, and the output projection

By the end of Episode 04, you'll have a `model.py` file that, when called with a tensor of token IDs, runs through all 9 steps and returns logits. Then Episode 05 trains it.

---

## References and Further Reading

The architecture we're building is GPT-2's, with minor modernizations. Some essential resources if you want to go deeper:

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — the original transformer paper. Hard to read, but every concept here traces back to it.
- [Andrej Karpathy's "Let's build GPT" video](https://www.youtube.com/watch?v=kCc8FmEb1nY) — single best video on building a transformer from scratch.
- [The Illustrated Transformer (Jay Alammar)](http://jalammar.github.io/illustrated-transformer/) — beautiful visual walkthrough of attention and the full architecture.
- [3Blue1Brown's transformer series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — animated explanations of attention and embeddings, with an emphasis on geometric intuition.
- [Anthropic's "A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html) — research-grade view of what attention heads actually compute. Heavy reading, but rewarding.

For interpretability work specifically (the "feed-forward stores facts" claim):

- [Geva et al. — "Transformer Feed-Forward Layers Are Key-Value Memories"](https://arxiv.org/abs/2012.14913)
- [Meng et al. — "Locating and Editing Factual Associations in GPT"](https://arxiv.org/abs/2202.05262)
