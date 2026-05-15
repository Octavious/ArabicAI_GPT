"""
═══════════════════════════════════════════════════════════════════════════════
 model.py — GPT Language Model
 Built in Episode 04 of "Build a GPT From Scratch on a 6 GB GPU"
 YouTube: https://www.youtube.com/playlist?list=PLvmpljk9TE2v4LbSrQxmnbOKkLa0Mw1oD
═══════════════════════════════════════════════════════════════════════════════

 This file is the code translation of the 9-step pipeline we walked through in
 Episode 3 (Shape Flow). The Episode 3 storyboard and this file are designed to
 be read together — every step in the storyboard has a corresponding section
 here.

 The mental model: a row of students, each holding a card with a word on it.
 Over six rounds, they turn around, read each other's cards, think quietly
 about what they learned, and update their own notes. By the end, each student
 understands not just their own word, but where it fits in the full sentence.

 ┌──── Episode 3 Step ──────────────────┬── Where it lives in this file ──────┐
 │  1. Tokens             (B, T)        │  Input to GPT.forward()              │
 │  2. Token Embeddings   (B, T, C)     │  GPT.tok_emb                         │
 │  3. + Position         (B, T, C)     │  GPT.pos_emb + addition              │
 │  4. Transformer Block × 6            │  Block class (Attention + FFN)       │
 │  5. Why It Works                     │  LayerNorm + residual in Block       │
 │  6. Final LayerNorm    (B, T, C)     │  GPT.ln_f                            │
 │  7. Output Projection  (B, T, V)     │  GPT.lm_head                         │
 │  8. Softmax            (B, T, V)     │  inside generate() / cross_entropy   │
 │  9. Loss / Sample                    │  inside forward() / generate()       │
 └──────────────────────────────────────┴──────────────────────────────────────┘

 Architecture summary:
   - 6 transformer blocks (n_layer = 6)
   - 6 attention heads per block (n_head = 6)
   - 384-dimensional embeddings (n_embd = 384)
   - 4,096-token vocabulary (our custom BPE tokenizer)
   - ~12.4 million parameters total

 Every non-obvious line has a comment. Read alongside the Episode 3 Shape Flow.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# All hyperparameters in one place. Change here, everything else adapts.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GPTConfig:
    # Vocabulary and sequence
    vocab_size: int = 4096   # number of tokens in our BPE tokenizer
    block_size: int = 256    # maximum sequence length (context window)

    # Model dimensions
    n_embd: int = 384        # embedding dimension — every token is 384 numbers
    n_layer: int = 6         # number of transformer blocks stacked
    n_head: int = 6          # attention heads per block (n_embd must be divisible by n_head)

    # Regularization
    dropout: float = 0.1     # randomly zero out 10% of activations during training


# ═══════════════════════════════════════════════════════════════════════════════
# EPISODE 3 — STEP 4 (PART A): "TALK"
# ═══════════════════════════════════════════════════════════════════════════════
#
# CausalSelfAttention — the "talk" phase.
#
# In the analogy: students turn around and read each other's cards.
# Each student forms three things from their card:
#
#   Q (Query) — "what kind of word would help me right now?"
#   K (Key)   — "here's what I am, in case anyone needs me"
#   V (Value) — "here's the actual content I'd share"
#
# Each student matches their Q against everyone's K. Whoever's K best matches
# wins more attention. The student then receives a weighted blend of everyone's
# V's, weighted by how well each K matched the Q.
#
# "Causal" means students can only look at earlier seats — never the future.
# This is what makes the model usable for text generation.
#
# Input:  (B, T, C) — the row of students with their current notes
# Output: (B, T, C) — same shape, but each student's notes now reflect what
#                    they learned from the conversation
# ═══════════════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        # n_embd must be divisible by n_head so we can split evenly across heads
        assert config.n_embd % config.n_head == 0, \
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"

        # ── Q, K, V projections ──────────────────────────────────────────────
        # Three separate linear transformations of the input — one to produce
        # Queries, one for Keys, one for Values. All three operate on the FULL
        # 384-dim vector. The split into 6 heads happens later via reshape.
        #
        #   W_q   "what am I looking for?"  — produces Q
        #   W_k   "here's what I am"        — produces K
        #   W_v   "here's what I'd share"   — produces V
        #
        # Note: many real implementations combine these into a single Linear
        # layer with output size (3 * n_embd) for a small speed gain, e.g.:
        #     self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # then split the result with .split(n_embd, dim=2). Mathematically
        # identical — just less readable on first encounter.
        self.W_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.W_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.W_v = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # ── Output projection ────────────────────────────────────────────────
        # After all heads have done their work, their outputs are concatenated
        # back into a 384-dim vector. This final layer learns which combinations
        # of head outputs are most useful.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # ── Dropout ──────────────────────────────────────────────────────────
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # ── Store config values for use in forward() ─────────────────────────
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # ── Causal mask ───────────────────────────────────────────────────────
        # A triangular matrix of 1s in the lower-left, 0s in the upper-right.
        # The 0s mark "future positions" — positions the current token cannot see.
        # Built in three clear steps:

        # Step 1 — make a square matrix of all 1s
        ones = torch.ones(config.block_size, config.block_size)

        # Step 2 — keep only the lower triangle (upper-right becomes 0s)
        mask = torch.tril(ones)
        # mask shape: (block_size, block_size)

        # Step 3 — add two leading dimensions so the mask can broadcast
        # across batch and head dimensions when we apply it later.
        # Final shape: (1, 1, block_size, block_size)
        mask = mask.view(1, 1, config.block_size, config.block_size)

        # Register as a buffer — it moves to GPU with the model but never
        # gets updated by the optimizer (it's not a learnable parameter).
        self.register_buffer("bias", mask)

    def forward(self, x):
        # x shape: (B, T, C) — batch, sequence length, embedding dim
        B, T, C = x.size()

        # ─────────────────────────────────────────────────────────────────────
        # Step 1 — Compute Q, K, V
        # ─────────────────────────────────────────────────────────────────────
        # Each student forms three things from their card.
        q = self.W_q(x)     # (B, T, C) — "what am I looking for?"
        k = self.W_k(x)     # (B, T, C) — "here's what I am"
        v = self.W_v(x)     # (B, T, C) — "here's what I'd share"

        # ─────────────────────────────────────────────────────────────────────
        # Step 2 — Split into heads
        # ─────────────────────────────────────────────────────────────────────
        # Each head gets head_size = (C // n_head) = 64 dimensions.
        # We do this in TWO steps for clarity:
        #
        # Step 2a — split the last dimension into (n_head, head_size)
        # Step 2b — move the head dimension next to the batch dimension
        #
        # After this, each head can be treated as an independent attention
        # operation — PyTorch handles all 6 heads in one matmul.
        head_size = C // self.n_head

        # Step 2a — split into heads
        q = q.view(B, T, self.n_head, head_size)
        k = k.view(B, T, self.n_head, head_size)
        v = v.view(B, T, self.n_head, head_size)
        # Shape now: (B, T, n_head, head_size)

        # Step 2b — move head dimension next to batch
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Shape now: (B, n_head, T, head_size)
        #
        # Many implementations write 2a + 2b as one chained line:
        #     q = q.view(B, T, n_head, head_size).transpose(1, 2)

        # ─────────────────────────────────────────────────────────────────────
        # Step 3 — Compute attention scores (Q · Kᵀ)
        # ─────────────────────────────────────────────────────────────────────
        # For each head, for each query position, compute the dot product
        # with every key position. This produces a (T, T) score matrix per head.
        #
        # Why divide by √head_size?
        # Dot products grow in magnitude as head_size increases. Large values
        # push softmax into its flat regions, making gradients tiny. Dividing
        # by √head_size keeps the variance stable regardless of head_size.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
        # att shape: (B, n_head, T, T)

        # ─────────────────────────────────────────────────────────────────────
        # Step 4 — Apply the causal mask
        # ─────────────────────────────────────────────────────────────────────
        # Wherever the mask is 0 (future positions), replace the score with -inf.
        # After softmax, -inf becomes exactly 0.0 — the student pays zero
        # attention to anyone sitting in a later seat.
        #
        # Why -inf and not -1000?
        # After softmax, -inf becomes exactly 0.0. A large negative number like
        # -1000 produces a tiny but nonzero probability — mathematically wrong.
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # ─────────────────────────────────────────────────────────────────────
        # Step 5 — Softmax: scores become attention weights
        # ─────────────────────────────────────────────────────────────────────
        # The weights now sum to 1 across the key dimension.
        # Higher score → higher weight → that position's V contributes more.
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # ─────────────────────────────────────────────────────────────────────
        # Step 6 — Weighted blend of Values
        # ─────────────────────────────────────────────────────────────────────
        # Each query position receives a weighted blend of all earlier value
        # vectors, weighted by the attention scores.
        y = att @ v
        # y shape: (B, n_head, T, head_size)

        # ─────────────────────────────────────────────────────────────────────
        # Step 7 — Reassemble heads
        # ─────────────────────────────────────────────────────────────────────
        # Undo the reshape from Step 2 — turn (B, n_head, T, head_size) back
        # into (B, T, C). Three explicit steps:

        # Step 7a — undo the transpose
        y = y.transpose(1, 2)
        # Shape: (B, T, n_head, head_size)

        # Step 7b — force a contiguous memory layout
        # transpose() doesn't actually move data in memory — it just changes
        # how PyTorch reads it. .view() requires contiguous memory, so we
        # have to force a copy here.
        y = y.contiguous()

        # Step 7c — merge n_head and head_size back into C
        y = y.view(B, T, C)
        # Shape: (B, T, C) — same as the input!
        #
        # Compact form: y = y.transpose(1, 2).contiguous().view(B, T, C)

        # ─────────────────────────────────────────────────────────────────────
        # Step 8 — Output projection
        # ─────────────────────────────────────────────────────────────────────
        # The concatenated head outputs are mixed through a learned linear
        # layer. The model learns which combinations of head outputs matter.
        y = self.resid_dropout(self.c_proj(y))

        return y  # shape: (B, T, C) — same shape as input


# ═══════════════════════════════════════════════════════════════════════════════
# EPISODE 3 — STEP 4 (PART B): "THINK"
# ═══════════════════════════════════════════════════════════════════════════════
#
# FeedForward — the "think" phase.
#
# In the analogy: each student processes their notes alone, in silence.
# No communication with other students.
#
# After attention, each student has a card with bits of context mixed in
# from other tokens. Now they transform those gathered notes into something
# more useful — three sub-steps:
#
#   1. Expand   — stretch 384 dims into 1,536 (4× wider) — many learned "lenses"
#   2. Filter   — apply a non-linearity (GELU) — keep positives, suppress negatives
#   3. Compress — squeeze back down to 384 dims, ready for the next round
#
# The non-linearity (step 2) is the most important. Without it, the whole
# network collapses into a single linear transformation and adds nothing.
#
# Most of the model's parameters live here — roughly 2× as many as in attention.
# Research suggests that factual knowledge is stored in feed-forward weights.
#
# Input:  (B, T, C) — each position processed independently
# Output: (B, T, C) — same shape, richer content
# ═══════════════════════════════════════════════════════════════════════════════

class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()

        # ── Expand: 384 → 1,536 ──────────────────────────────────────────────
        # The first linear layer projects the token's vector into a wider space.
        # Think of it as creating many "lenses" — different combinations of
        # the input — that the next step will then filter.
        # Why 4×? Empirically established in the original transformer paper.
        self.expand = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)

        # ── Filter: the non-linearity ──────────────────────────────────────────
        # GELU (Gaussian Error Linear Unit) is a smooth version of ReLU.
        # It keeps positive values and smoothly suppresses negatives.
        #
        # This is the MOST IMPORTANT step in the feed-forward.
        # Without a non-linearity here, the expand and compress layers would
        # collapse mathematically into a single linear transformation,
        # adding nothing useful to the network's expressive power.
        #
        # GPT-2 and later models use GELU. Earlier transformers used ReLU.
        self.activation = nn.GELU()

        # ── Compress: 1,536 → 384 ────────────────────────────────────────────
        # The second linear layer brings the wider representation back down
        # to the original n_embd size, ready for the next block.
        self.compress = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

        # Dropout for regularization — randomly zeros 10% of activations
        self.dropout = nn.Dropout(config.dropout)

        # Note: this whole class can also be written as one nn.Sequential:
        #     self.net = nn.Sequential(
        #         nn.Linear(n_embd, 4 * n_embd),
        #         nn.GELU(),
        #         nn.Linear(4 * n_embd, n_embd),
        #         nn.Dropout(dropout),
        #     )
        # We use named layers here for clarity.

    def forward(self, x):
        # x shape: (B, T, C)
        # Each position processed independently — no communication between tokens.

        x = self.expand(x)        # Step 1: (B, T, C)  →  (B, T, 4C)
        x = self.activation(x)    # Step 2: apply GELU element-wise (shape unchanged)
        x = self.compress(x)      # Step 3: (B, T, 4C) →  (B, T, C)
        x = self.dropout(x)       # Regularize: zero out 10% of values during training

        return x  # shape: (B, T, C) — same shape, richer content


# ═══════════════════════════════════════════════════════════════════════════════
# EPISODE 3 — STEPS 4 AND 5: ONE COMPLETE ROUND
# ═══════════════════════════════════════════════════════════════════════════════
#
# Block — one full round of "talk, then think."
#
# This wraps the two phases (attention + feed-forward) with the two helpers
# from Episode 3 Step 5 (Why It Works):
#
#   - LayerNorm before each phase  → "settle down, everyone"
#   - Residual around each phase   → "keep your original notes, add the new"
#
# These two scaffolding mechanisms are what make 6 stacked layers actually
# trainable. Without residuals, gradients fade as they flow back through
# the stack. Without LayerNorm, values drift to extreme magnitudes.
#
# Input:  (B, T, C)
# Output: (B, T, C) — same shape, so we can stack any number of these
# ═══════════════════════════════════════════════════════════════════════════════

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        # LayerNorm before attention (pre-norm arrangement).
        # The original 2017 "Attention Is All You Need" paper applied LayerNorm
        # AFTER each sub-layer. GPT-2 switched to BEFORE because it trains
        # more stably for deeper models. We use the GPT-2 style here.
        self.ln_1 = nn.LayerNorm(config.n_embd)

        # The attention sub-layer (the "talk" phase)
        self.attn = CausalSelfAttention(config)

        # LayerNorm before feed-forward
        self.ln_2 = nn.LayerNorm(config.n_embd)

        # The feed-forward sub-layer (the "think" phase)
        self.ffn = FeedForward(config)

    def forward(self, x):
        # ─────────────────────────────────────────────────────────────────────
        # Phase 1 — Talk (attention with residual)
        # ─────────────────────────────────────────────────────────────────────
        # The "+ x" is the residual connection — students keep their original
        # notes and ADD the new information learned from the conversation.
        # Without this, gradients fade as they flow backward through 6 layers.
        x = x + self.attn(self.ln_1(x))

        # ─────────────────────────────────────────────────────────────────────
        # Phase 2 — Think (feed-forward with residual)
        # ─────────────────────────────────────────────────────────────────────
        # Same pattern: normalize, process, add back the original.
        x = x + self.ffn(self.ln_2(x))

        return x  # shape: (B, T, C) — identical to input shape


# ═══════════════════════════════════════════════════════════════════════════════
# EPISODE 3 — STEPS 1 THROUGH 9: THE FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
#
# GPT — the full model.
#
# Combines everything into one nn.Module. Calling model(idx) runs through
# all 9 steps from Episode 3 in order:
#
#   Step 1 — Tokens                  (B, T)         input parameter
#   Step 2 — Token Embeddings        (B, T, C)      tok_emb lookup
#   Step 3 — + Position              (B, T, C)      pos_emb lookup + addition
#   Step 4 — Transformer Block × 6   (B, T, C)      6 rounds of "talk + think"
#   Step 5 — Why It Works                           handled inside each Block
#   Step 6 — Final LayerNorm         (B, T, C)      one last "settle down"
#   Step 7 — Output Projection       (B, T, V)      score every vocab token
#   Step 8 — Softmax                 (B, T, V)      handled inside cross_entropy
#   Step 9 — Loss / Sample                          forward() returns loss
#                                                   generate() samples a token
# ═══════════════════════════════════════════════════════════════════════════════

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ─────────────────────────────────────────────────────────────────────
        # Step 2 — Token embedding table
        # ─────────────────────────────────────────────────────────────────────
        # The "giant dictionary" — one entry per possible token.
        # Maps each token ID to a 384-number meaning vector.
        # Shape of the table: (vocab_size, n_embd) = (4096, 384)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)

        # ─────────────────────────────────────────────────────────────────────
        # Step 3 — Positional embedding table
        # ─────────────────────────────────────────────────────────────────────
        # The second dictionary, keyed by seat number.
        # Maps each position (0..block_size-1) to a 384-number signature.
        # Shape: (block_size, n_embd) = (256, 384)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        # Dropout applied to (token + position) embeddings before blocks
        self.drop = nn.Dropout(config.dropout)

        # ─────────────────────────────────────────────────────────────────────
        # Step 4 — Six transformer blocks
        # ─────────────────────────────────────────────────────────────────────
        # We need 6 transformer blocks stacked one after another.
        # Build them with an explicit for-loop and store in a ModuleList.
        # ModuleList tells PyTorch "these are sub-modules, register their parameters."
        self.blocks = nn.ModuleList()
        for _ in range(config.n_layer):
            self.blocks.append(Block(config))
        # self.blocks is now a list of 6 Block instances.
        #
        # Compact alternative: nn.Sequential(*[Block(config) for _ in range(n_layer)])
        # We use ModuleList here so the forward pass shows the loop explicitly.

        # ─────────────────────────────────────────────────────────────────────
        # Step 6 — Final LayerNorm
        # ─────────────────────────────────────────────────────────────────────
        # After 6 blocks of residual additions, values may have drifted to
        # large magnitudes. This is the last "settle down, everyone" before
        # the final classifier sees the data.
        self.ln_f = nn.LayerNorm(config.n_embd)

        # ─────────────────────────────────────────────────────────────────────
        # Step 7 — Output projection (the language model head)
        # ─────────────────────────────────────────────────────────────────────
        # Maps each token's 384-number vector to 4,096 logits — one score
        # per vocabulary token. Higher score = the model prefers that token.
        # bias=False because the embedding table provides bias implicitly.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # ── Weight tying ──────────────────────────────────────────────────────
        # The output projection and the token embedding table share the same
        # matrix. Same dictionary, used in both directions:
        #   - tok_emb:  token ID  →  meaning vector   (input side)
        #   - lm_head:  meaning vector  →  logits      (output side)
        # This trick saves ~1.6M parameters and improves perplexity.
        self.tok_emb.weight = self.lm_head.weight

        # ── Weight initialization ──────────────────────────────────────────────
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights with small Gaussian noise.
        Why not zero? Zero initialization makes all neurons in a layer identical,
        so gradients are identical, so they all update identically — the layer
        never breaks symmetry and never learns diverse features.
        Small random values break symmetry while keeping the initial outputs small.
        The 0.02 std matches GPT-2's initialization.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass through the full model — runs Episode 3's 9 steps.

        Args:
            idx:     token IDs,  shape (B, T)
            targets: next tokens, shape (B, T) — only needed during training

        Returns:
            logits: raw scores for each next token, shape (B, T, vocab_size)
            loss:   cross-entropy loss (None during generation)
        """
        # Step 1 already happened — `idx` IS the row of seated students.
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block_size {self.config.block_size}"

        # ─────────────────────────────────────────────────────────────────────
        # Step 2 — Token embeddings
        # ─────────────────────────────────────────────────────────────────────
        # Look up each token ID in the embedding table.
        # Each integer card-number → 384-number meaning vector.
        tok = self.tok_emb(idx)              # (B, T, C)

        # ─────────────────────────────────────────────────────────────────────
        # Step 3 — + Positional embeddings
        # ─────────────────────────────────────────────────────────────────────
        # Tell each student their seat number.
        # Build [0, 1, 2, ..., T-1] and look up each seat's vector.
        pos = torch.arange(T, device=idx.device)   # (T,)
        pos = self.pos_emb(pos)                    # (T, C) — broadcasts across batch

        # Add the two pieces together: each student now knows their word's
        # meaning AND their seat's signature, combined into one vector.
        x = self.drop(tok + pos)            # (B, T, C)

        # ─────────────────────────────────────────────────────────────────────
        # Step 4 — Six transformer blocks (talk + think × 6)
        # ─────────────────────────────────────────────────────────────────────
        # Six rounds of conversation. Each block takes (B, T, C) and returns
        # (B, T, C) — same shape, richer content.
        for block in self.blocks:
            x = block(x)
        # Compact form (with nn.Sequential): x = self.blocks(x)

        # ─────────────────────────────────────────────────────────────────────
        # Step 6 — Final LayerNorm
        # ─────────────────────────────────────────────────────────────────────
        # One last "settle down" before the classifier.
        x = self.ln_f(x)                    # (B, T, C)

        # ─────────────────────────────────────────────────────────────────────
        # Step 7 — Output projection → vocabulary scores (logits)
        # ─────────────────────────────────────────────────────────────────────
        # Each student writes a score for every possible next word.
        logits = self.lm_head(x)            # (B, T, vocab_size)

        # ─────────────────────────────────────────────────────────────────────
        # Steps 8 + 9 — Softmax + Loss (during training)
        # ─────────────────────────────────────────────────────────────────────
        # F.cross_entropy combines softmax (Step 8) and negative log-likelihood
        # (Step 9) into one numerically stable function.
        #
        # During generation (no targets), we skip the loss and let the
        # generate() method handle softmax + sampling separately.
        loss = None
        if targets is not None:
            # cross_entropy expects:
            #   - logits flattened to (N, vocab_size) where N = B*T
            #   - targets flattened to (N,)

            # Flatten logits from (B, T, vocab_size) to (B*T, vocab_size)
            flat_logits = logits.view(-1, logits.size(-1))

            # Flatten targets from (B, T) to (B*T,)
            flat_targets = targets.view(-1)

            # Compute the loss
            loss = F.cross_entropy(flat_logits, flat_targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressively generate new tokens after an initial prompt.

        This is Step 9 of Episode 3 — the generation use case.
        Each iteration runs the full 9-step pipeline, picks one token from
        the resulting probability distribution, appends it, and repeats.

        Args:
            idx:            prompt token IDs,  shape (B, T)
            max_new_tokens: how many tokens to generate
            temperature:    > 1.0 makes distribution flatter (more creative)
                            < 1.0 makes distribution sharper (more predictable)
            top_k:          if set, only sample from the top-k most likely tokens

        Returns:
            idx: original prompt + generated tokens, shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):

            # Crop to block_size if the sequence has grown too long
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                            else idx[:, -self.config.block_size:]

            # Run the forward pass — we only need the logits, not the loss
            logits, _ = self(idx_cond)

            # Take the logits at the LAST position only (the "next token" prediction)
            logits = logits[:, -1, :]        # (B, vocab_size)

            # Apply temperature — divide logits before softmax
            # High temperature → logits closer together → more uniform distribution
            # Low temperature  → logits farther apart   → more peaked distribution
            logits = logits / temperature

            # Apply top-k filtering — zero out everything outside the top k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Step 8 — Softmax: convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Step 9 — Sample one token from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append the sampled token — a new student joins the row
            idx = torch.cat((idx, idx_next), dim=1)              # (B, T+1)

        return idx

    def get_num_params(self, non_embedding=True):
        """
        Count the model's parameters.

        Args:
            non_embedding: if True, subtract the position embedding parameters.
                           These don't contribute to the language modelling task
                           and are typically excluded from the reported count.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pos_emb.weight.numel()
        return n_params


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# Run this file directly to verify the model instantiates and passes a
# forward pass without error.
# Expected loss: ~ln(4096) ≈ 8.32 (random initialization = uniform distribution)
# Expected parameters: ~12.4 million
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    config = GPTConfig()
    model = GPT(config)

    # Parameter count
    n_params = model.get_num_params()
    print(f"Parameters: {n_params:,}")

    # Dummy forward pass — runs all 9 steps from Episode 3
    B, T = 4, 64
    x = torch.randint(0, config.vocab_size, (B, T))
    logits, loss = model(x, x)

    print(f"Input shape:  {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss:         {loss.item():.4f}  (expect ~{math.log(config.vocab_size):.2f})")
    print("Sanity check passed." if abs(loss.item() - math.log(config.vocab_size)) < 1.0
          else "WARNING: loss looks wrong.")