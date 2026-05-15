# Build a GPT From Scratch on a 6 GB GPU

A YouTube series and accompanying code where we build a small transformer language model from absolute zero — empty folder to a model that writes children's stories — on consumer hardware.

**▶ [Watch the YouTube playlist](https://www.youtube.com/playlist?list=PLvmpljk9TE2v4LbSrQxmnbOKkLa0Mw1oD)**

---

## What We're Building

A 12-million-parameter GPT, hand-coded line by line, trained on TinyStories. Runs on a 6 GB GPU in about 2 hours. By the end, it writes coherent short stories.

No `transformers` library. No pre-built model classes. Every component built from PyTorch primitives so you understand exactly how transformers work.

## Who This Is For

- Developers who use ChatGPT and want to actually understand how it works under the hood
- ML beginners who want a project that goes deeper than "fine-tune this Hugging Face model"
- Anyone with a consumer GPU who's been told "you can't train an LLM" — you can, just a small one

You need basic Python familiarity. No ML background required; we build up from the ground.

---

## The Episodes

| # | Episode | Status | Code |
|---|---------|--------|------|
| 01 | [Setup — environment and GPU verification](./episode-01-setup/) | ✅ Released | `GPUVerification.py` |
| 02 | [Data and tokenization — choosing what to feed the model](./episode-02-data-and-tokenizer/) | ✅ Released  | `train_tokenizer.py`, `prepare_data.py` |
| 03 | [What is a transformer, really? (theory only)](./episode-03-architecture-theory/) | ✅ Released 🎬 Recording | — |
| 04 | [Building the model — attention, FFN, blocks](./episode-04-building-the-model/) | ✅ Released  | `model.py` |
| 05 | [Training — the optimization story](./episode-05-training/) | ✅ Released  | `train.py` |
| 06 | [Generation & Deployment — sampling and a web UI](./episode-07-generation/) | 📝 Planned | `generate.py`, `server.py` |

> **Status legend:** ✅ Released · 🎬 Recording · 📝 Planned · 🚧 Coming soon

---

## Quick Start

If you want to dive in:

1. **Start with [Episode 01](./episode-01-setup/)** to set up your environment.
2. Follow the episodes in order — each builds on the previous.

---

## Project Structure

```
build-gpt-from-scratch/
├── README.md                          ← you are here
├── episode-01-setup/                  ← env setup, GPU verification
├── episode-02-data-and-tokenizer/     ← TinyStories + custom BPE
├── episode-03-architecture-theory/    ← intuition, no code
├── episode-04-building-the-model/     ← model.py from scratch
├── episode-05-training/               ← train.py + optimization
├── episode-06-watching-it-learn/      ← reading training curves
├── episode-07-generation/             ← inference + web UI
├── episode-08-deploy-huggingface/     ← share with the world
└── final-project/                     ← complete combined codebase
```

Each episode folder contains its own README with the full written walkthrough of that video, plus the code files built during it.

---

## Hardware Requirements

| Component | Recommended | Minimum |
|-----------|-------------|---------|
| GPU | NVIDIA, 6 GB+ VRAM | 4 GB VRAM (smaller model needed) |
| RAM | 16 GB | 8 GB |
| Storage | 10 GB free | 5 GB free |
| OS | Windows 10/11, Linux, macOS | any of the above |

**No GPU?** You can train on CPU, but a 2-hour training run becomes a multi-day ordeal. Renting a cloud GPU (RunPod, Vast.ai, Google Colab Pro) is the practical alternative.

**AMD GPU?** Possible via ROCm on Linux. Limited support on Windows. Use Apple Silicon's `mps` backend if on a Mac. The principles are identical; only the install command and `device` string change.

---

## What You'll Understand by the End

At a mechanistic level — not "I've heard the term," but "I could explain it to a friend":

- How tokens become vectors (embeddings)
- How attention works (queries, keys, values, masks)
- Why transformers stack (layers as hierarchical understanding)
- How training actually works (gradients, AdamW, learning rate schedules)
- Why we hold out validation data (generalization vs memorization)
- How sampling produces text (top-k, temperature, EOT handling)
- How to deploy a model so others can use it

This is the same architecture as GPT-2, GPT-3, GPT-4, LLaMA, and Mistral. They differ only in scale and minor modernizations. You'll have built the core.

---

## Questions and Discussion

- **Found a bug?** Open an issue on this repo.
- **Stuck during a video?** Comment on the YouTube video — I read every comment.
- **Have an idea for a future episode?** Issues welcome there too.

---

## License

Code: MIT. Educational content: free to share with attribution.

If this series helped you understand something that felt impenetrable before, the best thank-you is to share it with someone else who'd benefit.

---

## Useful References

- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) — the reference implementation that inspired this project
- [Karpathy's "Let's build GPT" video](https://www.youtube.com/watch?v=kCc8FmEb1nY) — single best video on the topic
- [TinyStories paper](https://arxiv.org/abs/2305.07759) — Microsoft Research, 2023
- [PyTorch Get Started](https://pytorch.org/get-started/locally/) — always-current install commands
