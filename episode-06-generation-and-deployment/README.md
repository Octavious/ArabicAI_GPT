# Episode 06 — Generation & Deployment

The sixth episode in the [Build a GPT From Scratch on a 6 GB GPU](../README.md) series.

**▶ [Watch the video](https://www.youtube.com/playlist?list=PLvmpljk9TE2v4LbSrQxmnbOKkLa0Mw1oD)**

---

## What This Episode Is

Episode 5 trained the model. This episode uses it.

We take the saved `checkpoint.pt` and run **inference** — the model reads a prompt, predicts the next token, appends it, and repeats until it has written a story. First from the terminal (`generate.py`), then through a small web playground (`generate_ui.html` + `server.py`).

The training loop is gone. No loss, no backward pass, no optimizer. Just load weights, sample tokens, decode text.

---

## Prerequisites

- [Episode 05](../episode-05-training/) finished — you need a trained `checkpoint.pt`
- `model.py` and `tinystories_tokenizer.json` in this folder (copied from Episodes 4 and 2, or already present here)
- Python environment with `torch` from [Episode 01](../episode-01-setup/)

### Bring your checkpoint

`checkpoint.pt` is **not** in the repo (it's ~140 MB and gitignored). Copy it from Episode 5 after training:

```powershell
copy ..\episode-05-training\checkpoint.pt .
```

To get the tokenizer file, copy it from Episode 2:

```powershell
copy ..\episode-02-tokenization\tinystories_tokenizer.json .
```

All of these files must sit in the **same folder** when you run the scripts:

```
episode-06-generation-and-deployment/
├── model.py
├── tinystories_tokenizer.json   ← from Episode 2 (you copy this in)
├── checkpoint.pt                ← from Episode 5 (you copy this in)
├── generate.py
├── server.py
├── generate_ui.html
└── hugging-face/                ← upload to Hub + Gradio Space (see below)
```

---

## Files in This Episode

| File | What it is |
|---|---|
| `generate.py` | Terminal generator — prompt in, story out |
| `server.py` | Flask API that loads the model and serves `/generate` |
| `generate_ui.html` | Browser playground — sliders for temperature, top-k, max tokens |
| `inference_utils.py` | Story-boundary helper when the model does not sample `<\|endoftext\|>` |
| `model.py` | Same architecture as Episode 4 (includes `generate()` with EOS stopping) |
| `tinystories_tokenizer.json` | Same BPE tokenizer as Episode 2 |
| [`hugging-face/`](hugging-face/README.md) | Upload script, model card, and Gradio Space for Hugging Face |

The episode flows: **terminal first → web server second → Hugging Face (optional).**

---

## Running the Web UI (`generate_ui.html`)

**You cannot use the playground by double-clicking the HTML file alone.** The page talks to a Python backend over HTTP (`/info` and `/generate`). Without the server, the status dot stays red and generation fails.

### What you need

| Requirement | Why |
|---|---|
| `checkpoint.pt` in this folder | Trained weights (~142 MB) |
| `model.py` | Architecture the checkpoint was saved with |
| `tinystories_tokenizer.json` | Encode prompts / decode output |
| `torch` | Loads and runs the model (CUDA if available) |
| `flask` + `flask-cors` | `server.py` dependencies |
| `tokenizers` | Loads the BPE tokenizer file |

### Install dependencies

```powershell
uv pip install flask flask-cors tokenizers
```

(`torch` should already be installed from Episode 01.)

### Start the server

```powershell
cd episode-06-generation-and-deployment
python server.py
```

You should see the checkpoint step and val loss printed, then:

```
Server ready. Open http://127.0.0.1:5000 in your browser.
```

### Open the UI

Go to **http://127.0.0.1:5000** in your browser.

The server serves `generate_ui.html` at the root URL. The green status dot means `/info` succeeded and the model is loaded.

**Optional:** you can open `generate_ui.html` directly from disk, but you still must have `python server.py` running on port 5000 — the page calls `http://localhost:5000` when not served from the server itself.

### Using the playground

- Type or pick a **prompt** (e.g. `Once upon a time`)
- Adjust **temperature** (lower = safer, higher = more random)
- Adjust **top-k** (restrict sampling to the K most likely tokens; `0` = off)
- Set **max tokens** (length cap)
- **Stop at story end** — when checked, generation stops at the `<|endoftext|>` token so you get one complete story
- Click **Generate story** (or Ctrl/Cmd+Enter)

Each request runs inference on the GPU (or CPU). First generation after startup may take a few seconds while CUDA warms up.

---

## Running from the Terminal (`generate.py`)

No web server needed — just the checkpoint, tokenizer, and model:

```powershell
python generate.py
python generate.py "Lily found a"
python generate.py "Once upon a time" --tokens 200 --temperature 0.8 --top-k 40
```

| Flag | Default | Meaning |
|---|---|---|
| `prompt` | `Once upon a time` | Starting text |
| `--tokens` | `200` | Max new tokens to generate |
| `--temperature` | `0.8` | Sampling temperature |
| `--top-k` | `40` | Top-k filtering |
| `--no-stop` | off | Keep going past `<|endoftext|>` |

---

## How It Works

### Loading the checkpoint

```python
checkpoint = torch.load("checkpoint.pt", map_location=DEVICE, weights_only=False)
config = checkpoint['config']
model = GPT(config)
model.load_state_dict(checkpoint['model'])
model.eval()
```

The checkpoint dict contains everything `train.py` saved: weights, config, training step, val loss, and optimizer state (we ignore the optimizer at inference time).

### The generation loop

`model.generate()` runs autoregressively:

1. Forward pass on the current token sequence
2. Take logits at the last position
3. Apply temperature and optional top-k
4. Sample one token from the distribution
5. Append it and repeat

If `eos_token_id` is set (our `<|endoftext|>` token, ID 0), the loop stops early when the model samples that token.

**Why you might still see a second story:** older training runs used `prepare_data.py` without inserting `<|endoftext|>` between stories, so the model was never taught to emit token 0. With **Stop at story end** checked, the server also cuts before a repeated opener like `!"Once upon a time` (see `inference_utils.py`). Re-run `prepare_data.py` (now appends EOS per story) and retrain for true EOS stopping.

### What the server adds

`server.py` wraps the same logic in two HTTP endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serves `generate_ui.html` |
| `/info` | GET | Returns step, val loss, device — UI status line |
| `/generate` | POST | JSON in → generated story out |

The model stays loaded in memory between requests so you are not reloading 142 MB on every click.

---

## Troubleshooting

### Red dot: "server unreachable"

`server.py` is not running, or something else is using port 5000. Start the server and refresh the page.

### `FileNotFoundError: checkpoint.pt`

Copy the file from `episode-05-training/` after training finishes, or point `CHECKPOINT_PATH` in `server.py` / `generate.py` at your copy.

### `TypeError: generate() got an unexpected keyword argument 'eos_token_id'`

Your `model.py` is missing the EOS early-stop update from this episode. Use the `model.py` in this folder (or add `eos_token_id=None` to `GPT.generate()` and break when that token is sampled).

### Generation is very slow on CPU

Expected. A 12M-parameter model is small, but autoregressive sampling runs one forward pass per token. Use CUDA if available; otherwise lower **max tokens** in the UI.

### Garbled output

Almost always a **tokenizer mismatch** — the checkpoint must have been trained with the same `tinystories_tokenizer.json` you load at inference time.

---

## Deploy to Hugging Face

To share the model publicly — upload weights to the Hub and host a browser demo — follow the guide in **[`hugging-face/README.md`](hugging-face/README.md)**.

That folder contains:

| Path | Purpose |
|------|---------|
| [`hugging-face/model-repo/upload_model.py`](hugging-face/model-repo/upload_model.py) | Push `checkpoint.pt`, tokenizer, and model card to a Model repo |
| [`hugging-face/space/`](hugging-face/space/) | Gradio Space (`app.py`) that loads your model from the Hub |

Quick start after `huggingface-cli login`:

```powershell
cd hugging-face\model-repo
python upload_model.py --repo-id YOUR_USERNAME/tinystories-gpt
```

Then create a Space, set `HF_MODEL_REPO` to the same repo id, and push the files in `hugging-face/space/`. Full steps are in the [Hugging Face guide](hugging-face/README.md).

---

## What's Next

You now have a trained model you can prompt from the terminal, a local browser UI, or a public Hugging Face Space. For Hub upload and Space deployment, use [`hugging-face/README.md`](hugging-face/README.md).

---

## References

- [Andrej Karpathy's nanoGPT `sample.py`](https://github.com/karpathy/nanoGPT/blob/master/sample.py) — minimal reference for autoregressive sampling
- [Flask quickstart](https://flask.palletsprojects.com/en/latest/quickstart/) — the web framework used in `server.py`
