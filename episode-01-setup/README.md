# Episode 01 — Setup

The first episode in the [Build a GPT From Scratch on a 6 GB GPU](../README.md) series.

**▶ [Watch the video](https://www.youtube.com/playlist?list=PLvmpljk9TE2v4LbSrQxmnbOKkLa0Mw1oD)**

---

## What We Build in This Episode

By the end, you'll have a working environment ready for the rest of the series:

- Python 3.11 in a clean isolated virtual environment
- PyTorch installed with CUDA support for your GPU
- A verification script that confirms your full stack works and tells you which advanced ML features your specific GPU supports

This is the foundation everything else sits on. Once it's solid, every subsequent episode focuses on building the actual model.

---

## Hardware Requirements

| Component | Recommended | Minimum |
|-----------|-------------|---------|
| GPU | NVIDIA, 6 GB+ VRAM | 4 GB VRAM (smaller model needed) |
| RAM | 16 GB | 8 GB |
| Storage | 10 GB free | 5 GB free |
| OS | Windows 10/11, Linux, macOS | any of the above |

**No GPU?** You can technically train on CPU, but a 2-hour training run becomes a multi-day ordeal. Renting a cloud GPU (RunPod, Vast.ai, Google Colab Pro) is the practical alternative.

**AMD GPU?** Possible via ROCm on Linux. Limited support on Windows. Use Apple Silicon's `mps` backend if on a Mac. The principles are identical; only the install command and `device` string change.

---

## Confirm Your GPU Is Detected

Open PowerShell and run:

```powershell
nvidia-smi
```

You should see a table listing your GPU and a "CUDA Version" number in the top-right corner. If the command isn't found, install the latest NVIDIA driver from [nvidia.com/Download](https://www.nvidia.com/Download/index.aspx) and reboot.

### Reading `nvidia-smi`'s "CUDA Version" Correctly

This is the single most misunderstood thing in the entire setup process.

**The "CUDA Version" shown in `nvidia-smi` is NOT what is installed.** It is the *maximum* CUDA version your driver can support.

Think of it like a power outlet:

- Your driver says: "I can supply up to CUDA 13.1."
- That doesn't mean a CUDA 13.1 toolkit is installed.
- You can install any PyTorch built for CUDA ≤ 13.1.

PyTorch bundles its own CUDA runtime inside its wheel. Your driver simply needs to support a version equal to or higher than what PyTorch was built against. Newer drivers are backward compatible with older CUDA runtimes.

**Practical translation:**

| Driver max CUDA | Recommended PyTorch wheel |
|-----------------|---------------------------|
| 12.4, 13.x | `cu121` (the safe default) |
| 12.1–12.3 | `cu121` |
| 11.8–12.0 | `cu118` |
| Below 11.8 | Update your driver — too old |

---

## Step 1 — Install Python 3.11

We deliberately use Python 3.11, not the latest version.

### Why 3.11 Specifically

| Python version | ML library support | Recommendation |
|----------------|-------------------|----------------|
| 3.13 (latest) | Wheels often missing for new ML libraries | Avoid for now |
| 3.12 | Excellent | Fine alternative |
| **3.11** | **Mature, fastest, fully supported** | **Use this** |
| 3.10 | Solid but slower than 3.11 | Acceptable |
| 3.9 or older | Reaching end-of-life | Avoid |

Python 3.11 hit a sweet spot: significantly faster than 3.10 (about 25% interpreter performance improvement), and now mature enough that every ML library has battle-tested wheels for it. Supported through October 2027.

### Installation

1. Open your browser and visit:
   [python.org/downloads/release/python-3119](https://www.python.org/downloads/release/python-3119/)

2. Scroll to the bottom and download **"Windows installer (64-bit)"**.

3. Open the downloaded `.exe`.

4. **Critical**: on the first installer screen, check the box ✅ **"Add python.exe to PATH"**.

5. Click **"Install Now"** and wait.

6. Open a **new** PowerShell window (so it picks up the updated PATH).

7. Verify:

   ```powershell
   py -3.11 --version
   ```

   Expected: `Python 3.11.9` (or similar 3.11.x).

You can have multiple Python versions installed simultaneously — they don't conflict.

---

## Step 2 — Install `uv`

We use `uv` instead of plain `pip`. It's a dramatically faster Python package installer written in Rust.

### Why `uv` Over the Alternatives

| Tool | Speed | Why we picked / passed |
|------|-------|------------------------|
| `pip` + `venv` | Slow | Built into Python; works fine but slow |
| `conda` | Slow + heavy (3 GB install) | Was the standard 2015–2022; environment resolution has gotten painful |
| **`uv`** | **10–100× faster** | **Drop-in replacement for pip; modern default** |

For installing PyTorch (~2.5 GB download), `uv` saves real time. Commands are nearly identical to `pip`, so there's no learning curve.

If you prefer plain pip, substitute `pip install` for `uv pip install` everywhere — it works the same.

### Installation

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then close PowerShell completely and open a new window. Verify:

```powershell
uv --version
```

Expected: `uv 0.x.x`.

---

## Step 3 — Create Project Folder and Virtual Environment

A virtual environment isolates this project's libraries from your system Python. You can break it, delete it, recreate it — your system stays clean.

This is non-negotiable for ML work.

```powershell
cd M:\
mkdir my-llm-project
cd my-llm-project
uv venv llm-env --python 3.11
.\llm-env\Scripts\activate
```

Your prompt should now show `(llm-env)`:

```
(llm-env) PS M:\my-llm-project>
```

If you see *"running scripts is disabled on this system"*, run this once as Administrator:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

---

## Step 4 — Install PyTorch with CUDA Support

This is where most beginners trip up.

### The Trap

Plain `pip install torch` gives you the **CPU-only** wheel. It works, but `torch.cuda.is_available()` will return `False` and training will be 50–100× slower than expected.

You must explicitly request the CUDA wheel via `--index-url`.

### The Command

```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

This downloads ~2.5 GB. Be patient.

For older systems with CUDA 11.8 only:

```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Always check [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for the current install command — the selector there reflects live support.

### Why PyTorch?

| Library | Why we picked / passed |
|---------|------------------------|
| **PyTorch** | **Define-by-run, every modern LLM paper uses it, debugging is trivial** |
| TensorFlow / Keras | The ecosystem for LLM research has consolidated around PyTorch |
| JAX | Excellent but functional style adds cognitive overhead for beginners |

---

## Step 5 — Verify the Installation

The verification script ([`GPUVerification.py`](./GPUVerification.py)) does three things:

1. Confirms your full Python + PyTorch + CUDA stack is aligned
2. Tells you which advanced ML features your specific GPU supports
3. Runs a real GPU computation to confirm the pipeline actually works

### Why Verify with a Real Computation

`torch.cuda.is_available()` only checks that PyTorch *recognizes* a GPU. The matrix multiplication confirms the full pipeline — driver, CUDA runtime, PyTorch — is actually working end-to-end. Catches subtle environment issues before you waste hours into a training run.

### Run It

```powershell
python GPUVerification.py
```

### Expected Output

On a GTX 1660 SUPER, you should see:

```
============================================================
 PyTorch & GPU Verification
============================================================

Python version : 3.11.9
PyTorch version: 2.5.1+cu121
CUDA available : True

CUDA runtime  : 12.1
GPU            : NVIDIA GeForce GTX 1660 SUPER
Total VRAM     : 6.44 GB
Compute capability: 7.5
  Architecture: Turing (GTX 16-series, RTX 20-series)
  Tensor Cores; no BF16

Features supported on this GPU:
  FP16 mixed precision      yes          halves activation memory
  Tensor Cores              yes          huge speedup on matmul
  BF16 training             no           use FP16 instead
  Flash Attention           no           use standard attention
  FP8 training              no           Hopper or newer required

Running a real computation on the GPU...
  Matrix multiply succeeded. Output shape: torch.Size([1000, 1000])
  Memory allocated: 8.0 MB / 6438 MB total

============================================================
 Your GPU is suitable for educational LLM training (10-30 M params).
============================================================
```

If you see this output, **your environment is ready**.

---

## Understanding the Output

### The `+cu121` Suffix on PyTorch Version

`PyTorch version: 2.5.1+cu121` means:

- `2.5.1` is the PyTorch version itself
- `+cu121` confirms it's the CUDA 12.1 build (not CPU-only)

**If you see `2.5.1` with no suffix**, you accidentally got the CPU-only wheel. Fix it by uninstalling and reinstalling with the explicit `--index-url`.

### Compute Capability — The Hardware Generation

Every NVIDIA GPU has a compute capability number indicating its hardware generation. Higher = newer = more features.

| Compute Capability | Architecture | Examples | Key features |
|--------------------|--------------|----------|--------------|
| 6.1 | Pascal | GTX 10-series | FP16 in software |
| 7.0 | Volta | Tesla V100 | First Tensor Cores |
| **7.5** | **Turing** | **GTX 16-series, RTX 20-series** | **Tensor Cores** |
| 8.0 | Ampere | A100 | BF16, Flash Attention |
| 8.6 | Ampere | RTX 30-series | BF16, Flash Attention |
| 8.9 | Ada Lovelace | RTX 40-series | FP8 |
| 9.0 | Hopper | H100 | Cutting-edge FP8 |
| 10.0 | Blackwell | RTX 50-series | FP4 inference |

For our project, anything ≥ 7.0 is comfortable. The GTX 1660 (compute capability 7.5) is exactly what we use.

**Important hardware limits:**
- **BF16** (modern training-friendly float format) requires compute capability **8.0+**. No software trick changes this.
- **Flash Attention** requires compute capability **8.0+**. Without it, attention memory scales as `seq_length²`.

For our small model on a GTX 1660, we don't need either. We use FP32 for stability.

---

## Why FP32, Not FP16, in Episode 1

You'll see "FP16 mixed precision: yes" in the verification output. Why aren't we using it?

Mixed precision is faster and uses less memory, but it requires careful handling. Tiny gradients can underflow to zero in FP16, silently breaking training. The fix (loss scaling) is built into PyTorch's `GradScaler`, but adds complexity.

For a first project, FP32 is the safe choice. Fight one battle at a time. Once your training works in FP32, switching to mixed precision is a 5-line change. We may add it in a later episode.

---

## Troubleshooting

### "Script runs but prints nothing"

The file may be empty or have a save problem. Verify:

```powershell
type GPUVerification.py
```

If contents look wrong, re-save the script.

### "CUDA available: False"

You got the CPU-only PyTorch wheel:

```powershell
uv pip uninstall torch torchvision
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### "ImportError: No module named torch"

Virtual environment not active, or PyTorch was installed in a different environment. Make sure `(llm-env)` shows in your prompt.

### "nvidia-smi not found"

NVIDIA driver not installed or not in PATH. Download latest from [nvidia.com/Download](https://www.nvidia.com/Download/index.aspx) and reboot.

### "I see WARNING about driver too old"

Update your driver, or pick an older PyTorch wheel (`cu118` instead of `cu121`).

### "I'm on Linux"

Mostly the same. Replace PowerShell with Bash equivalents (`source llm-env/bin/activate` instead of `.\llm-env\Scripts\activate`). The `uv` install command is different — see [docs.astral.sh/uv](https://docs.astral.sh/uv/).

### "I'm on macOS"

PyTorch supports Apple Silicon via the `mps` backend. Install with `pip install torch torchvision` (no `--index-url` — there's no CUDA on Mac). In any code that says `'cuda'`, use `'mps'` instead.

### "I'm on Windows with an AMD GPU"

PyTorch supports AMD via ROCm, but Windows ROCm support is very recent and limited. Use WSL2 (Windows Subsystem for Linux) and install ROCm there. Many older AMD cards aren't supported.

---

## What We Set Up, In One Sentence

Python 3.11 in an isolated virtual environment, PyTorch with the correct CUDA build, and a verified GPU pipeline. Everything else in the series builds on this.

---

## What's Next: Episode 02

In [Episode 02](../episode-02-data-and-tokenizer/), we make three big decisions:

1. **How big a model can we actually train?** (Spoiler: not 1 billion parameters. We work out exactly why.)
2. **What dataset should we feed it?** (Spoiler: TinyStories — engineered specifically for tiny models.)
3. **How do we convert text into numbers a network can read?** (Spoiler: Byte-Pair Encoding, with a custom tokenizer.)

By the end of Episode 02, you'll have a 970 MB binary file of token IDs ready to train on.

---

## References

- [PyTorch Get Started](https://pytorch.org/get-started/locally/) — always-current install commands
- [Python 3.11 Downloads](https://www.python.org/downloads/release/python-3119/) — direct installer
- [uv Documentation](https://docs.astral.sh/uv/) — modern Python package manager
