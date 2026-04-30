# LLM From Scratch — Setup Guide

Build a small GPT-style language model from scratch on your own machine. This README covers the **setup phase only** — installing Python, `uv`, creating a virtual environment, and verifying your GPU is ready for training.

---

## Prerequisites

- **Windows 10/11** (these instructions are Windows-specific; Linux/macOS users should adapt commands)
- **NVIDIA GPU** with at least 4GB VRAM (6GB recommended)
- **NVIDIA driver** installed and up to date
- **~5GB of free disk space** for Python, libraries, and PyTorch

To confirm your GPU is detected, open PowerShell and run:

```powershell
nvidia-smi
```

You should see a table with your GPU listed and a "CUDA Version" in the top-right corner. If `nvidia-smi` is not found, install the latest NVIDIA driver from [nvidia.com/Download](https://www.nvidia.com/Download/index.aspx) before continuing.

> **Important note about `nvidia-smi`'s "CUDA Version":** This number is the *maximum* CUDA version your driver supports — not what is installed. PyTorch bundles its own CUDA runtime inside the wheel. As long as your driver's max ≥ PyTorch's CUDA version, everything works.

---

## Step 1 — Install Python 3.11

We use Python 3.11 specifically. Newer versions (3.13) sometimes lack ML library wheels; older versions are slower.

1. Open your browser and go to:
   [https://www.python.org/downloads/release/python-3119/](https://www.python.org/downloads/release/python-3119/)

2. Scroll to the bottom and download **"Windows installer (64-bit)"**.

3. Open the downloaded `.exe`.

4. **On the first installer screen, check the box** ✅ **"Add python.exe to PATH"** — this is critical.

5. Click **"Install Now"** and wait for it to finish.

6. Open a **new** PowerShell window (must be fresh to pick up the updated PATH).

7. Verify:

   ```powershell
   py -3.11 --version
   ```

   You should see `Python 3.11.9`.

---

## Step 2 — Install `uv`

`uv` is a fast Python package installer. It's a drop-in replacement for `pip` and is significantly faster, especially for large packages like PyTorch.

1. In PowerShell, run:

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Wait for it to finish.

3. Close PowerShell completely and open a new window.

4. Verify:

   ```powershell
   uv --version
   ```

   You should see something like `uv 0.x.x`.

---

## Step 3 — Create a Project Folder and Virtual Environment

A virtual environment isolates your project's libraries from the rest of your system.

1. Choose a location and create your project folder. Example:

   ```powershell
   cd M:\
   mkdir my-llm-project
   cd my-llm-project
   ```

2. Create the virtual environment using Python 3.11:

   ```powershell
   uv venv llm-env --python 3.11
   ```

3. Activate the environment:

   ```powershell
   .\llm-env\Scripts\activate
   ```

4. Confirm activation. Your prompt should show `(llm-env)` at the start:

   ```
   (llm-env) PS M:\my-llm-project>
   ```

   Anything you `uv pip install` from now on goes into this isolated environment.

> If you see an error about *"running scripts is disabled on this system"*, run PowerShell as Administrator once and execute:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

---

## Step 4 — Install PyTorch with CUDA Support

The trick: regular `pip install torch` gives you a CPU-only build. We need to explicitly request the CUDA build.

For most modern NVIDIA GPUs (driver supports CUDA 12.x or higher), use:

```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

This downloads ~2.5GB. Be patient.

If your driver only supports CUDA 11.8 (older systems), use `cu118` instead:

```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> **Always check [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)** for the current correct install command. The selector on that page reflects live support.

---

## Step 5 — Verify the Installation

Run GPUVerification.py:

```powershell
python GPUVerification.py
```

You should see something like:

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
 Your GPU is suitable for educational LLM training (10-30M params).
============================================================
```

If you see this output, **your environment is ready for training**.

---

## Troubleshooting

**Script runs but prints nothing**
The file may be empty or have a save problem. Check it:
```powershell
type GPUVerification.py
```
If the contents look wrong, re-save the script.

**`CUDA available: False`**
You likely got the CPU-only PyTorch wheel. Fix:
```powershell
uv pip uninstall torch torchvision
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**`ImportError: No module named torch`**
The virtual environment is not active, or PyTorch was installed in a different environment. Make sure you see `(llm-env)` in your prompt before running scripts.

**`nvidia-smi` not found**
NVIDIA driver is not installed or not in PATH. Download the latest driver from [nvidia.com/Download](https://www.nvidia.com/Download/index.aspx) and reboot.

---

## What's Next

Once verification passes, you're ready for the next phase: **building the tokenizer and preparing the TinyStories dataset**. That's covered in the next part of the series.

## Compute Capability Reference

If you're curious what your GPU's "compute capability" means and which generation it belongs to:

| Compute Capability | Architecture | GPU Family |
|--------------------|---------------|------------|
| 6.1 | Pascal | GTX 10-series |
| 7.0 | Volta | Tesla V100 |
| 7.5 | Turing | GTX 16-series, RTX 20-series |
| 8.0 | Ampere | A100 |
| 8.6 | Ampere | RTX 30-series |
| 8.9 | Ada Lovelace | RTX 40-series |
| 9.0 | Hopper | H100 |
| 10.0 | Blackwell | RTX 50-series |

Higher numbers = newer hardware = more ML features supported.

- **≥ 7.0** unlocks Tensor Cores (huge speedup for ML)
- **≥ 8.0** unlocks BF16 training and Flash Attention
- **≥ 8.9** unlocks FP8 training

Our project does not require any features beyond 7.0, so any modern NVIDIA card works.
