"""
GPU & PyTorch verification script.

Run once after installation to confirm everything works.
Prints your full stack and tells you which advanced features
your hardware supports.
"""

import torch
import sys


# ---------- Compute capability reference ----------
COMPUTE_CAPABILITY_INFO = {
    # (major, minor): (architecture name, GPU family, feature notes)
    (3, 5): ("Kepler", "GTX 700-series, Tesla K-series", "Very old; PyTorch dropped support after 2.0"),
    (3, 7): ("Kepler", "Tesla K80", "Very old; minimal modern support"),
    (5, 0): ("Maxwell", "GTX 750 / 750 Ti", "Old; supported but no modern features"),
    (5, 2): ("Maxwell", "GTX 9-series, Titan X", "Old; supported but slow for ML"),
    (6, 0): ("Pascal", "Tesla P100", "Datacenter; FP16 supported"),
    (6, 1): ("Pascal", "GTX 10-series, Titan Xp", "Common gaming card; FP16 in software"),
    (7, 0): ("Volta", "Tesla V100", "First Tensor Cores; major ML jump"),
    (7, 5): ("Turing", "GTX 16-series, RTX 20-series", "Tensor Cores; no BF16; what most hobbyists have"),
    (8, 0): ("Ampere", "A100", "BF16 native; Flash Attention supported"),
    (8, 6): ("Ampere", "RTX 30-series, A40", "BF16 native; Flash Attention supported"),
    (8, 9): ("Ada Lovelace", "RTX 40-series, L40", "Latest BF16/FP8; great for ML"),
    (9, 0): ("Hopper", "H100, H200", "FP8 training; cutting edge datacenter"),
    (10, 0): ("Blackwell", "RTX 50-series, B100", "Newest; FP4 inference, FP8 training"),
}


def lookup_capability(major, minor):
    """Return architecture info for a given compute capability, or a generic note."""
    key = (major, minor)
    if key in COMPUTE_CAPABILITY_INFO:
        return COMPUTE_CAPABILITY_INFO[key]
    return ("Unknown", "Unrecognized GPU", "Compute capability not in lookup table")


def feature_support(major, minor):
    """Tell the user what modern ML features their card supports."""
    cc = major + minor / 10  # e.g. 7.5
    features = []

    # FP16 (half precision)
    if cc >= 5.3:
        features.append(("FP16 mixed precision", True, "halves activation memory, faster matmul"))
    else:
        features.append(("FP16 mixed precision", False, "GPU too old"))

    # Tensor Cores (specialized matmul hardware)
    if cc >= 7.0:
        features.append(("Tensor Cores", True, "huge speedup on FP16 matmul"))
    else:
        features.append(("Tensor Cores", False, "matmul runs on regular CUDA cores"))

    # BF16 (bfloat16) — preferred for training stability
    if cc >= 8.0:
        features.append(("BF16 training", True, "stable training, no loss scaling needed"))
    else:
        features.append(("BF16 training", False, "use FP16 mixed precision instead"))

    # Flash Attention (memory-efficient attention)
    if cc >= 8.0:
        features.append(("Flash Attention", True, "essential for long-context training"))
    else:
        features.append(("Flash Attention", False, "use standard attention; fine for ≤512 tokens"))

    # FP8 (8-bit floats, very new)
    if cc >= 8.9:
        features.append(("FP8 training", True, "2× faster than FP16 if your model supports it"))
    else:
        features.append(("FP8 training", False, "Hopper or newer required"))

    return features


# ---------- Run the checks ----------
print("=" * 60)
print(" PyTorch & GPU Verification")
print("=" * 60)

print(f"\nPython version : {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\n⚠  WARNING: CUDA not available. Training will run on CPU and be VERY slow.")
    print("   Likely causes:")
    print("     - PyTorch installed without CUDA support (CPU-only wheel)")
    print("     - NVIDIA driver too old or missing")
    print("     - GPU not detected (check `nvidia-smi`)")
    sys.exit(1)


# ---------- GPU details ----------
print(f"\nCUDA runtime  : {torch.version.cuda}")
print(f"  ↑ This is the CUDA version PyTorch was built against — bundled inside the wheel.")
print(f"    It is NOT the same as the 'CUDA Version' shown in `nvidia-smi`,")
print(f"    which is the *maximum* version your driver can support.")
print(f"    As long as nvidia-smi's number ≥ this number, you're fine.")

props = torch.cuda.get_device_properties(0)
print(f"\nGPU            : {torch.cuda.get_device_name(0)}")
print(f"Total VRAM     : {props.total_memory / 1e9:.2f} GB")
print(f"Compute capability: {props.major}.{props.minor}")


# ---------- Architecture explanation ----------
arch, family, notes = lookup_capability(props.major, props.minor)
print(f"  ↑ Architecture: {arch} ({family})")
print(f"    {notes}")
print()
print(f"  About 'compute capability':")
print(f"    Every NVIDIA GPU has a version number indicating its hardware generation.")
print(f"    Higher numbers = newer architecture = more ML features supported.")
print(f"    Anything ≥ 7.0 (Volta or newer) has Tensor Cores and trains modern models well.")
print(f"    Anything ≥ 8.0 (Ampere or newer) supports BF16 and Flash Attention.")


# ---------- Feature support summary ----------
print(f"\nFeatures supported on this GPU:")
print(f"  {'Feature':<25} {'Supported':<12} Notes")
print(f"  {'-' * 25:<25} {'-' * 12:<12} {'-' * 40}")
for feature, supported, note in feature_support(props.major, props.minor):
    mark = "✓ yes" if supported else "✗ no"
    print(f"  {feature:<25} {mark:<12} {note}")


# ---------- Real GPU computation ----------
print(f"\nRunning a real computation on the GPU to confirm it works...")
x = torch.randn(1000, 1000, device='cuda')
y = x @ x.T
torch.cuda.synchronize()  # wait for the GPU to finish
print(f"  Matrix multiply succeeded. Output shape: {y.shape}")

allocated_mb = torch.cuda.memory_allocated() / 1e6
total_mb = props.total_memory / 1e6
print(f"  Memory allocated: {allocated_mb:.1f} MB / {total_mb:.0f} MB total ({100*allocated_mb/total_mb:.1f}%)")


# ---------- Final verdict ----------
print(f"\n{'=' * 60}")
if props.total_memory < 4e9:
    print(" ⚠  Your GPU has less than 4GB VRAM. Plan for very small models.")
elif props.total_memory < 7e9:
    print(" ✓  Your GPU is suitable for educational LLM training (10–30M params).")
elif props.total_memory < 13e9:
    print(" ✓  Your GPU is suitable for serious training (50–150M params).")
else:
    print(" ✓  Your GPU is large enough for substantial models (300M+ params).")
print(f"{'=' * 60}\n")