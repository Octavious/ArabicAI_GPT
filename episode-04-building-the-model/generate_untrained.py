import sys
import torch
from model import GPT, GPTConfig
from pathlib import Path
try:
    from tokenizers import Tokenizer
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: tokenizers\n"
        "Install it in the active uv environment with:\n"
        "  uv pip install tokenizers"
    ) from exc

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load our tokenizer from Episode 2
tokenizer_path = Path(__file__).with_name("tinystories_tokenizer.json")
tokenizer = Tokenizer.from_file(str(tokenizer_path))

# Build the (randomly initialized) model
model = GPT(GPTConfig())
model.eval()

# Encode a prompt
prompt = "Once upon a time"
ids = tokenizer.encode(prompt).ids
idx = torch.tensor([ids], dtype=torch.long)  # shape (1, T)

# Generate 100 tokens with no training whatsoever
with torch.no_grad():
    output = model.generate(idx, max_new_tokens=100, temperature=1.0)

# Decode and print
text = tokenizer.decode(output[0].tolist())
print(text)