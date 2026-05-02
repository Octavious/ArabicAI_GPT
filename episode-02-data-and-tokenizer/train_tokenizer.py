from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# 1. Load the dataset (already cached from last time, so this is instant)
print("Loading dataset...")
dataset = load_dataset("roneneldan/TinyStories")

# 2. Create an iterator that yields text in batches
# We don't load all 2M stories into memory at once — we stream them
def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset['train']), batch_size):
        yield dataset['train'][i : i + batch_size]['text']

# 3. Initialize an empty BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))

# 4. Configure it to work at the byte level (handles any Unicode safely)
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

# 5. Set up the trainer with our desired vocabulary size
trainer = BpeTrainer(
    vocab_size=4096,
    special_tokens=["<|endoftext|>", "<|unk|>"],
    initial_alphabet=ByteLevel.alphabet(),
    show_progress=True,
)

# 6. Train!
print("Training tokenizer on TinyStories (this takes a few minutes)...")
tokenizer.train_from_iterator(
    batch_iterator(),
    trainer=trainer,
    length=len(dataset['train']),
)

# 7. Save it to disk so we don't have to retrain
tokenizer.save("tinystories_tokenizer.json")
print(f"\nDone! Vocabulary size: {tokenizer.get_vocab_size()}")
print("Saved to tinystories_tokenizer.json")