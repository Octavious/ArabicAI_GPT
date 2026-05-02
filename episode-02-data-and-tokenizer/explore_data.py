from datasets import load_dataset

# This downloads TinyStories the first time (~2GB) and caches it locally
# Subsequent runs will be instant
print("Loading TinyStories...")
dataset = load_dataset("roneneldan/TinyStories")

# A dataset has "splits" — typically train and validation
print(f"\nDataset structure: {dataset}")

# Look at the first story
print(f"\n--- First story ---")
print(dataset['train'][0]['text'])

# Look at the 100th story to see variety
print(f"\n--- Story #100 ---")
print(dataset['train'][100]['text'])

# How many stories do we have?
print(f"\nTotal training stories: {len(dataset['train']):,}")
print(f"Total validation stories: {len(dataset['validation']):,}")