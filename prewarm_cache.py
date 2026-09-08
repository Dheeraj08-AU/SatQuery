import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from modules.model_registry import ModelRegistry

print("Starting VQA Pre-warming Script...")
registry = ModelRegistry()
print("\nRegistry loaded. Running pre-warming queries...\n")

TEST_SAMPLES = [
    {
        "image": "sanity_imgs/real_freeway_0.jpg",
        "query": "Is there a highway in this image?"
    },
    {
        "image": "sanity_imgs/real_freeway_0.jpg",
        "query": "What is the primary feature of this image?"
    },
    {
        "image": "sanity_imgs/real_lake.jpg",
        "query": "What is the color of the water body?"
    },
    {
        "image": "sanity_imgs/real_freeway.jpg",
        "query": "How many vehicles are visible?"
    },
    {
        "image": "sanity_imgs/real_urban.jpg",
        "query": "how many buildings are in this image?"
    },
    {
        "image": "sanity_imgs/real_river.jpg",
        "query": "how many rivers are visible?"
    }
]

for sample in TEST_SAMPLES:
    if not os.path.exists(sample["image"]):
        print(f"Skipping missing image: {sample['image']}")
        continue
        
    print(f"Running VQA: '{sample['query']}' on {sample['image']}")
    start_time = time.time()
    result = registry.run_single_vqa(sample["image"], sample["query"], {"max_new_tokens": 128})
    elapsed = time.time() - start_time
    print(f"-> Answer: {result.get('answer', 'N/A')} (Took {elapsed:.2f}s)\n")

CHANGE_SAMPLES = [
    {
        "t1": "sanity_imgs/real_freeway_0.jpg",
        "t2": "sanity_imgs/real_freeway_1.jpg",
        "query": "What changed?"
    },
    {
        "t1": "sanity_imgs/real_freeway_0.jpg",
        "t2": "sanity_imgs/real_freeway_2.jpg",
        "query": "What changed?"
    }
]

print("\nRunning pre-warming for Change Analysis (this may take ~5-6 minutes per pair)...")
for sample in CHANGE_SAMPLES:
    if not (os.path.exists(sample["t1"]) and os.path.exists(sample["t2"])):
        print(f"Skipping missing image pair: {sample['t1']} / {sample['t2']}")
        continue
        
    print(f"Running Change Analysis on {sample['t1']} and {sample['t2']}")
    start_time = time.time()
    result = registry.run_change_analysis(sample["t1"], sample["t2"], sample["query"], {})
    elapsed = time.time() - start_time
    print(f"-> Answer: {result.get('answer', 'N/A')} (Took {elapsed:.2f}s)\n")

print("Pre-warming complete!")
