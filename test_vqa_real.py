"""Test script to verify real VQA inference on CPU against real samples."""
import sys
import os
import json
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from modules.model_registry import ModelRegistry

print("Starting VQA test script...")
registry = ModelRegistry()
print("\nRegistry loaded. Running VQA Inference...\n")

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
    }
]

import time

for sample in TEST_SAMPLES:
    if not os.path.exists(sample["image"]):
        print(f"Skipping missing image: {sample['image']}")
        continue
        
    print("=" * 60)
    print(f"IMAGE: {sample['image']}")
    print(f"QUERY: {sample['query']}")
    
    start_time = time.time()
    result = registry.run_single_vqa(sample["image"], sample["query"], {"max_new_tokens": 128})
    elapsed = time.time() - start_time
    
    print(f"ANSWER: {result.get('answer', 'N/A')}")
    print(f"CONFIDENCE: {result.get('confidence', 0.0):.4f}")
    print(f"TIME TAKEN: {elapsed:.2f} seconds")
    print("=" * 60)
    print()
