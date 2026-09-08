import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from modules.model_registry import ModelRegistry

print("Starting VQA repetition test script...")
registry = ModelRegistry()
print("\nRegistry loaded. Running VQA Inference...\n")

TEST_CASES = [
    # 3 times on the same image
    {
        "image": "sanity_imgs/real_freeway.jpg",
        "query": "How many vehicles are visible?"
    },
    {
        "image": "sanity_imgs/real_freeway.jpg",
        "query": "How many vehicles are visible?"
    },
    {
        "image": "sanity_imgs/real_freeway.jpg",
        "query": "How many vehicles are visible?"
    },
    # 2 different counting queries on 2 other images
    {
        "image": "sanity_imgs/real_urban.jpg",
        "query": "how many buildings are in this image?"
    },
    {
        "image": "sanity_imgs/real_river.jpg",
        "query": "how many rivers are visible?"
    }
]

for sample in TEST_CASES:
    if not os.path.exists(sample["image"]):
        print(f"Skipping missing image: {sample['image']}")
        continue
        
    print("-" * 40)
    print(f"IMAGE: {sample['image']}")
    print(f"QUERY: {sample['query']}")
    
    start_time = time.time()
    # temperature 0.2 was the default in agent_controller.py
    result = registry.run_single_vqa(sample["image"], sample["query"], {"max_new_tokens": 128})
    
    print(f"RAW ANSWER: {result.get('answer', 'N/A')}")
    print(f"CONFIDENCE: {result.get('confidence', 0.0):.4f}")
    print("-" * 40)
    print()
