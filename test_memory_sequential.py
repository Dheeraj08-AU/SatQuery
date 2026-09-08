import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from modules.model_registry import ModelRegistry

print("Starting Sequential Memory Test...")
registry = ModelRegistry()

image_path = "sanity_imgs/real_freeway.jpg"
vqa_query = "How many vehicles are visible?"
grounding_query = "vehicle."

print("\n--- STEP 1: Running VQA ---")
print(f"Querying: '{vqa_query}' on {image_path}")
vqa_result = registry.run_single_vqa(image_path, vqa_query, {})
print(f"VQA Result: {vqa_result}")

print("\n--- STEP 2: Running Grounding ---")
print(f"Querying: '{grounding_query}' on {image_path}")
grounding_result = registry.run_grounding(image_path, grounding_query, {})
print(f"Grounding Result Box: {grounding_result.get('box')}")

print("\nSequential run completed successfully without OOM!")
