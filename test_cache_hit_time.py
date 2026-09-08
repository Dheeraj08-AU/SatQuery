import sys
import os
import time
sys.path.insert(0, os.path.dirname(__file__))
from modules.model_registry import ModelRegistry

print("Initializing Registry...")
registry = ModelRegistry()

image = "sanity_imgs/real_freeway_0.jpg"
query = "Is there a highway in this image?"

print("\n--- Testing Cache Hit Speed (Cold Start of run_single_vqa) ---")
start = time.perf_counter()
res = registry.run_single_vqa(image, query, {})
end = time.perf_counter()

print(f"Query: '{query}'")
print(f"Result: {res}")
print(f"Elapsed Time: {(end - start) * 1000:.2f} ms")
