import sys
import os
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.dirname(__file__))

from modules.model_registry import ModelRegistry

print("Initializing Registry...")
registry = ModelRegistry()
print("Registry ready.\n")

t1_path = "sanity_imgs/real_freeway_0.jpg"
# Use exactly the same image for T2 to simulate absolutely zero change
t2_path = "sanity_imgs/real_freeway_0.jpg"

query = "What changed?"
params = {}

print(f"Query: '{query}'")
start = time.time()
result = registry.run_change_analysis(t1_path, t2_path, query, params)
end = time.time()

print("\n--- FINAL MULTI-PROBE RESULT ---")
print(f"Answer: {result.get('answer')}")
print(f"Confidence: {result.get('confidence'):.4f}")
print(f"Total Time: {end - start:.2f}s\n")
