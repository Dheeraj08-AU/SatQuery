import sys
import os
import time
import warnings
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.dirname(__file__))

from modules.model_registry import ModelRegistry

print("Initializing Registry...")
registry = ModelRegistry()
print("Registry ready.\n")

t1_path = "sanity_imgs/real_freeway.jpg"
t2_path = "sanity_imgs/fake_t2_realistic.jpg"

print(f"Creating fake T2 image by copying a grass patch over the highway...")
img = Image.open(t1_path).convert("RGB")
w, h = img.size
# Crop a section (bottom-left quadrant is grass)
patch = img.crop((0, h//2, w//2, h))
# Paste it over the top-right quadrant (where the highway is)
img.paste(patch, (w//2, 0))
img.save(t2_path)
print(f"Saved realistic fake T2 to {t2_path}.\n")

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

if os.path.exists(t2_path):
    os.remove(t2_path)
