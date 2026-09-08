import sys
import os
import time
import warnings
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.dirname(__file__))

from modules.model_registry import ModelRegistry

print("Initializing Registry...")
registry = ModelRegistry()
print("Registry ready.\n")

# Use a real image we already have
t1_path = "sanity_imgs/real_freeway.jpg"
t2_path = "sanity_imgs/fake_t2.jpg"

print(f"Creating fake T2 image by copying a patch over a section of {t1_path}...")
img = Image.open(t1_path).convert("RGB")
w, h = img.size
# Crop a section (e.g., bottom-left quadrant which is likely grass)
patch = img.crop((0, h//2, w//2, h))
# Paste it over the top-right quadrant (where the highway is)
img.paste(patch, (w//2, 0))
img.save(t2_path)
print(f"Saved realistic fake T2 to {t2_path}.\n")

query = "What changed in the land cover?"
params = {}

print(f"Running Change Analysis with query: '{query}'")
print(f"T1: {t1_path}")
print(f"T2: {t2_path}")

start_time = time.time()
result = registry.run_change_analysis(t1_path, t2_path, query, params)
end_time = time.time()

print("\n==================================================")
print("RESULTS:")
print("==================================================")
print(f"Answer: {result.get('answer')}")
print(f"Confidence: {result.get('confidence', 0.0):.4f}")
print(f"Inference Time: {end_time - start_time:.2f} seconds")

# Clean up
if os.path.exists(t2_path):
    os.remove(t2_path)
