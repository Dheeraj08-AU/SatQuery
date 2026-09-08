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

t1_path = "sanity_imgs/real_freeway.jpg"
t2_path = "sanity_imgs/fake_t2_black.jpg"

print(f"Creating fake T2 image with a black rectangle over the highway...")
img = Image.open(t1_path).convert("RGB")
w, h = img.size
draw = ImageDraw.Draw(img)
# Draw a black rectangle over the highway (right side/top quadrant)
draw.rectangle([(w//2, 0), (w, h)], fill="black")
img.save(t2_path)
print(f"Saved fake T2 to {t2_path}.\n")

queries = [
    "What changed in the land cover?",
    "Please briefly describe the changes in these two images.",
    "Did the road area change?",
    "Did the built-up area change?"
]

params = {}

for q in queries:
    print(f"Query: '{q}'")
    start = time.time()
    result = registry.run_change_analysis(t1_path, t2_path, q, params)
    end = time.time()
    print(f"  Answer: {result.get('answer')}")
    print(f"  Time: {end - start:.2f}s\n")

if os.path.exists(t2_path):
    os.remove(t2_path)
