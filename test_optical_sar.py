import sys
import os
import time
import warnings
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.dirname(__file__))

from modules.model_registry import ModelRegistry

def create_synthetic_sar(optical_path, output_path):
    """Creates a rough synthetic SAR stand-in by converting optical to grayscale."""
    img = Image.open(optical_path).convert("L")
    img = img.convert("RGB") # Save as 3-channel for consistency
    img.save(output_path)
    print(f"Created synthetic SAR stand-in: {output_path}")

print("Initializing Registry...")
registry = ModelRegistry()
print("Registry ready.\n")

# Prepare test pairs using synthetic SAR (grayscale)
img1_opt = "sanity_imgs/real_lake.jpg"
img1_sar = "sanity_imgs/fake_sar_lake.jpg"
create_synthetic_sar(img1_opt, img1_sar)

img2_opt = "sanity_imgs/real_freeway.jpg"
img2_sar = "sanity_imgs/fake_sar_freeway.jpg"
create_synthetic_sar(img2_opt, img2_sar)

queries = [
    {
        "opt": img1_opt,
        "sar": img1_sar,
        "query": "Identify built-up and water-covered regions in these images."
    },
    {
        "opt": img2_opt,
        "sar": img2_sar,
        "query": "What type of infrastructure is visible across both images?"
    }
]

for idx, q in enumerate(queries):
    print(f"\n============================================================")
    print(f"PAIR {idx+1}: {q['opt']} & {q['sar']} (Synthetic)")
    print(f"QUERY: {q['query']}")
    
    start_time = time.time()
    result = registry.run_optical_sar(q["opt"], q["sar"], q["query"], {})
    elapsed = time.time() - start_time
    
    print(f"ANSWER: {result.get('answer')}")
    print(f"CONFIDENCE: {result.get('confidence'):.4f}")
    print(f"TIME TAKEN: {elapsed:.2f} seconds")
    print(f"============================================================\n")

print("Optical-SAR tests complete.")
