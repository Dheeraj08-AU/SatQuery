import sys
import os
import warnings
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.dirname(__file__))

from modules.model_registry import ModelRegistry

print("Initializing Registry...")
registry = ModelRegistry()
print("Registry ready.\n")

CHECKS = [
    {
        "path": "sanity_imgs/real_forest.jpg",
        "queries": ["trees.", "forest."],
        "desc": "Real Forest Scene"
    },
    {
        "path": "sanity_imgs/real_freeway.jpg",
        "queries": ["built-up area."],
        "desc": "Real Freeway Scene"
    },
    {
        "path": "sanity_imgs/real_airport.jpg",
        "queries": ["airport."],
        "desc": "Real Airport Scene"
    }
]

# We use box_threshold=0.20 to make sure we get the confident detections
params = {"box_threshold": 0.20, "text_threshold": 0.15}

for c in CHECKS:
    print("=" * 70)
    print(f"IMAGE : {c['desc']} ({c['path']})")
    
    img = Image.open(c["path"])
    img_width, img_height = img.size
    img_area = img_width * img_height
    
    print("-" * 70)
    
    for query in c["queries"]:
        r = registry.run_grounding(c["path"], [query], params)
        conf = r.get("confidence", 0.0)
        box = r.get("box")
        
        if box:
            x_min, y_min, x_max, y_max = box
            box_area = (x_max - x_min) * (y_max - y_min)
            area_ratio = box_area / img_area
            
            print(f"QUERY: '{query}' -> Conf: {conf:.4f} | Box: {[round(b, 1) for b in box]}")
            print(f"       -> Box Area: {box_area:.1f} / {img_area} ({area_ratio*100:.2f}%)")
            
            if area_ratio > 0.95:
                print("       -> [REJECTED] The >95% area filter would INCORRECTLY REJECT this true positive!")
            else:
                print("       -> [PASSED] The >95% area filter would KEEP this true positive.")
        else:
            print(f"QUERY: '{query}' -> No detection.")
        print()
