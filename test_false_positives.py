import sys
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.dirname(__file__))

from modules.model_registry import ModelRegistry

print("Initializing Registry...")
registry = ModelRegistry()
print("Registry ready.\n")

# We will test queries for objects that definitely do NOT exist in these remote sensing images
# to see if a relaxed box_threshold of 0.20 causes "hallucinated" bounding boxes.
CHECKS = [
    {
        "path": "sanity_imgs/real_forest.jpg",
        "queries": ["elephant.", "swimming pool.", "car."],
        "desc": "Forest Scene"
    },
    {
        "path": "sanity_imgs/real_airport.jpg",
        "queries": ["elephant.", "swimming pool.", "boat."],
        "desc": "Airport Scene"
    }
]

thresholds_to_test = [0.35, 0.25, 0.20, 0.15]

for c in CHECKS:
    print("=" * 70)
    print(f"IMAGE : {c['desc']} ({c['path']})")
    print("=" * 70)
    
    for query in c["queries"]:
        print(f"\n- QUERY: '{query}'")
        for thresh in thresholds_to_test:
            params = {"box_threshold": thresh, "text_threshold": 0.15}
            r = registry.run_grounding(c["path"], [query], params)
            
            conf = r.get("confidence", 0.0)
            box = r.get("box")
            
            if conf > 0 and box:
                print(f"  [HALLUCINATION] thresh={thresh:.2f} -> Detected '{r['label']}' with conf {conf:.4f} at {box}")
            else:
                print(f"  [PASS]          thresh={thresh:.2f} -> No detection.")
print("\nDone.")
