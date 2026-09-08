import sys
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.dirname(__file__))

from modules.model_registry import ModelRegistry

print("Initializing Registry...")
registry = ModelRegistry()
print("Registry ready.\n")

CHECKS = [
    # 4 known-good scene-wide cases
    {"path": "sanity_imgs/real_forest.jpg", "query": "trees.", "desc": "Good: trees"},
    {"path": "sanity_imgs/real_forest.jpg", "query": "forest.", "desc": "Good: forest"},
    {"path": "sanity_imgs/real_freeway.jpg", "query": "built-up area.", "desc": "Good: built-up area"},
    {"path": "sanity_imgs/real_airport.jpg", "query": "airport.", "desc": "Good: airport"},
    
    # 3 known hallucinations (or negative cases)
    {"path": "sanity_imgs/real_forest.jpg", "query": "swimming pool.", "desc": "Hallucination: swimming pool in forest"},
    {"path": "sanity_imgs/real_airport.jpg", "query": "boat.", "desc": "Hallucination: boat in airport"},
    {"path": "sanity_imgs/real_airport.jpg", "query": "elephant.", "desc": "Negative: elephant"}
]

# We use box_threshold=0.20 for all, which caused the hallucinations earlier
params = {"box_threshold": 0.20, "text_threshold": 0.15}

for c in CHECKS:
    print(f"Testing {c['desc']} (query: '{c['query']}') ...")
    r = registry.run_grounding(c["path"], [c["query"]], params)
    
    if r.get("box"):
        print(f"  -> PASSED: Detection kept ({r['label']})")
    else:
        print(f"  -> REJECTED / NO DETECTION")
    print()
