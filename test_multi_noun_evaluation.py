"""Test multi-candidate noun evaluation through ModelRegistry."""

import warnings, sys, os
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.dirname(__file__))

from modules.model_registry import ModelRegistry

print("Initializing Registry...")
registry = ModelRegistry()
print("Registry ready.\n")

CHECKS = [
    {
        "path": "sanity_imgs/real_freeway_0.jpg",
        "query": ["road.", "highway."],
        "desc": "Freeway 0 (road vs highway)"
    },
    {
        "path": "sanity_imgs/real_freeway_1.jpg",
        "query": ["road.", "highway."],
        "desc": "Freeway 1 (road vs highway)"
    },
    {
        "path": "sanity_imgs/real_freeway_2.jpg",
        "query": ["road.", "highway."],
        "desc": "Freeway 2 (road vs highway)"
    },
    {
        "path": "sanity_imgs/real_lake.jpg",
        "query": ["water body.", "lake."],
        "desc": "Lake Mapourika (water body vs lake)"
    },
    {
        "path": "sanity_imgs/real_freeway.jpg",
        "query": ["building.", "built-up area."],
        "desc": "Original Freeway (building vs built-up area)"
    }
]

for c in CHECKS:
    if not os.path.exists(c["path"]):
        print(f"Skipping {c['path']}, not found.")
        continue
        
    print("=" * 65)
    print(f"IMAGE : {c['desc']} ({c['path']})")
    
    r = registry.run_grounding(c["path"], c["query"], {"box_threshold": 0.35, "text_threshold": 0.25})
    
    print(f"  Chosen label : {r['label']}")
    print(f"  Confidence   : {r['confidence']:.4f}")
    print("=" * 65)
    print()
