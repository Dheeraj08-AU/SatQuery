import sys
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.dirname(__file__))

from modules.model_registry import ModelRegistry
from datasets import load_dataset

OUTPUT_DIR = "sanity_imgs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# We will use timm/resisc45 via HF datasets, which is the same approach that 
# successfully gave us real_freeway.jpg earlier and is likely cached.
# RESISC45 classes: 0: airplane, 1: airport, ... 13: forest, ... 16: harbor, 38: storage_tank
print("Loading RESISC45 dataset to extract new scenes...")
try:
    ds = load_dataset('timm/resisc45', split='train', streaming=True)
    airport_saved = False
    harbor_saved = False
    forest_saved = False
    tank_saved = False

    for item in ds:
        label = item['label']
        # 1: airport, 16: harbor, 13: forest, 38: storage_tank (using known RESISC45 class indices)
        if label == 1 and not airport_saved:
            item['image'].save('sanity_imgs/real_airport.jpg')
            airport_saved = True
        elif label == 16 and not harbor_saved:
            item['image'].save('sanity_imgs/real_harbor.jpg')
            harbor_saved = True
        elif label == 13 and not forest_saved:
            item['image'].save('sanity_imgs/real_forest.jpg')
            forest_saved = True
        elif label == 38 and not tank_saved:
            item['image'].save('sanity_imgs/real_tanks.jpg')
            tank_saved = True
            
        if airport_saved and harbor_saved and forest_saved and tank_saved:
            break
    print("Successfully extracted test images from RESISC45.")
except Exception as e:
    print(f"Failed to load datasets: {e}")
    sys.exit(1)

print("\nInitializing Registry...")
registry = ModelRegistry()
print("Registry ready.\n")

CHECKS = [
    {
        "path": "sanity_imgs/real_airport.jpg",
        "single": ["airport."],
        "multi": ["airport.", "runway.", "airstrip."],
        "desc": "RESISC45 Airport (airport vs runway vs airstrip)",
        "thresh": {"box_threshold": 0.25, "text_threshold": 0.15}
    },
    {
        "path": "sanity_imgs/real_harbor.jpg",
        "single": ["ships."],
        "multi": ["ships.", "boats.", "vessels."],
        "desc": "RESISC45 Harbor (ships vs boats vs vessels)",
        "thresh": {"box_threshold": 0.20, "text_threshold": 0.15}
    },
    {
        "path": "sanity_imgs/real_forest.jpg",
        "single": ["trees."],
        "multi": ["trees.", "forest.", "vegetation."],
        "desc": "RESISC45 Forest (trees vs forest vs vegetation)",
        "thresh": {"box_threshold": 0.20, "text_threshold": 0.15}
    },
    {
        "path": "sanity_imgs/real_tanks.jpg",
        "single": ["storage tanks."],
        "multi": ["storage tanks.", "silos.", "containers."],
        "desc": "RESISC45 Storage Tanks (storage tanks vs silos vs containers)",
        "thresh": {"box_threshold": 0.20, "text_threshold": 0.15}
    }
]

for c in CHECKS:
    print("=" * 75)
    print(f"IMAGE : {c['desc']}")
    
    # Run Single Best
    r_single = registry.run_grounding(c["path"], c["single"], c["thresh"])
    conf_single = r_single.get("confidence", 0.0)
    
    # Run Multi-Candidate
    r_multi = registry.run_grounding(c["path"], c["multi"], c["thresh"])
    conf_multi = r_multi.get("confidence", 0.0)
    label_multi = r_multi.get("label", "none")
    box_multi = r_multi.get("box", [])
    
    print(f"  Single Candidate [{c['single'][0]}] Confidence : {conf_single:.4f}")
    print(f"  Multi-Candidate Winner [{label_multi}] Confidence : {conf_multi:.4f}")
    if box_multi:
        print(f"  Box extracted: {[round(x, 1) for x in box_multi]}")
    
    if conf_multi >= conf_single:
        print("  [PASS] Multi-candidate matches or beats single best.")
    else:
        print("  [FAIL] Multi-candidate performed worse!")
    print("=" * 75)
    print()
