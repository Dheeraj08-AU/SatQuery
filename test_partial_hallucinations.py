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

# Images to test
IMAGES = [
    {"path": "sanity_imgs/real_forest.jpg", "desc": "Forest"},
    {"path": "sanity_imgs/real_freeway.jpg", "desc": "Freeway"},
    {"path": "sanity_imgs/real_airport.jpg", "desc": "Airport"}
]

# Queries to test (discrete objects, not scene-wide)
QUERIES = [
    "swimming pool.",
    "person.",
    "bicycle.",
    "aircraft.",
    "car."
]

params = {"box_threshold": 0.20, "text_threshold": 0.15}
partial_hallucinations_found = False

for img_dict in IMAGES:
    print(f"==================================================")
    print(f"IMAGE: {img_dict['desc']}")
    print(f"==================================================")
    
    # We load image to compute the area manually for reporting
    img = Image.open(img_dict["path"])
    img_area = img.size[0] * img.size[1]
    
    for query in QUERIES:
        # Note: 'aircraft' on 'Airport' might be a true positive, so we should take that with a grain of salt
        if img_dict['desc'] == "Airport" and query == "aircraft.":
            continue
            
        # Run detection (this automatically applies the >95% filter internally)
        r = registry.run_grounding(img_dict["path"], [query], params)
        box = r.get("box")
        conf = r.get("confidence", 0.0)
        
        if box:
            x_min, y_min, x_max, y_max = box
            box_area = (x_max - x_min) * (y_max - y_min)
            area_ratio = box_area / img_area
            
            # Since the >95% filter already rejected it if it was 100%, 
            # if we get a box here it MUST be a partial false positive!
            print(f"  [HALLUCINATION DETECTED] '{query}' -> Conf: {conf:.4f} | Box Area: {area_ratio*100:.1f}%")
            print(f"    Box: {[round(b, 1) for b in box]}")
            partial_hallucinations_found = True
        else:
            print(f"  [PASS] '{query}' -> No confident detection.")
    print()

if not partial_hallucinations_found:
    print("SUCCESS: No partial hallucinations slipped through the 0.20 threshold filter!")
else:
    print("WARNING: Partial hallucinations were detected at box_threshold=0.20!")
