"""Test solo nouns vs paired nouns on GroundingDINO to check confidence margins."""

import warnings, sys, os
import torch
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.dirname(__file__))

from transformers import AutoProcessor, GroundingDinoForObjectDetection
from PIL import Image

print("Loading model...")
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").eval().float()
print("Model loaded.\n")

def check_query(img_path, query, desc):
    if not os.path.exists(img_path):
        print(f"Skipping {img_path}, not found.")
        return
        
    print("=" * 65)
    print(f"IMAGE : {desc} ({img_path})")
    print(f"QUERY : '{query}'")
    
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, text=query, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits[0]  
    box_scores, _ = logits.max(dim=-1)
    box_probs = torch.sigmoid(box_scores)
    max_box_prob, _ = box_probs.max(dim=0)
    
    print(f"  Highest box prob in raw output: {max_box_prob.item():.4f}")
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        text_threshold=0.10,
        target_sizes=[img.size[::-1]]
    )[0]
    
    keep = results["scores"] >= 0.35
    scores = results["scores"][keep]
    boxes = results["boxes"][keep]
    raw_labels = results.get("text_labels", results.get("labels", []))
    labels = [l for l, k in zip(raw_labels, keep.tolist()) if k]
    
    print(f"  Detections above box_thresh 0.35: {len(scores)}")
    for j in range(len(scores)):
        print(f"    - [{labels[j]}] conf={scores[j]:.4f}")
    print("-" * 65)

# 1. Test "road." on freeways
for i in range(3):
    check_query(f"sanity_imgs/real_freeway_{i}.jpg", "road.", f"Freeway {i} (solo road)")
    
# 3. Test "water body." on lake
check_query("sanity_imgs/real_lake.jpg", "water body.", "Lake Mapourika (solo water body)")
