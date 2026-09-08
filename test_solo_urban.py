"""Test solo nouns for urban queries."""

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
    
    # We will use 0.10 threshold as in the previous test for comparison of count
    keep = results["scores"] >= 0.10
    scores = results["scores"][keep]
    boxes = results["boxes"][keep]
    raw_labels = results.get("text_labels", results.get("labels", []))
    labels = [l for l, k in zip(raw_labels, keep.tolist()) if k]
    
    print(f"  Detections above box_thresh 0.10: {len(scores)}")
    for j in range(len(scores)):
        print(f"    - [{labels[j]}] conf={scores[j]:.4f}")
    print("-" * 65)

# Test on the original freeway image
img_path = "sanity_imgs/real_freeway.jpg"
check_query(img_path, "building. built-up area.", "Original paired")
check_query(img_path, "built-up area.", "Solo built-up area")
check_query(img_path, "building.", "Solo building")
