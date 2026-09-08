"""Test 3 more real freeway images to check GroundingDINO confidence margins."""

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

for i in range(3):
    img_path = f"sanity_imgs/real_freeway_{i}.jpg"
    if not os.path.exists(img_path):
        continue
        
    query = "road. highway."
    print("=" * 65)
    print(f"IMAGE : Freeway {i} ({img_path})")
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
    
    # Using 0.35 threshold which is the default in our app
    keep = results["scores"] >= 0.35
    scores = results["scores"][keep]
    boxes = results["boxes"][keep]
    raw_labels = results.get("text_labels", results.get("labels", []))
    labels = [l for l, k in zip(raw_labels, keep.tolist()) if k]
    
    print(f"  Detections above box_thresh 0.35: {len(scores)}")
    for j in range(len(scores)):
        print(f"    - [{labels[j]}] conf={scores[j]:.4f} box={[round(x,1) for x in boxes[j].tolist()]}")
    print("-" * 65)
