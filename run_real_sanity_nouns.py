"""Run GroundingDINO on real satellite images with correct noun-phrase formatting."""

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

CHECKS = [
    {
        "path": "sanity_imgs/real_lake.jpg",
        "query": "water body. lake.",
        "desc": "Lake Mapourika (Real)",
    },
    {
        "path": "sanity_imgs/real_lake.jpg",
        "query": "river. water.",
        "desc": "Lake Mapourika (Real)",
    },
    {
        "path": "sanity_imgs/real_freeway.jpg",
        "query": "road. highway.",
        "desc": "Freeway crop (Real)",
    },
    {
        "path": "sanity_imgs/real_freeway.jpg",
        "query": "building. built-up area.",
        "desc": "Freeway crop (Real)",
    },
]

for c in CHECKS:
    img_path = c["path"]
    query = c["query"]
    print("=" * 65)
    print(f"IMAGE : {c['desc']} ({img_path})")
    print(f"QUERY : '{query}'")
    print("-" * 65)
    
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, text=query, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Get the highest box score across all queries
    logits = outputs.logits[0]  # (900, 256)
    # The classification score is the max over classes for each box
    box_scores, _ = logits.max(dim=-1)
    box_probs = torch.sigmoid(box_scores)
    max_box_prob, max_box_idx = box_probs.max(dim=0)
    
    print(f"  Highest box prob in raw output: {max_box_prob.item():.4f}")
    
    # Process at 0.10 threshold
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        text_threshold=0.10,
        target_sizes=[img.size[::-1]]
    )[0]
    
    keep = results["scores"] >= 0.10
    scores = results["scores"][keep]
    boxes = results["boxes"][keep]
    raw_labels = results.get("text_labels", results.get("labels", []))
    labels = [l for l, k in zip(raw_labels, keep.tolist()) if k]
    
    print(f"  Detections above box_thresh 0.10: {len(scores)}")
    for i in range(len(scores)):
        print(f"    - [{labels[i]}] conf={scores[i]:.4f} box={[round(x,1) for x in boxes[i].tolist()]}")
    
    # If no detections at 0.10, sweep down to find the threshold where the first detection appears
    if len(scores) == 0:
        found = False
        for thresh in [0.08, 0.05, 0.03, 0.01]:
            keep = results["scores"] >= thresh
            scores_t = results["scores"][keep]
            if len(scores_t) > 0:
                raw_labels = results.get("text_labels", results.get("labels", []))
                labels_t = [l for l, k in zip(raw_labels, keep.tolist()) if k]
                print(f"  First detection appears at thresh={thresh}: [{labels_t[0]}] conf={scores_t[0]:.4f}")
                found = True
                break
        if not found:
            print("  No detections even down to thresh=0.01")
    print("=" * 65)
    print()
