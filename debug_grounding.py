"""Debug GroundingDINO tokenization and raw outputs."""

import warnings, sys, os
import torch
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers import AutoProcessor, GroundingDinoForObjectDetection
from PIL import Image

print("Loading model for debugging...")
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").eval().float()
print("Model loaded.")

img_path = "sanity_imgs/real_freeway.jpg"
query = "detect the road"
print(f"\nImage: {img_path}")
print(f"Query: '{query}'")

img = Image.open(img_path).convert("RGB")
inputs = processor(images=img, text=query, return_tensors="pt")

print("\n--- TOKENIZATION ---")
input_ids = inputs.input_ids[0]
tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
print(f"Input IDs: {input_ids.tolist()}")
print(f"Tokens: {tokens}")

print("\n--- RAW MODEL OUTPUT ---")
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits[0]  # (num_queries, num_tokens)
pred_boxes = outputs.pred_boxes[0] # (num_queries, 4)

print(f"Logits shape: {logits.shape}")
print(f"Pred boxes shape: {pred_boxes.shape}")

# Let's see the maximum logit for each token across all queries
max_logits, _ = logits.max(dim=0)
for i, token in enumerate(tokens):
    print(f"Token '{token}': max logit = {max_logits[i]:.4f} (prob = {torch.sigmoid(max_logits[i]):.4f})")

print("\n--- POST-PROCESSING AT DIFFERENT TEXT THRESHOLDS ---")
for text_thresh in [0.05, 0.10, 0.25, 0.50]:
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        text_threshold=text_thresh,
        target_sizes=[img.size[::-1]]
    )[0]
    
    keep = results["scores"] >= 0.05 # Lower filter
    scores = results["scores"][keep]
    raw_labels = results.get("text_labels", results.get("labels", []))
    labels = [l for l, k in zip(raw_labels, keep.tolist()) if k]
    
    print(f"\nText Threshold: {text_thresh}")
    if len(scores) == 0:
        print("  No detections.")
    else:
        for i in range(len(scores)):
            print(f"  Detection {i+1}: label = '{labels[i]}', score = {scores[i]:.4f}")

