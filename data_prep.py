import os
import json
import glob
from huggingface_hub import hf_hub_download

def find_cached_vrsbench_json():
    """Locates the downloaded VRSBench_train.json inside the local Hugging Face cache."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--xiang709--VRSBench")
    matches = glob.glob(os.path.join(cache_dir, "**", "VRSBench_train.json"), recursive=True)
    return matches[0] if matches else None

def prepare_vrsbench_for_tuning(output_file="data/vrsbench_train.json"):
    os.makedirs("data", exist_ok=True)
    print("Locating downloaded VRSBench dataset...")
    
    json_path = find_cached_vrsbench_json()
    
    # If not found in cache, fetch the JSON file directly
    if not json_path or not os.path.exists(json_path):
        print("Fetching VRSBench_train.json directly from Hugging Face Hub...")
        json_path = hf_hub_download(
            repo_id="xiang709/VRSBench", 
            filename="VRSBench_train.json", 
            repo_type="dataset"
        )

    print(f"Loading raw dataset from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"Successfully loaded {len(raw_data)} records using standard JSON parser.")
    
    formatted_data = []
    for i, item in enumerate(raw_data):
        # Extract question, answer, and image path safely
        question = item.get('question') or item.get('caption') or 'Describe the visible features in this image.'
        answer = item.get('answer') or item.get('label') or 'Urban and natural land-cover.'
        img_id = str(item.get('id', i))
        
        conversation = {
            "id": f"vrsbench_{img_id}",
            "image": item.get('image_path', f"images/{img_id}.jpg"),
            "conversations": [
                {"from": "human", "value": f"<image>\n{question}"},
                {"from": "gpt", "value": str(answer)}
            ]
        }
        formatted_data.append(conversation)
        
        # Limit to first 1,000 samples for quick local testing/fine-tuning
        if len(formatted_data) >= 1000:
            break

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=4)
        
    print(f"✅ Saved {len(formatted_data)} formatted samples to {output_file}")

if __name__ == "__main__":
    prepare_vrsbench_for_tuning()