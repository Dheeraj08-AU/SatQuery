import os
import tarfile
import json
import subprocess
import io
import random
from PIL import Image

def download_shards(shards, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    base_url = "https://huggingface.co/datasets/ljx620/CDVQA/resolve/main/train/"
    for shard in shards:
        tar_path = os.path.join(dest_dir, shard)
        if not os.path.exists(tar_path) or os.path.getsize(tar_path) < 100000:
            print(f"Downloading {shard}...")
            subprocess.run(["curl", "-L", "-o", tar_path, base_url + shard], check=True)
        else:
            print(f"{shard} already exists, skipping download.")

def extract_and_format(shards, dest_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    samples = []
    
    # We use the explicit framing prefix tested during inference
    prefix = "The left image is from time T1, the right image is from time T2. Answer the question about what changed: "
    
    for shard in shards:
        tar_path = os.path.join(dest_dir, shard)
        print(f"Processing {tar_path}...")
        try:
            with tarfile.open(tar_path, "r") as tar:
                members = tar.getmembers()
                json_members = [m for m in members if m.name.endswith(".json")]
                
                for jm in json_members:
                    try:
                        data = json.load(tar.extractfile(jm))
                        
                        convs = data.get("conversations", [])
                        if len(convs) < 2: continue
                        
                        q_raw = convs[0].get("value", "")
                        ans = convs[1].get("value", "")
                        
                        # Strip boilerplate: "Image 1: <image>\nImage 2: <image>\n"
                        q_clean = q_raw.replace("Image 1: <image>\nImage 2: <image>\n", "").strip()
                        
                        base_name = jm.name.replace(".json", "")
                        t1_member = next((m for m in members if m.name == f"{base_name}.0.img"), None)
                        t2_member = next((m for m in members if m.name == f"{base_name}.1.img"), None)
                        
                        if not t1_member or not t2_member:
                            continue
                            
                        t1_img = Image.open(io.BytesIO(tar.extractfile(t1_member).read())).convert("RGB")
                        t2_img = Image.open(io.BytesIO(tar.extractfile(t2_member).read())).convert("RGB")
                        
                        # Build Composite (T1 | T2)
                        w = t1_img.width + t2_img.width
                        h = max(t1_img.height, t2_img.height)
                        comp_img = Image.new("RGB", (w, h))
                        comp_img.paste(t1_img, (0, 0))
                        comp_img.paste(t2_img, (t1_img.width, 0))
                        
                        img_filename = f"{base_name}_composite.jpg"
                        comp_img.save(os.path.join(out_dir, img_filename))
                        
                        formatted_q = f"{prefix}{q_clean}"
                        
                        samples.append({
                            "image": img_filename,
                            "composite_size": comp_img.size,
                            "question": formatted_q,
                            "answer": ans,
                            "raw_question": q_clean
                        })
                        
                    except Exception as e:
                        pass
        except Exception as e:
            print(f"Error opening tar {shard}: {e}")
            
    return samples

def main():
    shards = ["train-00000.tar", "train-00010.tar", "train-00020.tar", "train-00030.tar", "train-00040.tar"]
    data_dir = "cdvqa_raw"
    out_dir = "cdvqa_formatted"
    
    download_shards(shards, data_dir)
    samples = extract_and_format(shards, data_dir, out_dir)
    
    print(f"\nTotal samples extracted: {len(samples)}")
    
    random.seed(42)
    random.shuffle(samples)
    
    eval_samples = samples[:10]
    train_samples = samples[10:]
    
    with open(os.path.join(out_dir, "train.jsonl"), "w") as f:
        for s in train_samples:
            f.write(json.dumps({"image": s["image"], "prefix": s["question"], "suffix": s["answer"]}) + "\n")
            
    with open(os.path.join(out_dir, "eval.jsonl"), "w") as f:
        for s in eval_samples:
            f.write(json.dumps({"image": s["image"], "prefix": s["question"], "suffix": s["answer"]}) + "\n")
            
    print("\n--- SANITY CHECK: 2 REAL SAMPLES ---")
    for i in range(2):
        print(f"Sample {i+1}:")
        print(f"  Image file: {samples[i]['image']} (Dimensions: {samples[i]['composite_size']})")
        print(f"  Formatted Prompt: {samples[i]['question']}")
        print(f"  Answer: {samples[i]['answer']}")
        print(f"  Raw Question: {samples[i]['raw_question']}")
        print("-" * 40)

if __name__ == "__main__":
    main()
