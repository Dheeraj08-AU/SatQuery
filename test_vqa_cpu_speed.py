import time
import torch
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
from PIL import Image

def test_speed():
    model_id = "google/paligemma-3b-pt-224"
    adapter_path = "modules/satquery_vqa_adapter"
    img_path = "sanity_imgs/real_freeway_0.jpg"
    
    print("Testing bfloat16 loading...")
    dtype = torch.bfloat16
    start_load = time.time()
    
    try:
        processor = AutoProcessor.from_pretrained(adapter_path)
        # Load base model in bfloat16
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        # Load adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print(f"Loaded in {time.time() - start_load:.2f}s with {dtype}")
        
    except Exception as e:
        print(f"Failed with bfloat16: {e}")
        print("Falling back to float32...")
        dtype = torch.float32
        start_load = time.time()
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print(f"Loaded in {time.time() - start_load:.2f}s with {dtype}")
        
    model.eval()
    
    # Prepare input
    query = "answer en what is in this image?"
    img = Image.open(img_path).convert("RGB")
    inputs = processor(text=query, images=img, return_tensors="pt").to(dtype)
    # The input_ids and attention_mask must be long, pixel_values should be dtype
    inputs["input_ids"] = inputs["input_ids"].to(torch.long)
    inputs["attention_mask"] = inputs["attention_mask"].to(torch.long)
    
    print("\nStarting inference...")
    start_infer = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=20,
            return_dict_in_generate=True,
            output_scores=True
        )
    infer_time = time.time() - start_infer
    print(f"Inference took: {infer_time:.2f}s")
    
    prompt_len = inputs["input_ids"].shape[1]
    decoded = processor.decode(output.sequences[0][prompt_len:], skip_special_tokens=True)
    print(f"Output: {decoded}")
    
    print(f"Tokens generated: {output.sequences.shape[1] - prompt_len}")
    print(f"Time per token: {infer_time / (output.sequences.shape[1] - prompt_len):.2f}s")

if __name__ == "__main__":
    test_speed()
