# %% [markdown]
# # SatQuery AI - VQA Fine-Tuning (Colab T4 GPU)
# Run this notebook sequentially in Google Colab with a T4 GPU.
# It fine-tunes PaliGemma 3B on the remote-sensing dataset using LoRA in 4-bit, 
# then downloads the lightweight adapter weights to your local machine.

# %% [markdown]
# ### 1. Setup & Installation
# Install the necessary libraries for 4-bit quantization, PEFT, and fine-tuning.

# %%
!pip install -q -U torch transformers peft trl bitsandbytes accelerate datasets huggingface_hub

# %% [markdown]
# ### 2. Data Preparation
# Downloading the dataset directly inside Colab is much faster than uploading from your laptop.
# We'll pull the JSON and Images zip directly from the VRSBench HuggingFace repo.

# %%
import os
import json
import zipfile
import shutil
from huggingface_hub import hf_hub_download
from datasets import Dataset
from PIL import Image

print("Downloading VRSBench dataset from Hugging Face...")
# Download JSON
json_path = hf_hub_download(repo_id="xiang709/VRSBench", filename="VRSBench_train.json", repo_type="dataset")
print(f"Loaded JSON from {json_path}")

# Download Images Zip
zip_path = hf_hub_download(repo_id="xiang709/VRSBench", filename="images.zip", repo_type="dataset")
print(f"Loaded Images ZIP from {zip_path}. Extracting...")

os.makedirs("vrsbench_data", exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("vrsbench_data")
print("Extraction complete.")

# Format data
with open(json_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

formatted_data = []
# Take a subset of 1000 for quick local fine-tuning
# We hold out the last 10 for our manual sanity check eval
NUM_TRAIN = 1000
NUM_EVAL = 10

for i, item in enumerate(raw_data[:NUM_TRAIN + NUM_EVAL]):
    question = item.get('question') or item.get('caption') or 'Describe the visible features.'
    answer = item.get('answer') or item.get('label') or 'Urban and natural land-cover.'
    img_name = item.get('image_path', "").split("/")[-1]
    
    # The zip extracts to a folder called 'images' usually, or directly in the folder.
    # We construct the absolute path:
    img_path = os.path.join("vrsbench_data", "images", img_name)
    if not os.path.exists(img_path):
        # Fallback if structure is flat
        img_path = os.path.join("vrsbench_data", img_name)
        
    formatted_data.append({
        "image_path": img_path,
        "question": question,
        "answer": str(answer)
    })

train_data = formatted_data[:-NUM_EVAL]
eval_data = formatted_data[-NUM_EVAL:]

hf_dataset = Dataset.from_list(train_data)
print(f"Prepared {len(hf_dataset)} training samples and {len(eval_data)} evaluation samples.")

# %% [markdown]
# ### 3. Load Model and Processor (4-bit Quantization)
# We load PaliGemma 3B in 4-bit to fit easily on a 16GB T4 GPU.

# %%
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

model_id = "google/paligemma-3b-pt-224"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_id)

print("Loading model in 4-bit...")
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0}
)

# Freeze vision encoder and multi-modal projector, only train language model attention
for param in model.vision_tower.parameters():
    param.requires_grad = False
for param in model.multi_modal_projector.parameters():
    param.requires_grad = False

# Setup LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# %% [markdown]
# ### 4. Training Loop
# We use a custom collator to format the data for PaliGemma.

# %%
from transformers import TrainingArguments, Trainer

def collate_fn(examples):
    texts = [f"answer en {ex['question']}" for ex in examples]
    labels = [ex['answer'] for ex in examples]
    images = [Image.open(ex['image_path']).convert("RGB") for ex in examples]
    
    # PaliGemma requires prompt and suffix (label) to be processed together for training
    inputs = processor(
        text=texts,
        images=images,
        suffix=labels,
        return_tensors="pt",
        padding="longest",
    )
    
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs["pixel_values"],
        "labels": inputs["labels"],
    }

training_args = TrainingArguments(
    output_dir="./paligemma_satquery_adapter",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    bf16=False, # T4 does not fully support bf16 natively, use fp16
    fp16=True,
    logging_steps=10,
    save_strategy="no",
    optim="paged_adamw_8bit",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_dataset,
    data_collator=collate_fn,
)

print("Starting training...")
trainer.train()

print("Saving LoRA adapter...")
model.save_pretrained("./paligemma_satquery_adapter")
processor.save_pretrained("./paligemma_satquery_adapter")
print("Saved!")

# %% [markdown]
# ### 5. Held-Out Sanity Check (Eval)
# Let's test the model on the 10 samples it didn't see during training.

# %%
model.eval()

print("=== SANITY CHECK EVALUATION ===")
for ex in eval_data:
    img = Image.open(ex['image_path']).convert("RGB")
    prompt = f"answer en {ex['question']}"
    
    inputs = processor(text=prompt, images=img, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    
    # PaliGemma's output includes the prompt, we decode just the newly generated tokens
    prompt_len = inputs["input_ids"].shape[1]
    predicted = processor.decode(output[0][prompt_len:], skip_special_tokens=True)
    
    print(f"Q: {ex['question']}")
    print(f"Truth: {ex['answer']}")
    print(f"Pred : {predicted}")
    print("-" * 40)

# %% [markdown]
# ### 6. Download Adapter Weights
# Run this cell to zip the weights and download them to your local laptop.

# %%
import shutil
from google.colab import files

print("Zipping adapter...")
shutil.make_archive("satquery_vqa_adapter", 'zip', "./paligemma_satquery_adapter")
print("Downloading...")
files.download("satquery_vqa_adapter.zip")
