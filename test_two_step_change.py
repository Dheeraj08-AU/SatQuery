import sys
import os
import time
import warnings
from PIL import Image, ImageDraw
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.dirname(__file__))

from modules.model_registry import ModelRegistry
from google import genai

print("Initializing Registry...")
registry = ModelRegistry()
print("Registry ready.\n")

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")
client = genai.Client(api_key=api_key)

t1_path = "sanity_imgs/real_freeway.jpg"
t2_path = "sanity_imgs/fake_t2.jpg"

if not os.path.exists(t2_path):
    print(f"Creating fake T2 image by painting over a section of {t1_path}...")
    img = Image.open(t1_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    draw.rectangle([w//2, 0, w, h//2], fill="black")
    img.save(t2_path)

desc_query = "Describe the land cover, structures, and notable features in this image."
user_query = "What changed in the land cover?"
params = {}

print("Starting Two-Step Describe-Then-Diff approach...")
start_time = time.time()

# Step 1: Describe T1
print("Running VQA on T1...")
t1_start = time.time()
result_t1 = registry.run_single_vqa(t1_path, desc_query, params)
t1_end = time.time()
desc1 = result_t1.get("answer", "")
print(f"  T1 Description ({t1_end - t1_start:.2f}s): {desc1}")

# Step 2: Describe T2
print("Running VQA on T2...")
t2_start = time.time()
result_t2 = registry.run_single_vqa(t2_path, desc_query, params)
t2_end = time.time()
desc2 = result_t2.get("answer", "")
print(f"  T2 Description ({t2_end - t2_start:.2f}s): {desc2}")

# Step 3: Gemini LLM diff
print("Running LLM comparison with Gemini...")
llm_start = time.time()
prompt = f"Description at time T1: {desc1}\nDescription at time T2: {desc2}\nBased on these two descriptions, what changed? Answer the user's specific question: {user_query}"
response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt
)
llm_end = time.time()
final_answer = response.text
print(f"  Gemini Time: {llm_end - llm_start:.2f}s")

end_time = time.time()

print("\n==================================================")
print("RESULTS:")
print("==================================================")
print(f"Final Answer: {final_answer}")
print(f"Total Inference Time: {end_time - start_time:.2f} seconds")
