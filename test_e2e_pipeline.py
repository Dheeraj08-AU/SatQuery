"""
End-to-end pipeline test simulating the exact Streamlit app flow:
1. Upload a single image (via temp file, mimicking Streamlit's file_uploader)
2. Submit a VQA query → verify classify_intent routes correctly, run_single_vqa is called, real answer returned
3. Submit a grounding query → verify classify_intent routes correctly, run_grounding is called, bounding box returned
4. Inspect the ExecutionTrace for each → flag any placeholders or fake data
"""
import sys
import os
import shutil
import tempfile
import time
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from modules.agent_controller import AgentController, TaskType

print("=" * 70)
print("END-TO-END PIPELINE TEST (simulating Streamlit app flow)")
print("=" * 70)

# Simulate Streamlit's file_uploader: copy to a temp path (like save_uploaded_file does)
source_image = "sanity_imgs/real_freeway.jpg"
suffix = os.path.splitext(source_image)[1]
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    with open(source_image, "rb") as src:
        tmp.write(src.read())
    temp_path_1 = tmp.name
print(f"\n[SETUP] Original image: {source_image}")
print(f"[SETUP] Simulated Streamlit temp path: {temp_path_1}")

# Initialize the agent controller (this loads ModelRegistry with lazy loading)
print("\n[INIT] Creating AgentController...")
agent = AgentController()
print("[INIT] AgentController ready.\n")

# =====================================================================
# TEST 1: VQA Query
# =====================================================================
print("=" * 70)
print("TEST 1: VQA Query")
print("=" * 70)

vqa_query = "How many vehicles are visible?"
print(f"Query: '{vqa_query}'")
print(f"Image: {temp_path_1}")

start = time.perf_counter()
trace_vqa = agent.route_and_configure(vqa_query, [temp_path_1], input_mode="single")
elapsed_vqa = time.perf_counter() - start

print(f"\n--- Execution Trace ---")
print(f"  Selected Task:      {trace_vqa.selected_task.value}")
print(f"  Selected Tool:      {trace_vqa.selected_tool}")
print(f"  Parameters:         {trace_vqa.permitted_parameters}")
print(f"  Status:             {trace_vqa.status}")
print(f"  Reasoning Summary:  {trace_vqa.reasoning_summary}")
print(f"  Execution Result:   {trace_vqa.execution_result}")
print(f"  Elapsed Time:       {elapsed_vqa:.2f}s")

# Validate
assert trace_vqa.selected_task == TaskType.SINGLE_VQA, f"FAIL: Expected SINGLE_VQA, got {trace_vqa.selected_task}"
assert trace_vqa.execution_result is not None, "FAIL: execution_result is None"
assert "answer" in trace_vqa.execution_result, "FAIL: No 'answer' key in result"
assert "confidence" in trace_vqa.execution_result, "FAIL: No 'confidence' key in result"
assert trace_vqa.execution_result["answer"] != "", "FAIL: Answer is empty string"
assert trace_vqa.status == "Executed", f"FAIL: Status is '{trace_vqa.status}', not 'Executed'"

# Check for placeholder/fake text in the trace
for placeholder in ["placeholder", "mock", "fake", "dummy", "hardcoded", "canned"]:
    assert placeholder not in trace_vqa.reasoning_summary.lower(), f"FAIL: Found '{placeholder}' in reasoning summary"
    assert placeholder not in str(trace_vqa.execution_result).lower(), f"FAIL: Found '{placeholder}' in execution result"

print("\n[PASS] TEST 1 PASSED: VQA classified correctly, real answer returned, no placeholders detected.")

# What the UI would render:
print(f"\n--- What the UI Would Render ---")
answer = trace_vqa.execution_result.get("answer", "No answer generated.")
real_conf = trace_vqa.execution_result.get("confidence", 0.0) * 100
print(f"  st.success():  '{answer}'")
print(f"  st.metric():   'Model Confidence Score' = '{round(real_conf, 1)}%'")
print(f"  Agent Trace JSON would show:")
print(f"    Selected Task: '{trace_vqa.selected_task.value}'")
print(f"    Selected Tool: '{trace_vqa.selected_tool}'")

# =====================================================================
# TEST 2: Grounding Query (same process, no restart)
# =====================================================================
print("\n" + "=" * 70)
print("TEST 2: Grounding Query (same process, both models will be loaded)")
print("=" * 70)

# Use a different image for grounding
source_image_2 = "sanity_imgs/real_lake.jpg"
with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
    with open(source_image_2, "rb") as src:
        tmp.write(src.read())
    temp_path_2 = tmp.name

grounding_query = "Highlight the water body in this image."
print(f"Query: '{grounding_query}'")
print(f"Image: {temp_path_2} (from {source_image_2})")

start = time.perf_counter()
trace_grnd = agent.route_and_configure(grounding_query, [temp_path_2], input_mode="single")
elapsed_grnd = time.perf_counter() - start

print(f"\n--- Execution Trace ---")
print(f"  Selected Task:      {trace_grnd.selected_task.value}")
print(f"  Selected Tool:      {trace_grnd.selected_tool}")
print(f"  Parameters:         {trace_grnd.permitted_parameters}")
print(f"  Status:             {trace_grnd.status}")
print(f"  Reasoning Summary:  {trace_grnd.reasoning_summary}")
print(f"  Execution Result:   {trace_grnd.execution_result}")
print(f"  Elapsed Time:       {elapsed_grnd:.2f}s")

# Validate
assert trace_grnd.selected_task == TaskType.SINGLE_GROUNDING, f"FAIL: Expected SINGLE_GROUNDING, got {trace_grnd.selected_task}"
assert trace_grnd.execution_result is not None, "FAIL: execution_result is None"
assert "confidence" in trace_grnd.execution_result, "FAIL: No 'confidence' key in result"
assert trace_grnd.status == "Executed", f"FAIL: Status is '{trace_grnd.status}', not 'Executed'"

# Check for placeholder/fake text in the trace  
for placeholder in ["placeholder", "mock", "fake", "dummy", "hardcoded", "canned"]:
    assert placeholder not in trace_grnd.reasoning_summary.lower(), f"FAIL: Found '{placeholder}' in reasoning summary"
    assert placeholder not in str(trace_grnd.execution_result).lower(), f"FAIL: Found '{placeholder}' in execution result"

print("\n[PASS] TEST 2 PASSED: Grounding classified correctly, real result returned, no placeholders detected.")

# What the UI would render:
print(f"\n--- What the UI Would Render ---")
res = trace_grnd.execution_result
box = res.get("box")
label = res.get("label") or "Unknown Entity"
conf = res.get("confidence", 0.0)
if box:
    print(f"  st.success():  'Grounding Result: {label} detected.'")
    print(f"  st.metric():   'Model Confidence Score' = '{round(conf * 100, 1)}%'")
    print(f"  st.image():    Bounding box [{', '.join(f'{c:.1f}' for c in box)}] drawn on image in RED")
else:
    print(f"  st.success():  'Grounding Result: {label} detected.'")
    print(f"  st.metric():   'Model Confidence Score' = '{round(conf * 100, 1)}%'")
    print(f"  st.image():    Original image shown (no box — detection below threshold)")

# =====================================================================
# TEST 3: Check for static/placeholder text in the trace panel
# =====================================================================
print("\n" + "=" * 70)
print("TEST 3: Audit Trace Panel for Fake/Static Content")
print("=" * 70)

issues = []

# Check VQA trace
if "Qwen2-VL-RS" not in trace_vqa.selected_tool:
    issues.append(f"VQA tool name '{trace_vqa.selected_tool}' doesn't match registry entry")

# Check Grounding trace  
if "GroundingDINO-RS" not in trace_grnd.selected_tool:
    issues.append(f"Grounding tool name '{trace_grnd.selected_tool}' doesn't match registry entry")

# Check that the confidence in the GeoJSON export (line 96 of app.py) is hardcoded to 0.942
print("  [AUDIT] GeoJSON confidence: app.py line 96 has hardcoded 0.942 — this is static/fake.")
issues.append("GeoJSON 'confidence' property is hardcoded to 0.942 (app.py:96)")

# Check loaded_tools names vs what the UI actually uses
print(f"  [AUDIT] VQA tool shown in trace: '{trace_vqa.selected_tool}'")
print(f"  [AUDIT] Grounding tool shown in trace: '{trace_grnd.selected_tool}'")

if issues:
    print(f"\n[WARN] Found {len(issues)} issue(s):")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("\n[PASS] No issues found in trace panel.")

# Cleanup temp files
os.unlink(temp_path_1)
os.unlink(temp_path_2)

print("\n" + "=" * 70)
print("END-TO-END TEST COMPLETE")
print("=" * 70)
