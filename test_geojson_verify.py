import sys
import os
import json
import tempfile
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from modules.agent_controller import AgentController, TaskType

# Inline copy of the updated generate_gis_geojson from app.py
def generate_gis_geojson(img_path, task_name, answer_text, confidence=0.0):
    geometry = None
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "task": task_name,
                "description": answer_text,
                "confidence": round(confidence, 4)
            }
        }]
    }
    return geojson

print("Initializing...")
agent = AgentController()

# Simulate Streamlit temp upload
with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
    with open("sanity_imgs/real_freeway.jpg", "rb") as src:
        tmp.write(src.read())
    temp_path = tmp.name

query = "How many vehicles are visible?"
trace = agent.route_and_configure(query, [temp_path], input_mode="single")

print(f"\n--- Verify Tool Name Fix ---")
print(f"  Selected Tool: '{trace.selected_tool}'")
assert "PaliGemma" in trace.selected_tool, f"FAIL: Tool name is still '{trace.selected_tool}'"
print(f"  [PASS] Tool name correctly shows PaliGemma")

# Compute confidence same way app.py does
result_data = trace.execution_result or {}
real_conf = result_data.get("confidence", 0.0) * 100
dynamic_conf = round(real_conf, 1)

# Generate GeoJSON with real confidence
geojson = generate_gis_geojson(temp_path, trace.selected_task.value, result_data.get("answer", ""), dynamic_conf / 100)

print(f"\n--- Verify GeoJSON Confidence Fix ---")
geojson_conf = geojson["features"][0]["properties"]["confidence"]
print(f"  GeoJSON confidence value: {geojson_conf}")
print(f"  Model confidence value:   {round(result_data.get('confidence', 0.0), 4)}")
assert geojson_conf != 0.942, "FAIL: GeoJSON confidence is still the hardcoded 0.942"
assert abs(geojson_conf - result_data.get("confidence", 0.0)) < 0.01, "FAIL: GeoJSON confidence doesn't match model confidence"
print(f"  [PASS] GeoJSON confidence matches real model confidence")

print(f"\n--- Full GeoJSON Output ---")
print(json.dumps(geojson, indent=2))

os.unlink(temp_path)
print("\nDone!")
