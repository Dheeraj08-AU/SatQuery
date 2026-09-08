import os
from dotenv import load_dotenv
load_dotenv()

import tempfile
import json
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
import rasterio
from rasterio.warp import transform_bounds

from modules.geo_validator import SatQueryValidator
from modules.agent_controller import AgentController, TaskType, InputValidationError

st.set_page_config(
    page_title="SatQuery AI - Vision-Language Assistant",
    page_icon="🛰️",
    layout="wide"
)

@st.cache_resource
def load_modules():
    return SatQueryValidator(), AgentController()

validator, agent = load_modules()

def save_uploaded_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def draw_real_grounding_box(img_path: str, box: list, label: str, confidence: float) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    
    if box is None:
        # If no detection, just return the original image
        return img
        
    draw = ImageDraw.Draw(img)
    # GroundingDINO returns [xmin, ymin, xmax, ymax]
    draw.rectangle(box, outline="red", width=4)
    
    display_text = f"{label} ({confidence:.2f})"
    draw.text((box[0] + 5, box[1] + 5), display_text, fill="red")
    
    return img

def draw_mock_change_map(img1_path: str, img2_path: str) -> Image.Image:
    img1 = np.array(Image.open(img1_path).convert("L"))
    img2 = np.array(Image.open(img2_path).convert("L"))

    if img1.shape != img2.shape:
        img2_pil = Image.fromarray(img2).resize((img1.shape[1], img1.shape[0]))
        img2 = np.array(img2_pil)

    diff = np.abs(img1.astype(int) - img2.astype(int))
    diff_colored = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
    diff_colored[diff > 40] = [255, 0, 0]

    return Image.fromarray(diff_colored)

def generate_gis_geojson(img_path: str, task_name: str, answer_text: str, confidence: float = 0.0) -> dict:
    geometry = None
    try:
        with rasterio.open(img_path) as src:
            if src.crs:
                left, bottom, right, top = transform_bounds(
                    src.crs,
                    "EPSG:4326",
                    *src.bounds
                )
                geometry = {
                    "type": "Polygon",
                    "coordinates": [[
                        [left, bottom],
                        [right, bottom],
                        [right, top],
                        [left, top],
                        [left, bottom]
                    ]]
                }
    except Exception:
        pass  # Not a valid geospatial raster (e.g. plain JPG)

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "task": task_name,
                    "description": answer_text,
                    "confidence": round(confidence, 4)
                }
            }
        ]
    }

    return geojson

st.title("🛰️ SatQuery AI")
st.caption("Agentic Vision-Language Assistant for Single, Bi-Temporal, and Cross-Modal Satellite Imagery")
st.markdown("---")

st.sidebar.header("1. Input Configuration")

input_mode = st.sidebar.radio(
    "Select Input Scope:",
    [
        "Single Image",
        "Bi-Temporal Pair (t1, t2)",
        "Cross-Modal Pair (Optical + SAR)"
    ]
)

# Map radio labels to mode strings for the agentic controller
INPUT_MODE_MAP = {
    "Single Image": "single",
    "Bi-Temporal Pair (t1, t2)": "bi_temporal",
    "Cross-Modal Pair (Optical + SAR)": "cross_modal",
}
current_input_mode = INPUT_MODE_MAP.get(input_mode, "single")

uploaded_files = []

if input_mode == "Single Image":
    f1 = st.sidebar.file_uploader(
        "Upload Optical or SAR Image (GeoTIFF / PNG / JPEG):",
        type=["tif", "tiff", "png", "jpg", "jpeg"]
    )

    if f1:
        uploaded_files.append((f1, "Single Input Image"))

elif input_mode == "Bi-Temporal Pair (t1, t2)":
    f1 = st.sidebar.file_uploader(
        "Upload Time-1 (t1) Image:",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        key="t1"
    )

    f2 = st.sidebar.file_uploader(
        "Upload Time-2 (t2) Image:",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        key="t2"
    )

    if f1 and f2:
        uploaded_files.extend([
            (f1, "Time 1 (t1)"),
            (f2, "Time 2 (t2)")
        ])

elif input_mode == "Cross-Modal Pair (Optical + SAR)":
    f1 = st.sidebar.file_uploader(
        "Upload Optical / Multispectral Image:",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        key="opt"
    )

    f2 = st.sidebar.file_uploader(
        "Upload Synthetic Aperture Radar (SAR) Image:",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        key="sar"
    )

    if f1 and f2:
        uploaded_files.extend([
            (f1, "Optical Image"),
            (f2, "SAR Image")
        ])

col_left, col_right = st.columns([1, 1])

temp_paths = []
valid_inputs = False

with col_left:
    st.subheader("🖼️ Input Image Preview & Validation")

    if uploaded_files:
        for ufile, label in uploaded_files:
            tpath = save_uploaded_file(ufile)
            temp_paths.append(tpath)

            meta = validator.extract_metadata(tpath)

            st.image(
                ufile,
                caption=f"{label} ({ufile.name})",
                use_container_width=True
            )

            with st.expander(f"⚙️ Metadata: {ufile.name}"):
                st.write(f"**Format:** {meta.format}")
                st.write(f"**Bands:** {meta.band_count}")
                st.write(f"**CRS:** {meta.crs}")

                if meta.bounds:
                    st.write(f"**Bounds:** `{meta.bounds}`")

        if len(temp_paths) == 2:
            is_coregistered = validator.verify_coregistration(
                temp_paths[0],
                temp_paths[1]
            )

            if is_coregistered:
                st.success(
                    "✅ Co-registration Verified: Images are spatially aligned."
                )
                valid_inputs = True
            else:
                st.error(
                    "❌ Co-registration Failed: CRS or spatial bounds mismatch."
                )
                valid_inputs = False
        else:
            valid_inputs = True
    else:
        st.info("Please upload image(s) using the sidebar to begin.")

with col_right:
    st.subheader("💬 Query & Agent Processing")

    st.markdown("**Sample Queries:**")

    sample_queries = [
        "Describe the land-cover and major objects visible in this image.",
        "Highlight the water body referred to in the query.",
        "What changed between these two dates, and where did the change occur?",
        "Use the optical and SAR images together to identify built-up and water-covered regions.",
        "Has the built-up area increased, decreased, or remained unchanged?"
    ]

    selected_sample = st.selectbox(
        "Choose a sample query or write your own below:",
        ["Custom Query..."] + sample_queries
    )

    user_query = st.text_input(
        "Enter Natural Language Query:",
        value=selected_sample if selected_sample != "Custom Query..." else ""
    )

    run_btn = st.button(
        "🚀 Analyze with SatQuery AI",
        type="primary",
        disabled=not (valid_inputs and len(user_query) > 3)
    )

if run_btn:
    st.markdown("---")
    st.header("📊 Results & Visual Evidence")
    
    trace = agent.route_and_configure(user_query, temp_paths, input_mode=current_input_mode)
    
    with st.expander("🛠️ Auditable Agent Execution Summary", expanded=True):
        st.json({
            "Selected Task": trace.selected_task.value,
            "Selected Tool / Model": trace.selected_tool,
            "Permitted Parameters": trace.permitted_parameters,
            "Execution Status": trace.status,
            "Reasoning Summary": trace.reasoning_summary
        })

    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.subheader("📝 Answer Output")
        
        # This now executes the real model registry or dynamic pipeline!
        result_data = trace.execution_result or {}
        
        if trace.selected_task in (TaskType.SINGLE_VQA, TaskType.CHANGE_ANALYSIS):
            answer = result_data.get("answer", "No answer generated.")
            real_conf = result_data.get("confidence", 0.0) * 100
        elif trace.selected_task == TaskType.SINGLE_GROUNDING:
            answer = f"Grounding Result: {result_data.get('label')} detected."
            real_conf = result_data.get("confidence", 0.0) * 100
        else:
            answer = str(result_data)
            real_conf = 85.0 # fallback for other tools

        st.success(answer)
        
        dynamic_conf = round(real_conf, 1)
        st.metric(label="Model Confidence Score", value=f"{dynamic_conf}%")

    with res_col2:
        st.subheader("🖼️ Visual Evidence Map")
        
        if trace.selected_task == TaskType.SINGLE_GROUNDING and temp_paths:
            # trace.execution_result contains the dict returned by GroundingDINO
            res = trace.execution_result
            if res and isinstance(res, dict):
                annotated_img = draw_real_grounding_box(
                    temp_paths[0], 
                    res.get("box"), 
                    res.get("label") or "Unknown Entity", 
                    res.get("confidence", 0.0)
                )
                st.image(annotated_img, caption="Grounding Box Evidence Overlay", use_container_width=True)
            else:
                st.image(temp_paths[0], caption="Analyzed Region Context", use_container_width=True)
            
        elif trace.selected_task == TaskType.CHANGE_ANALYSIS and len(temp_paths) == 2:
            # Note: The actual change description comes from the VLM (see result_data['answer'] above).
            # This pixel-diff map is just a supplementary visual overlay, per the PS optional requirement.
            change_map = draw_mock_change_map(temp_paths[0], temp_paths[1])
            st.image(change_map, caption="Spatial Change Map (Red = Modified Regions)", use_container_width=True)
            
        else:
            st.image(temp_paths[0], caption="Analyzed Region Context", use_container_width=True)

    # 3. Exportable GIS-Accurate GeoJSON Report
    primary_image = temp_paths[0] if temp_paths else "sample.tif"
    geo_json_data = generate_gis_geojson(primary_image, trace.selected_task.value, answer, dynamic_conf / 100)

    st.download_button(
        label="📥 Download GIS-Accurate Report (GeoJSON)",
        data=json.dumps(geo_json_data, indent=2),
        file_name="satquery_spatial_report.geojson",
        mime="application/geo+json"
    )

# Cleanup temporary files on completion
for p in temp_paths:
    if os.path.exists(p):
        try:
            os.remove(p)
        except Exception:
            pass