"""
SatQuery AI - interactive agentic remote-sensing assistant.

Changes from the previous UI, and why:

* Previews and analysis both go through `modules.raster_io`. Previously
  `st.image(uploaded_file)` handed the raw bytes to Streamlit, so a 12-band
  Sentinel-2 GeoTIFF or a float32 SAR scene rendered as an error or a black
  box - the user could not see what the model was being given.

* Modality is declared, not guessed. The Optical and SAR upload slots tell the
  backend which file is which, so SAR gets decibel conversion and speckle
  filtering and optical does not.

* Co-registration is reported honestly. The old check returned True for any
  pair containing a PNG, so every demo displayed "Co-registration Verified"
  without a check having run. There are now five distinct outcomes and only two
  of them say "verified".

* Confidence carries its type. A VLM sequence likelihood and a detector box
  score are different quantities; they are no longer averaged into one
  percentage, and there is no hardcoded 85% fallback.

* The execution summary lists the tools that actually ran, with real model
  identifiers and the parameters they actually consumed.

* Reports export the real geometry - grounding boxes and change regions
  projected to EPSG:4326 - not just the image footprint.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from PIL import Image, ImageDraw

from modules.agent_controller import AgentController, ExecutionTrace
from modules.geo_validator import CoregistrationStatus, SatQueryValidator

from modules.model_registry import CONFIDENCE_TYPES
from modules.raster_io import RasterReadError, load_as_rgb
from modules.report import build_html_report

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="\N{SATELLITE}",
    layout="wide",
)

ACCEPTED = ["tif", "tiff", "png", "jpg", "jpeg"]

MODE_LABELS = {
    "Single Image": "single",
    "Bi-Temporal Pair (T1, T2)": "bi_temporal",
    "Cross-Modal Pair (Optical + SAR)": "cross_modal",
}

SAMPLE_QUERIES = [
    "Describe the land-cover and major objects visible in this image.",
    "Highlight the water body referred to in the query.",
    "What changed between these two dates, and where did the change occur?",
    "Use the optical and SAR images together to identify built-up and water-covered regions.",
    "Has the built-up area increased, decreased, or remained unchanged?",
]

STATUS_RENDER = {
    CoregistrationStatus.IDENTICAL_GRID: ("success", "Co-registration verified"),
    CoregistrationStatus.ALIGNED_WITHIN_TOLERANCE: ("success", "Co-registration verified"),
    CoregistrationStatus.RESAMPLING_REQUIRED: ("warning", "Same area, different grid"),
    CoregistrationStatus.NOT_GEOREFERENCED: ("info", "Alignment not verifiable"),
    CoregistrationStatus.DIMENSION_MISMATCH: ("warning", "Alignment not verifiable"),
    CoregistrationStatus.NO_OVERLAP: ("error", "Different areas"),
    CoregistrationStatus.UNREADABLE: ("error", "Unreadable input"),
}


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Initialising agent and model registry...")
def load_agent() -> AgentController:
    # Never raises on a missing API key: the controller falls back to the
    # offline rule-based router and records that it did so.
    return AgentController()


@st.cache_resource
def load_validator() -> SatQueryValidator:
    return SatQueryValidator()


def upload_dir() -> Path:
    if "upload_dir" not in st.session_state:
        st.session_state.upload_dir = tempfile.mkdtemp(prefix="satquery_")
    return Path(st.session_state.upload_dir)


def persist_upload(uploaded_file) -> str:
    """
    Write an upload to a stable, content-addressed path.

    Streamlit reruns the whole script on every widget interaction. The previous
    version created a fresh NamedTemporaryFile each rerun and deleted every temp
    file at the bottom of the script, so paths churned constantly and the
    download buttons operated on files that had already been removed.
    """
    data = uploaded_file.getvalue()
    digest = hashlib.sha256(data).hexdigest()[:16]
    suffix = Path(uploaded_file.name).suffix or ".bin"
    path = upload_dir() / f"{digest}{suffix}"
    if not path.exists():
        path.write_bytes(data)
    return str(path)


@st.cache_data(show_spinner=False)
def render_preview(path: str, modality: str) -> Tuple[Optional[bytes], str, Dict[str, Any]]:
    """
    Render any supported raster to PNG bytes for display.

    Cached on (path, modality) alone, which is safe because `persist_upload`
    writes content-addressed filenames - a different image is a different path.
    """
    try:
        img, report = load_as_rgb(path, modality=modality)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), report.summary(), report.as_dict()
    except RasterReadError as exc:
        return None, f"Could not render: {exc}", {}
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Could not render: {type(exc).__name__}: {exc}", {}


# ---------------------------------------------------------------------------
# Geometry export
# ---------------------------------------------------------------------------


def pixel_box_to_lonlat(path: str, box: Sequence[float]) -> Optional[List[List[float]]]:
    """
    Convert a pixel-space box [x0, y0, x1, y1] into an EPSG:4326 polygon ring.

    Returns None for non-georeferenced input rather than fabricating
    coordinates. raster_io renders at native resolution for grounding, so pixel
    indices map straight through the source transform.
    """
    try:
        import rasterio
        from rasterio.warp import transform as warp_transform

        with rasterio.open(path) as src:
            if not src.crs:
                return None
            x0, y0, x1, y1 = box
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            xs, ys = [], []
            for px, py in corners:
                cx, cy = src.transform * (px, py)
                xs.append(cx)
                ys.append(cy)
            lons, lats = warp_transform(src.crs, "EPSG:4326", xs, ys)
            ring = [[float(lon), float(lat)] for lon, lat in zip(lons, lats)]
            ring.append(ring[0])
            return ring
    except Exception:
        return None


def footprint_polygon(path: str) -> Optional[List[List[float]]]:
    try:
        import rasterio
        from rasterio.warp import transform_bounds

        with rasterio.open(path) as src:
            if not src.crs:
                return None
            left, bottom, right, top = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        return [
            [left, bottom],
            [right, bottom],
            [right, top],
            [left, top],
            [left, bottom],
        ]
    except Exception:
        return None


def build_geojson(trace: ExecutionTrace, paths: List[str]) -> Dict[str, Any]:
    """
    GeoJSON carrying the actual detected geometry.

    The previous export wrote a single polygon of the whole image footprint and
    discarded `result["box"]` - the one real geometry the system produced.
    """
    features: List[Dict[str, Any]] = []
    primary = paths[0] if paths else None

    if primary:
        ring = footprint_polygon(primary)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]} if ring else None,
                "properties": {
                    "feature_type": "scene_footprint",
                    "task": trace.selected_task.value,
                    "answer": trace.answer,
                    "confidence": round(trace.confidence, 4),
                    "confidence_type": trace.confidence_type,
                    "georeferenced": ring is not None,
                },
            }
        )

    for step in trace.steps:
        for det in step.result.evidence.get("detections", []) or []:
            ring = pixel_box_to_lonlat(primary, det["box"]) if primary else None
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]} if ring else None,
                    "properties": {
                        "feature_type": "grounding_detection",
                        "label": det.get("label"),
                        "phrase": det.get("phrase"),
                        "detector_score": det.get("score"),
                        "pixel_box": det["box"],
                        "area_fraction": det.get("area_fraction"),
                        "tool": step.result.tool.model_id,
                        "georeferenced": ring is not None,
                    },
                }
            )

        for region in step.result.evidence.get("regions", []) or []:
            ring = pixel_box_to_lonlat(primary, region["bbox"]) if primary else None
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]} if ring else None,
                    "properties": {
                        "feature_type": "change_region",
                        "direction": region.get("direction"),
                        "area_px": region.get("area_px"),
                        "area_fraction": region.get("area_fraction"),
                        "pixel_box": region["bbox"],
                        "tool": step.result.tool.model_id,
                        "georeferenced": ring is not None,
                    },
                }
            )

    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "generator": "SatQuery AI",
            "query": trace.query,
            "task": trace.selected_task.value,
            "router_backend": trace.router.get("backend"),
            "tools_executed": [s.result.tool.model_id for s in trace.steps],
        },
    }


def draw_detections(image: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    palette = ["#ff2d2d", "#ffb300", "#00d0ff", "#7cff5c", "#ff59f8", "#ffffff"]
    width = max(2, int(min(out.size) * 0.005))

    for i, det in enumerate(detections):
        colour = palette[i % len(palette)]
        x0, y0, x1, y1 = det["box"]
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=width)
        tag = f"{det.get('label', '?')} {det.get('score', 0):.2f}"
        ty = max(0, y0 - 14)
        draw.rectangle([x0, ty, x0 + 8 * len(tag), ty + 14], fill=colour)
        draw.text((x0 + 3, ty + 1), tag, fill="black")
    return out


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("\N{SATELLITE} SatQuery AI")
st.caption(
    "Agentic vision-language assistant for single, bi-temporal and cross-modal "
    "remote-sensing imagery"
)

agent = load_agent()
validator = load_validator()

with st.sidebar:
    st.header("1. Input configuration")
    mode_label = st.radio("Input scope", list(MODE_LABELS.keys()))
    input_mode = MODE_LABELS[mode_label]

    uploads: List[Tuple[Any, str, str]] = []  # (file, label, modality)

    if input_mode == "single":
        modality = st.selectbox(
            "Image modality",
            ["auto", "optical", "sar"],
            help=(
                "SAR is converted to decibels and speckle-filtered before display "
                "and inference. 'auto' guesses from band count, dtype and filename."
            ),
        )
        f = st.file_uploader("Image (GeoTIFF / PNG / JPEG)", type=ACCEPTED)
        if f:
            uploads.append((f, "Input image", modality))

    elif input_mode == "bi_temporal":
        f1 = st.file_uploader("Time 1 (T1)", type=ACCEPTED, key="t1")
        f2 = st.file_uploader("Time 2 (T2)", type=ACCEPTED, key="t2")
        modality = st.selectbox("Pair modality", ["auto", "optical", "sar"])
        if f1 and f2:
            uploads.append((f1, "Time 1 (T1)", modality))
            uploads.append((f2, "Time 2 (T2)", modality))

    else:
        f1 = st.file_uploader("Optical / multispectral", type=ACCEPTED, key="opt")
        f2 = st.file_uploader("SAR", type=ACCEPTED, key="sar")
        if f1 and f2:
            uploads.append((f1, "Optical", "optical"))
            uploads.append((f2, "SAR", "sar"))

    st.divider()
    st.header("2. Query")
    sample = st.selectbox("Sample queries", ["Custom..."] + SAMPLE_QUERIES)
    query = st.text_area(
        "Natural-language query",
        value="" if sample == "Custom..." else sample,
        height=90,
    )

    st.divider()
    backend = "offline rule-based router"
    if agent.gemini_client is not None:
        backend = "cloud LLM, with offline fallback"
    st.caption(f"**Router:** {backend}")
    if agent.gemini_unavailable_reason:
        st.caption(f"Cloud classifier unavailable: {agent.gemini_unavailable_reason}")
    st.caption(f"**Compute:** {agent.registry.device} / {agent.registry.dtype}")
    if not agent.registry.cache_enabled:
        st.caption("**Cache:** disabled (live inference)")


paths: List[str] = [persist_upload(f) for f, _, _ in uploads]
modalities: List[str] = [m for _, _, m in uploads]

left, right = st.columns([1, 1])

with left:
    st.subheader("Input preview and validation")

    if not uploads:
        st.info("Upload imagery in the sidebar to begin.")
    else:
        for (ufile, label, modality), path in zip(uploads, paths):
            png, summary, report = render_preview(path, modality)
            if png is None:
                st.error(f"**{label}** - {summary}")
                continue
            st.image(png, caption=f"{label} - {ufile.name}", use_container_width=True)
            st.caption(summary)

            meta = validator.extract_metadata(path)
            with st.expander(f"Metadata - {ufile.name}"):
                if not meta.is_valid:
                    st.error(meta.error)
                else:
                    c1, c2 = st.columns(2)
                    c1.metric("Format", meta.format)
                    c2.metric("Bands", meta.band_count)
                    st.write(f"**Data type:** `{meta.dtype}`")
                    st.write(f"**Size:** {meta.width} x {meta.height} px")
                    st.write(f"**CRS:** {meta.crs or 'not georeferenced'}")
                    if meta.resolution:
                        st.write(f"**Resolution:** {meta.resolution}")
                    if meta.bounds:
                        st.write(f"**Bounds:** `{meta.bounds}`")
                    if report.get("warnings"):
                        for w in report["warnings"]:
                            st.warning(w)

        if len(paths) == 2:
            coreg = validator.check_coregistration(paths[0], paths[1])
            kind, headline = STATUS_RENDER.get(coreg.status, ("info", coreg.status.value))
            getattr(st, kind)(f"**{headline}** - {coreg.message}")
            if coreg.pixel_offset is not None:
                st.caption(
                    f"Footprint offset {max(coreg.pixel_offset):.2f} px "
                    f"(tolerance {coreg.tolerance_px:g} px), "
                    f"overlap {coreg.overlap_fraction:.1%}"
                )

with right:
    st.subheader("Query and execution")

    expected = {"single": 1, "bi_temporal": 2, "cross_modal": 2}[input_mode]
    ready = len(paths) == expected and len(query.strip()) > 3

    if len(paths) and len(paths) != expected:
        st.warning(f"This mode needs {expected} image(s); {len(paths)} uploaded.")

    run = st.button("Analyse", type="primary", disabled=not ready, use_container_width=True)

    if run:
        with st.spinner("Routing query, selecting tools, executing..."):
            st.session_state.trace = agent.run(query, paths, input_mode=input_mode)
            st.session_state.trace_paths = paths

    trace: Optional[ExecutionTrace] = st.session_state.get("trace")

    if trace is not None:
        st.markdown("**Routing decision**")
        st.write(
            f"Task: `{trace.selected_task.value}` - decided by "
            f"`{trace.router.get('backend')}` "
            f"(classification confidence {trace.router.get('classification_confidence', 0):.2f})"
        )
        st.caption(trace.router.get("rationale") or "")


if trace := st.session_state.get("trace"):
    st.divider()
    st.header("Results and visual evidence")

    if trace.status not in ("executed",):
        st.error(f"Status: `{trace.status}` - {trace.answer}")

    res_left, res_right = st.columns([1, 1])

    with res_left:
        st.subheader("Answer")
        if trace.answer:
            st.success(trace.answer)

        c1, c2 = st.columns(2)
        c1.metric(
            "Confidence",
            f"{trace.confidence * 100:.1f}%" if trace.confidence_type != "deterministic" else "n/a",
        )
        c2.metric("Total latency", f"{trace.total_latency_ms / 1000:.1f} s")
        st.caption(
            f"**{trace.confidence_type}** - "
            f"{CONFIDENCE_TYPES.get(trace.confidence_type, 'no description')}"
        )

        if any(s.result.cache_hit for s in trace.steps):
            st.info(
                "One or more answers were served from cache rather than computed "
                "live. Set `SATQUERY_DISABLE_CACHE=1` to force live inference."
            )

        for w in dict.fromkeys(trace.warnings):
            st.warning(w)

    with res_right:
        st.subheader("Visual evidence")
        images = trace.images()
        shown = False

        detections: List[Dict[str, Any]] = []
        for step in trace.steps:
            detections.extend(step.result.evidence.get("detections", []) or [])

        if detections and "input" in images:
            st.image(
                draw_detections(images["input"], detections),
                caption=f"{len(detections)} grounded region(s)",
                use_container_width=True,
            )
            shown = True

        if "change_overlay" in images:
            st.image(
                images["change_overlay"],
                caption="Change map (red) over T2",
                use_container_width=True,
            )
            shown = True

        if "sar_overlay" in images:
            st.image(
                images["sar_overlay"],
                caption="SAR surface classification - blue = water, red = built-up",
                use_container_width=True,
            )
            shown = True

        if "composite" in images:
            st.image(
                images["composite"],
                caption="Model input composite (exactly what the VLM saw)",
                use_container_width=True,
            )
            shown = True
        elif "input" in images and not shown:
            st.image(images["input"], caption="Model input", use_container_width=True)
            shown = True

        if not shown:
            st.info("No visual evidence was produced for this task.")

    st.subheader("Auditable execution summary")
    st.caption(
        "Model identifiers and parameters below are what actually ran, reported "
        "back by each tool after execution."
    )

    for step in trace.steps:
        r = step.result
        badge = "RS-adapted" if r.tool.remote_sensing_adapted else "NOT RS-adapted"
        with st.expander(
            f"Step {step.order}: {r.tool.display_name}  -  {badge}"
            f"  -  {r.latency_ms / 1000:.1f}s"
            + ("  (cached)" if r.cache_hit else ""),
            expanded=step.order == 1,
        ):
            st.write(f"**Purpose:** {step.purpose}")
            st.write(f"**Model / algorithm:** `{r.tool.model_id}`")
            st.write(f"**Kind:** {r.tool.kind}")
            if r.tool.notes:
                st.caption(r.tool.notes)
            if r.error:
                st.error(r.error)
            st.write("**Parameters applied**")
            st.json(r.applied_parameters, expanded=False)
            if r.preprocessing:
                st.write("**Preprocessing provenance**")
                st.json(r.preprocessing, expanded=False)
            if r.evidence:
                st.write("**Evidence**")
                st.json(r.evidence, expanded=False)

    st.subheader("Downloadable report")
    dl1, dl2, dl3 = st.columns(3)

    report_paths = st.session_state.get("trace_paths", [])
    geojson = build_geojson(trace, report_paths)

    dl1.download_button(
        "Download geometry (GeoJSON)",
        data=json.dumps(geojson, indent=2),
        file_name="satquery_geometry.geojson",
        mime="application/geo+json",
        use_container_width=True,
    )
    dl2.download_button(
        "Download full execution report (JSON)",
        data=json.dumps(trace.as_dict(), indent=2, default=str),
        file_name="satquery_execution_report.json",
        mime="application/json",
        use_container_width=True,
    )
    dl3.download_button(
        "Download readable report (HTML)",
        data=build_html_report(trace, geojson),
        file_name="satquery_report.html",
        mime="text/html",
        use_container_width=True,
        help="Self-contained: images embedded, no external requests. Print to PDF from the browser.",
    )

    georeferenced = sum(
        1 for f in geojson["features"] if f["properties"].get("georeferenced")
    )
    st.caption(
        f"GeoJSON contains {len(geojson['features'])} feature(s), "
        f"{georeferenced} with real EPSG:4326 geometry. Features from "
        f"non-georeferenced inputs carry null geometry rather than invented "
        f"coordinates."
    )
