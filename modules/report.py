"""
Human-readable report export for SatQuery AI.

The problem statement asks for "downloadable reports" alongside visual evidence,
confidence information and execution summaries. GeoJSON and raw JSON cover the
machine-readable half, but neither is something an analyst or a judge can open
and read.

This builds a single self-contained HTML file: images are embedded as base64
data URIs, styling is inline, and there are no external requests. It opens in
any browser offline and prints to PDF from there, which avoids taking on a PDF
toolchain dependency for one feature.
"""

from __future__ import annotations

import base64
import html
import io
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from PIL import Image

_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; padding: 32px; background: #f6f7f9; color: #16181d;
       font: 14px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.wrap { max-width: 980px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 17px; margin: 32px 0 12px; padding-bottom: 6px;
     border-bottom: 1px solid #d9dde3; }
h3 { font-size: 14px; margin: 18px 0 6px; }
.sub { color: #5b6470; margin: 0 0 24px; }
.card { background: #fff; border: 1px solid #e2e6ec; border-radius: 10px;
        padding: 18px 20px; margin-bottom: 16px; }
.answer { background: #eef7ef; border-left: 4px solid #2e7d43; padding: 14px 16px;
          border-radius: 6px; white-space: pre-wrap; font-size: 15px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
figure { margin: 0; }
figure img { width: 100%; border: 1px solid #d9dde3; border-radius: 8px; display: block; }
figcaption { color: #5b6470; font-size: 12px; margin-top: 6px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #e8ebef;
         vertical-align: top; }
th { color: #5b6470; font-weight: 600; width: 34%; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
pre { background: #f2f4f7; border: 1px solid #e2e6ec; border-radius: 6px;
      padding: 12px; overflow-x: auto; }
.badge { display: inline-block; padding: 2px 9px; border-radius: 999px;
         font-size: 11px; font-weight: 600; letter-spacing: 0.02em; }
.ok { background: #e3f2e7; color: #1f6b36; }
.warn { background: #fdf1dc; color: #8a5a10; }
.bad { background: #fbe4e4; color: #9c2626; }
.neutral { background: #eceff3; color: #4a525c; }
.metrics { display: flex; gap: 28px; flex-wrap: wrap; margin: 4px 0 10px; }
.metric .v { font-size: 22px; font-weight: 650; }
.metric .k { color: #5b6470; font-size: 12px; text-transform: uppercase;
             letter-spacing: 0.04em; }
.note { color: #5b6470; font-size: 12px; margin-top: 6px; }
ul.warnings { margin: 8px 0 0; padding-left: 18px; }
ul.warnings li { color: #8a5a10; margin-bottom: 4px; }
footer { color: #7a828d; font-size: 12px; margin-top: 36px; text-align: center; }
"""


def _b64(img: Image.Image, max_edge: int = 900) -> str:
    """PNG data URI, downscaled so a multi-image report stays a sane size."""
    im = img.convert("RGB")
    if max(im.size) > max_edge:
        scale = max_edge / max(im.size)
        im = im.resize((int(im.size[0] * scale), int(im.size[1] * scale)), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=False)


def _kv_table(data: Dict[str, Any], skip: tuple = ()) -> str:
    rows = []
    for key, value in data.items():
        if key in skip:
            continue
        if isinstance(value, (dict, list)):
            rendered = f"<pre>{_esc(json.dumps(value, indent=2, default=str))}</pre>"
        else:
            rendered = f"<code>{_esc(value)}</code>"
        rows.append(f"<tr><th>{_esc(key)}</th><td>{rendered}</td></tr>")
    return f"<table>{''.join(rows)}</table>" if rows else "<p class='note'>None.</p>"


IMAGE_CAPTIONS = {
    "input": "Model input (as rendered from the source raster)",
    "composite": "Composite handed to the vision-language model",
    "t1": "Time 1 (T1)",
    "t2": "Time 2 (T2)",
    "optical": "Optical / multispectral",
    "sar": "SAR (decibels, speckle-filtered)",
    "change_overlay": "Change map (red) over T2",
    "sar_overlay": "SAR surface classification: blue = water, red = built-up",
}


def build_html_report(
    trace: Any,
    geojson: Optional[Dict[str, Any]] = None,
    title: str = "SatQuery AI analysis report",
) -> str:
    """
    Render an `ExecutionTrace` as a standalone HTML document.

    Takes the trace object rather than its dict form so the PIL images survive.
    """
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = trace.as_dict()

    # -- header ------------------------------------------------------------
    parts: List[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>{_esc(title)}</title>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<style>{_CSS}</style></head><body><div class='wrap'>",
        f"<h1>{_esc(title)}</h1>",
        f"<p class='sub'>Generated {generated} &middot; "
        f"task: <strong>{_esc(data['selected_task'])}</strong></p>",
    ]

    # -- query and answer --------------------------------------------------
    status = data["status"]
    status_class = "ok" if status == "executed" else "bad"
    parts.append("<div class='card'>")
    parts.append(f"<h3>Query</h3><p><em>{_esc(data['query'])}</em></p>")
    parts.append(f"<h3>Answer</h3><div class='answer'>{_esc(data['answer'])}</div>")

    conf_pct = (
        "n/a" if data["confidence_type"] == "deterministic"
        else f"{data['confidence'] * 100:.1f}%"
    )
    parts.append(
        "<div class='metrics'>"
        f"<div class='metric'><div class='v'>{conf_pct}</div>"
        f"<div class='k'>{_esc(data['confidence_type'])}</div></div>"
        f"<div class='metric'><div class='v'>{data['total_latency_ms'] / 1000:.1f} s</div>"
        "<div class='k'>total latency</div></div>"
        f"<div class='metric'><div class='v'>{len(data['steps'])}</div>"
        "<div class='k'>tools executed</div></div>"
        "</div>"
    )
    parts.append(f"<span class='badge {status_class}'>{_esc(status)}</span>")

    if data.get("warnings"):
        seen = list(dict.fromkeys(data["warnings"]))
        items = "".join(f"<li>{_esc(w)}</li>" for w in seen)
        parts.append(f"<ul class='warnings'>{items}</ul>")
    parts.append("</div>")

    # -- visual evidence ---------------------------------------------------
    images = trace.images() if hasattr(trace, "images") else {}
    if images:
        parts.append("<h2>Visual evidence</h2><div class='grid'>")
        for name, img in images.items():
            caption = IMAGE_CAPTIONS.get(name, name.replace("_", " ").capitalize())
            parts.append(
                f"<figure><img src='{_b64(img)}' alt='{_esc(caption)}'>"
                f"<figcaption>{_esc(caption)}</figcaption></figure>"
            )
        parts.append("</div>")

    # -- routing -----------------------------------------------------------
    parts.append("<h2>Routing decision</h2>")
    parts.append(f"<div class='card'>{_kv_table(data['router'])}</div>")

    # -- validation --------------------------------------------------------
    validation = data.get("input_validation") or {}
    coreg = validation.get("coregistration")
    parts.append("<h2>Input validation</h2><div class='card'>")
    if coreg:
        badge = "ok" if coreg.get("verified") else "warn"
        if not coreg.get("ok_to_proceed"):
            badge = "bad"
        parts.append(
            f"<p><span class='badge {badge}'>{_esc(coreg['status'])}</span></p>"
            f"<p>{_esc(coreg['message'])}</p>"
        )
    for i, meta in enumerate(validation.get("images", []), 1):
        parts.append(f"<h3>Image {i}</h3>")
        parts.append(
            _kv_table(meta, skip=("path", "band_descriptions", "error"))
            if meta.get("is_valid")
            else f"<p class='note'>{_esc(meta.get('error'))}</p>"
        )
    parts.append("</div>")

    # -- execution -----------------------------------------------------------
    parts.append("<h2>Auditable execution summary</h2>")
    parts.append(
        "<p class='note'>Model identifiers and parameters below are what actually "
        "ran, reported back by each tool after execution.</p>"
    )
    for step in data["steps"]:
        tool = step["tool"]
        adapted = tool["remote_sensing_adapted"]
        badge = (
            "<span class='badge ok'>RS-adapted</span>"
            if adapted
            else "<span class='badge warn'>not RS-adapted</span>"
        )
        cached = "<span class='badge neutral'>cached</span>" if step["cache_hit"] else ""
        parts.append(
            f"<div class='card'><h3>Step {step['order']}: {_esc(tool['display_name'])} "
            f"{badge} {cached}</h3>"
            f"<p class='note'>{_esc(step['purpose'])}</p>"
            f"<table>"
            f"<tr><th>Model / algorithm</th><td><code>{_esc(tool['model_id'])}</code></td></tr>"
            f"<tr><th>Kind</th><td>{_esc(tool['kind'])}</td></tr>"
            f"<tr><th>Latency</th><td>{step['latency_ms'] / 1000:.2f} s</td></tr>"
            f"<tr><th>Output</th><td>{_esc(step['answer'])}</td></tr>"
            f"</table>"
        )
        if tool.get("notes"):
            parts.append(f"<p class='note'>{_esc(tool['notes'])}</p>")
        parts.append("<h3>Parameters applied</h3>")
        parts.append(_kv_table(step["applied_parameters"]))
        if step.get("evidence"):
            parts.append("<h3>Evidence</h3>")
            parts.append(
                f"<pre>{_esc(json.dumps(step['evidence'], indent=2, default=str))}</pre>"
            )
        if step.get("preprocessing"):
            parts.append("<h3>Preprocessing provenance</h3>")
            parts.append(
                f"<pre>{_esc(json.dumps(step['preprocessing'], indent=2, default=str))}</pre>"
            )
        parts.append("</div>")

    # -- geometry ----------------------------------------------------------
    if geojson and geojson.get("features"):
        georef = sum(
            1 for f in geojson["features"] if f.get("properties", {}).get("georeferenced")
        )
        parts.append("<h2>Exported geometry</h2><div class='card'>")
        parts.append(
            f"<p>{len(geojson['features'])} feature(s), {georef} with EPSG:4326 "
            f"coordinates. Features derived from non-georeferenced inputs carry "
            f"null geometry rather than invented coordinates.</p>"
        )
        parts.append(f"<pre>{_esc(json.dumps(geojson, indent=2, default=str))}</pre>")
        parts.append("</div>")

    parts.append(
        "<footer>SatQuery AI &middot; agentic vision-language assistant for "
        "multimodal remote-sensing image analysis</footer>"
    )
    parts.append("</div></body></html>")
    return "".join(parts)


__all__ = ["build_html_report"]
