"""
Real-image sanity check for Task 2 GroundingDINO grounding.

Downloads 2 genuine remote-sensing image crops from public sources:
  - A Sentinel-2 water body scene (lake visible)
  - An urban/built-up scene (roads + buildings visible)

Runs run_grounding with realistic queries, prints all raw numbers,
and saves annotated output images for manual inspection.

NOT part of the automated test suite — run this manually.
Usage:
    python sanity_check_real_images.py
"""

import os
import sys
import warnings
import urllib.request
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from modules.model_registry import ModelRegistry

# ─────────────────────────────────────────────────────────────────────────────
# Two real remote-sensing image sources (public domain / CC)
# ─────────────────────────────────────────────────────────────────────────────

# 1. UC Merced Land Use dataset – "river" class chip (256×256 px aerial)
#    Source: http://weegee.vision.ucmerced.edu/datasets/landuse.html (public domain)
IMAGES = [
    {
        "url": "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse/Images/river/river00.tif",
        "local": "sanity_imgs/river_scene.tif",
        "query": "highlight the water body",
        "description": "UC Merced river chip (aerial, ~256px)",
        "params": {"box_threshold": 0.20, "text_threshold": 0.15},
    },
    {
        "url": "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse/Images/buildings/buildings00.tif",
        "local": "sanity_imgs/urban_scene.tif",
        "query": "highlight the built-up area",
        "description": "UC Merced buildings chip (aerial, ~256px)",
        "params": {"box_threshold": 0.20, "text_threshold": 0.15},
    },
]

# Fallback: if UCMerced is unreachable, use NASA WorldWind sample tiles (JPEG)
FALLBACK_IMAGES = [
    {
        "url": "https://eoimages.gsfc.nasa.gov/images/imagerecords/150000/150001/NDVI_M_2020-01_rgb_360.jpg",
        "local": "sanity_imgs/ndvi_world.jpg",
        "query": "highlight the water body",
        "description": "NASA NDVI world map (river/ocean regions)",
        "params": {"box_threshold": 0.15, "text_threshold": 0.10},
    },
    {
        "url": "https://eoimages.gsfc.nasa.gov/images/imagerecords/150000/150002/world.topo.200407.3x1350x675.jpg",
        "local": "sanity_imgs/world_topo.jpg",
        "query": "detect the road or urban area",
        "description": "NASA World topography (land/water visible)",
        "params": {"box_threshold": 0.15, "text_threshold": 0.10},
    },
]

OUTPUT_DIR = "sanity_imgs"


def download_image(url: str, local_path: str) -> bool:
    """Download an image file with a 15s timeout. Returns True on success."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
        print(f"  [cached] {local_path}")
        return True
    try:
        print(f"  Downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "SatQueryAI-SanityCheck/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp, open(local_path, "wb") as f:
            f.write(resp.read())
        size_kb = os.path.getsize(local_path) / 1024
        print(f"  → saved {local_path} ({size_kb:.0f} KB)")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return False


def draw_annotated(img_path: str, box, label: str, confidence: float,
                   out_path: str) -> None:
    """Draw the detection box on the image and save it."""
    img = Image.open(img_path).convert("RGB")
    if box is not None:
        draw = ImageDraw.Draw(img)
        draw.rectangle(box, outline="red", width=3)
        text = f"{label} ({confidence:.3f})"
        draw.text((box[0] + 4, max(0, box[1] - 14)), text, fill="red")
    img.save(out_path)
    print(f"  → annotated image saved: {out_path}")


def run_check(registry: ModelRegistry, img_cfg: dict) -> dict:
    """Run grounding on one real image and return the result dict."""
    path = img_cfg["local"]
    query = img_cfg["query"]
    params = img_cfg["params"]
    desc = img_cfg["description"]

    print()
    print("=" * 65)
    print(f"IMAGE : {desc}")
    print(f"PATH  : {path}")
    print(f"QUERY : '{query}'")
    print(f"THRESH: box={params['box_threshold']}  text={params['text_threshold']}")
    print("-" * 65)

    result = registry.run_grounding(path, query, params)

    print()
    print(f"  box        : {result['box']}")
    if result["box"]:
        box = result["box"]
        print(f"             [xmin={box[0]:.1f}  ymin={box[1]:.1f}  "
              f"xmax={box[2]:.1f}  ymax={box[3]:.1f}]")
    print(f"  label      : {result['label']}")
    print(f"  confidence : {result['confidence']:.4f}")
    print(f"  status     : {result['status']}")
    print("=" * 65)

    # Save annotated image
    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{stem}_annotated.png")
    draw_annotated(path, result["box"], str(result["label"] or "no detection"),
                   result["confidence"], out_path)

    return result


def main():
    print("Loading ModelRegistry (GroundingDINO — CPU float32)...")
    registry = ModelRegistry()
    print("Registry ready.\n")

    # Try primary images first, fall back if download fails
    to_check = []
    for primary, fallback in zip(IMAGES, FALLBACK_IMAGES):
        ok = download_image(primary["url"], primary["local"])
        if ok:
            to_check.append(primary)
        else:
            print(f"  Primary download failed, trying fallback...")
            ok2 = download_image(fallback["url"], fallback["local"])
            if ok2:
                to_check.append(fallback)
            else:
                print("  Both sources failed — skipping this image.")

    if not to_check:
        print("\nERROR: Could not download any real images. Check network.")
        return

    results = []
    for cfg in to_check:
        r = run_check(registry, cfg)
        results.append({"image": cfg["local"], "query": cfg["query"], **r})

    # Summary table
    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    for r in results:
        box_str = (f"[{r['box'][0]:.1f}, {r['box'][1]:.1f}, "
                   f"{r['box'][2]:.1f}, {r['box'][3]:.1f}]"
                   if r["box"] else "None")
        print(f"  {os.path.basename(r['image']):<30s}  "
              f"conf={r['confidence']:.4f}  box={box_str}")
    print()
    print(f"Annotated images written to: {os.path.abspath(OUTPUT_DIR)}/")
    print("Open *_annotated.png to visually verify box placement.")


if __name__ == "__main__":
    main()
