"""Run GroundingDINO on 2 photorealistic generated satellite images and print raw results."""

import warnings, sys, os
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.dirname(__file__))

from modules.model_registry import ModelRegistry
from PIL import Image, ImageDraw

registry = ModelRegistry()
print("Registry ready.\n")

CHECKS = [
    {
        "path": "sanity_imgs/real_water_body.jpg",
        "query": "highlight the water body",
        "params": {"box_threshold": 0.20, "text_threshold": 0.15},
        "desc": "Photorealistic river/lake satellite scene",
    },
    {
        "path": "sanity_imgs/real_water_body.jpg",
        "query": "highlight the river",
        "params": {"box_threshold": 0.15, "text_threshold": 0.10},
        "desc": "Same image — 'river' query (more specific)",
    },
    {
        "path": "sanity_imgs/real_urban.jpg",
        "query": "highlight the built-up area",
        "params": {"box_threshold": 0.20, "text_threshold": 0.15},
        "desc": "Photorealistic urban grid satellite scene",
    },
    {
        "path": "sanity_imgs/real_urban.jpg",
        "query": "detect the road",
        "params": {"box_threshold": 0.15, "text_threshold": 0.10},
        "desc": "Same image — 'road' query",
    },
]


def draw_box(img_path, box, label, conf, out_path):
    img = Image.open(img_path).convert("RGB")
    if box:
        d = ImageDraw.Draw(img)
        d.rectangle(box, outline="red", width=6)
        d.text((box[0] + 5, max(0, box[1] - 18)), f"{label} ({conf:.3f})", fill="red")
    img.save(out_path)


results = []
for c in CHECKS:
    print("=" * 65)
    print(f"IMAGE : {c['desc']}")
    print(f"QUERY : \"{c['query']}\"")
    print(f"THRESH: box={c['params']['box_threshold']}  text={c['params']['text_threshold']}")
    print("-" * 65)

    r = registry.run_grounding(c["path"], c["query"], c["params"])
    box = r["box"]
    box_str = (
        f"[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]" if box else "None"
    )
    print(f"  box        : {box_str}")
    print(f"  label      : {r['label']}")
    print(f"  confidence : {r['confidence']:.4f}")
    print(f"  status     : {r['status']}")
    if box:
        w = box[2] - box[0]
        h = box[3] - box[1]
        print(f"  box size   : {w:.0f}w x {h:.0f}h px  (image is 1024x1024)")
    print()

    stem = os.path.splitext(os.path.basename(c["path"]))[0]
    q_slug = c["query"].replace(" ", "_")[:20]
    out = f"sanity_imgs/{stem}__{q_slug}.png"
    draw_box(c["path"], box, str(r["label"] or "no-detection"), r["confidence"], out)
    print(f"  annotated  : {out}")
    results.append(r)
    print("=" * 65)
    print()

print()
print("FINAL SUMMARY")
print("=" * 65)
for c, r in zip(CHECKS, results):
    b = r["box"]
    bs = (
        f"[{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}]" if b else "None"
    )
    status = "DETECTED" if b else "no-detection"
    print(
        f"  [{status:<11}]  conf={r['confidence']:.4f}  "
        f"box={bs}"
    )
    print(f"               query='{c['query']}'")
print()
print(f"Annotated PNGs written to: {os.path.abspath('sanity_imgs')}/")
