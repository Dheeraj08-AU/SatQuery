"""
BigEarthNet-MM builder: real co-registered Sentinel-1 SAR + Sentinel-2 optical.

Why this is the most important dataset in the project
-----------------------------------------------------
The VQA adapter is trained on VRSBench, which is optical RGB aerial imagery.
It has never seen a SAR image. SAR is speckled, single-channel, decibel-scaled,
and semantically inverted relative to optical - open water is DARK because it
scatters specularly away from the sensor, and buildings are BRIGHT because of
wall-ground double-bounce. That is a long way outside the adapter's training
distribution.

The ISRO/SAC evaluation set is Cartosat-2S optical paired with RISAT SAR. Half
of it is SAR. Without adaptation on real radar imagery, the cross-modal task is
being answered by a model that cannot interpret one of its two inputs.

BigEarthNet-MM is the fix, and it is the dataset the problem statement names as
primary: every patch carries co-registered Sentinel-1 (VV, VH) and Sentinel-2
(12-band) observations of the same ground area, with CORINE land-cover labels.

Question generation
-------------------
BigEarthNet ships multi-labels, not questions, so questions are synthesised from
the label set. Two safeguards against the failure that produced the last
degenerate adapter:

  * Every yes/no template is balanced INDEPENDENTLY. Balancing only the global
    answer distribution still lets "is there water?" be 90% yes while the
    overall mix looks healthy, and the model learns per-question priors.
  * Open-ended templates (which land-cover types are present) contribute
    high-entropy targets, so the set cannot collapse to a handful of strings.

Usage
-----
    python training/prep_bigearthnet.py --inspect
    python training/prep_bigearthnet.py --max-patches 12000

Outputs (under --out-dir, default `data/bigearthnet`):
    images/<patch_id>_opt.png, <patch_id>_sar.png
    train.jsonl / val.jsonl
        {"patch_id", "optical", "sar", "question", "answer", "template"}
    build_report.json
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.raster_io import lee_filter, percentile_stretch, to_decibels

# HF dataset ids tried in order. The v2.0 (reBEN) release is preferred.
CANDIDATE_REPOS = [
    "BIFOLD-BigEarthNetv2-0/BigEarthNet-V2.0",
    "blanchon/BigEarthNet-MM",
    "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt",
]

S2_RGB = ("B04", "B03", "B02")
S1_POLS = ("VV", "VH")

# ---------------------------------------------------------------------------
# CORINE nomenclature -> semantic groups used for question generation
# ---------------------------------------------------------------------------

SEMANTIC_GROUPS: Dict[str, Tuple[str, ...]] = {
    "water": (
        "inland waters", "marine waters", "coastal wetlands", "inland wetlands",
        "water bodies", "water courses",
    ),
    "built-up": (
        "urban fabric", "industrial or commercial units", "artificial surfaces",
        "construction sites", "airports", "road and rail networks", "port areas",
    ),
    "forest": (
        "broad-leaved forest", "coniferous forest", "mixed forest",
        "transitional woodland", "transitional woodland/shrub",
        "agro-forestry areas",
    ),
    "agriculture": (
        "arable land", "permanent crops", "pastures",
        "complex cultivation patterns",
        "land principally occupied by agriculture",
        "annual crops", "vineyards", "fruit trees", "rice fields",
    ),
    "bare or sparsely vegetated land": (
        "beaches", "beaches, dunes, sands", "natural grassland",
        "sparsely vegetated areas", "moors and heathland",
        "moors, heathland and sclerophyllous vegetation", "bare rock",
    ),
}

MIN_RECORDS = 4000
MIN_UNIQUE_PATCHES = 800
MIN_UNIQUE_ANSWERS = 10
MAX_TOP_ANSWER_FRACTION = 0.30


class DatasetBuildError(RuntimeError):
    pass


def _fail(msg: str) -> None:
    raise DatasetBuildError(msg)


# ---------------------------------------------------------------------------
# Schema resolution
# ---------------------------------------------------------------------------


def open_stream(repo: Optional[str], split: str):
    """Open a streaming dataset, trying each candidate repo in turn."""
    import datasets

    repos = [repo] if repo else CANDIDATE_REPOS
    errors: List[str] = []
    for candidate in repos:
        try:
            print(f"  trying {candidate} (split={split}, streaming) ...", flush=True)
            ds = datasets.load_dataset(candidate, split=split, streaming=True)
            print(f"  opened {candidate}")
            return ds, candidate
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")

    _fail(
        "Could not open any BigEarthNet dataset. Tried:\n  "
        + "\n  ".join(errors)
        + "\n\nPass --repo <id> with the correct Hugging Face dataset id, or run "
        "with --inspect once a repo opens to see its real schema."
    )


def describe_features(ds, n: int = 2) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        info["features"] = {k: str(v) for k, v in (ds.features or {}).items()}
    except Exception as exc:
        info["features_error"] = str(exc)

    samples = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        rendered = {}
        for k, v in ex.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                rendered[k] = v
            elif isinstance(v, list):
                rendered[k] = f"list[{len(v)}] head={v[:6]}"
            elif isinstance(v, np.ndarray):
                rendered[k] = f"ndarray shape={v.shape} dtype={v.dtype}"
            elif isinstance(v, Image.Image):
                rendered[k] = f"PIL.Image size={v.size} mode={v.mode}"
            elif isinstance(v, dict):
                rendered[k] = f"dict keys={list(v.keys())[:8]}"
            else:
                rendered[k] = f"{type(v).__name__}"
        samples.append(rendered)
    info["samples"] = samples
    return info


def _to_array(value: Any) -> Optional[np.ndarray]:
    """Coerce whatever a column holds into a 2-D float array."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, Image.Image):
        arr = np.asarray(value)
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value)
    elif isinstance(value, dict):
        if "bytes" in value and value["bytes"]:
            try:
                arr = np.asarray(Image.open(io.BytesIO(value["bytes"])))
            except Exception:
                return None
        elif "path" in value and value["path"]:
            try:
                arr = np.asarray(Image.open(value["path"]))
            except Exception:
                return None
        elif "array" in value:
            arr = np.asarray(value["array"])
        else:
            return None
    else:
        return None

    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        arr = arr[..., 0]
    if arr.ndim != 2:
        return None
    return arr.astype(np.float64)


def resolve_columns(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Work out where the optical bands, SAR bands, labels and id live.

    Handles the three layouts these releases use: one column per band
    ("B04", "VV", ...), a single stacked array per sensor, or an encoded image
    per sensor.
    """
    keys = list(example.keys())
    lower = {k.lower(): k for k in keys}
    plan: Dict[str, Any] = {"layout": None, "warnings": []}

    # --- per-band columns ---
    s2_hits = [lower.get(b.lower()) for b in S2_RGB]
    s1_hits = [lower.get(p.lower()) for p in S1_POLS]
    if all(s2_hits) and any(s1_hits):
        plan["layout"] = "per_band"
        plan["s2_rgb"] = s2_hits
        plan["s1"] = [h for h in s1_hits if h]
    else:
        # --- stacked arrays / encoded images ---
        s2_key = next(
            (lower[k] for k in ("s2", "sentinel2", "optical", "image_s2", "bands_s2", "s2_image")
             if k in lower),
            None,
        )
        s1_key = next(
            (lower[k] for k in ("s1", "sentinel1", "sar", "image_s1", "bands_s1", "s1_image")
             if k in lower),
            None,
        )
        if s2_key and s1_key:
            plan["layout"] = "stacked"
            plan["s2"] = s2_key
            plan["s1"] = s1_key

    label_key = next(
        (lower[k] for k in ("labels", "label", "new_labels", "labels_19", "classes", "multilabel")
         if k in lower),
        None,
    )
    id_key = next(
        (lower[k] for k in ("patch_id", "id", "name", "patch", "s2v1_name", "__key__")
         if k in lower),
        None,
    )
    plan["labels"] = label_key
    plan["id"] = id_key

    if plan["layout"] is None:
        _fail(
            "Could not locate optical and SAR columns.\n"
            f"Columns present: {keys}\n"
            "Expected either per-band columns (B04/B03/B02 and VV/VH) or stacked "
            "columns (s2/sentinel2/optical and s1/sentinel1/sar).\n"
            "Run with --inspect and check data/bigearthnet/schema_report.json."
        )
    if not label_key:
        _fail(
            f"Could not locate a labels column. Columns present: {keys}. "
            "Questions are generated from land-cover labels, so this is required."
        )
    return plan


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_optical(bands: Sequence[np.ndarray]) -> Optional[Image.Image]:
    """Percentile-stretch three Sentinel-2 bands into an RGB image."""
    planes = []
    for band in bands:
        if band is None or band.size == 0:
            return None
        u8, _ = percentile_stretch(band, lo_pct=2.0, hi_pct=98.0)
        planes.append(u8)
    if len(planes) < 3:
        planes = [planes[0]] * 3
    shape = planes[0].shape
    if any(p.shape != shape for p in planes):
        return None
    return Image.fromarray(np.stack(planes[:3], axis=-1), mode="RGB")


def render_sar(bands: Sequence[np.ndarray], speckle_radius: int = 2) -> Optional[Image.Image]:
    """
    Decibel-convert, speckle-filter and stretch Sentinel-1 VV/VH.

    Identical treatment to `raster_io.load_as_rgb(modality="sar")`, so training
    inputs match what inference produces for an uploaded SAR GeoTIFF. If these
    two diverged, the adapter would be trained on a rendering it never sees at
    serving time.
    """
    planes: List[np.ndarray] = []
    for band in bands:
        if band is None or band.size == 0:
            continue
        is_int = np.issubdtype(band.dtype, np.integer)
        filtered = lee_filter(np.nan_to_num(band, nan=0.0), radius=speckle_radius)
        db, _label, _clip = to_decibels(filtered, is_integer_dn=is_int)
        u8, _ = percentile_stretch(db, lo_pct=2.0, hi_pct=98.0)
        planes.append(u8)

    if not planes:
        return None
    if len(planes) == 1:
        rgb = np.stack([planes[0]] * 3, axis=-1)
    else:
        shape = planes[0].shape
        if planes[1].shape != shape:
            return None
        ratio = np.clip(
            planes[0].astype(np.float64) - planes[1].astype(np.float64) + 128.0, 0, 255
        ).astype(np.uint8)
        rgb = np.stack([planes[0], planes[1], ratio], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def extract_pair(
    example: Dict[str, Any], plan: Dict[str, Any]
) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    if plan["layout"] == "per_band":
        s2 = [_to_array(example.get(k)) for k in plan["s2_rgb"]]
        s1 = [_to_array(example.get(k)) for k in plan["s1"]]
        return render_optical(s2), render_sar([b for b in s1 if b is not None])

    s2_raw = example.get(plan["s2"])
    s1_raw = example.get(plan["s1"])

    def split_stack(raw: Any, rgb_indices: Optional[Sequence[int]]) -> List[np.ndarray]:
        if isinstance(raw, Image.Image):
            arr = np.asarray(raw).astype(np.float64)
            if arr.ndim == 3:
                return [arr[..., i] for i in range(min(3, arr.shape[2]))]
            return [arr]
        arr = np.asarray(raw, dtype=np.float64) if raw is not None else None
        if arr is None:
            return []
        if arr.ndim == 2:
            return [arr]
        if arr.ndim == 3:
            # (C, H, W) or (H, W, C)
            if arr.shape[0] <= 16 and arr.shape[0] < arr.shape[-1]:
                stack = [arr[i] for i in range(arr.shape[0])]
            else:
                stack = [arr[..., i] for i in range(arr.shape[-1])]
            if rgb_indices:
                try:
                    return [stack[i] for i in rgb_indices]
                except IndexError:
                    return stack[:3]
            return stack
        return []

    # 12-band Sentinel-2 ordering B01..B12: true colour is indices 3, 2, 1.
    s2_bands = split_stack(s2_raw, (3, 2, 1) if _stack_depth(s2_raw) >= 10 else None)
    s1_bands = split_stack(s1_raw, None)[:2]
    return render_optical(s2_bands[:3]), render_sar(s1_bands)


def _stack_depth(raw: Any) -> int:
    try:
        arr = np.asarray(raw)
        if arr.ndim == 3:
            return min(arr.shape[0], arr.shape[-1]) if arr.shape[0] > arr.shape[-1] else arr.shape[0]
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------


def normalise_labels(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple, np.ndarray)):
        items = list(raw)
    else:
        return []
    out: List[str] = []
    for item in items:
        if isinstance(item, (int, np.integer)):
            continue  # class indices without a names mapping are unusable
        text = str(item).strip().lower()
        if text:
            out.append(text)
    return out


def semantic_presence(labels: Sequence[str]) -> Dict[str, bool]:
    joined = " | ".join(labels)
    presence: Dict[str, bool] = {}
    for group, keywords in SEMANTIC_GROUPS.items():
        presence[group] = any(k in joined for k in keywords)
    return presence


def generate_qa(labels: Sequence[str], presence: Dict[str, bool]) -> List[Dict[str, str]]:
    """Candidate (question, answer, template) triples for one patch."""
    qa: List[Dict[str, str]] = []

    def add(template: str, question: str, answer: str) -> None:
        qa.append({"template": template, "question": question, "answer": answer})

    yn = lambda flag: "yes" if flag else "no"  # noqa: E731

    add(
        "presence_water",
        "Are water-covered regions present in this area?",
        yn(presence["water"]),
    )
    add(
        "presence_builtup",
        "Are built-up regions present in this area?",
        yn(presence["built-up"]),
    )
    add(
        "presence_forest",
        "Is there forest cover in this scene?",
        yn(presence["forest"]),
    )
    add(
        "presence_agriculture",
        "Is there agricultural land in this scene?",
        yn(presence["agriculture"]),
    )
    add(
        "sar_water",
        "Does the SAR image indicate open water through specular (dark) returns?",
        yn(presence["water"]),
    )
    add(
        "sar_builtup",
        "Does the SAR image show strong double-bounce returns indicating buildings?",
        yn(presence["built-up"]),
    )
    add(
        "urban_rural",
        "Is this predominantly an urban or a rural area?",
        "urban" if presence["built-up"] else "rural",
    )

    present_groups = sorted(g for g, flag in presence.items() if flag)
    if present_groups:
        add(
            "joint_builtup_water",
            "Use the optical and SAR images together to identify built-up and "
            "water-covered regions.",
            _describe_builtup_water(presence),
        )
        add(
            "land_cover_groups",
            "Which broad land-cover types are present in this area?",
            ", ".join(present_groups) + ".",
        )

    if labels:
        add(
            "land_cover_detail",
            "What land-cover classes are visible in this area?",
            ", ".join(sorted(set(labels))[:6]) + ".",
        )

    return qa


def _describe_builtup_water(presence: Dict[str, bool]) -> str:
    b = presence["built-up"]
    w = presence["water"]
    if b and w:
        return (
            "Both are present: built-up areas appear as bright, high-texture "
            "double-bounce returns in the SAR image, and water-covered regions "
            "appear dark due to specular scattering."
        )
    if b:
        return (
            "Built-up areas are present, visible as bright double-bounce returns "
            "in the SAR image. No significant water-covered regions are present."
        )
    if w:
        return (
            "Water-covered regions are present, visible as dark specular returns "
            "in the SAR image. No significant built-up areas are present."
        )
    return "Neither built-up areas nor water-covered regions are present in this area."


# ---------------------------------------------------------------------------
# Balancing / splitting / gates
# ---------------------------------------------------------------------------


def balance_per_template(
    records: List[Dict[str, Any]], max_fraction: float, seed: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """
    Balance each question template's answer distribution independently.

    Balancing only the global mix is not enough. If "Are water-covered regions
    present?" is 90% yes, the model learns to answer yes to that exact question
    regardless of the pixels, and the global distribution can still look fine
    because other templates dilute it.
    """
    by_template: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        by_template[r["template"]].append(r)

    rng = random.Random(seed)
    kept: List[Dict[str, Any]] = []
    report: Dict[str, Dict[str, int]] = {}

    for template, rows in by_template.items():
        by_answer: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in rows:
            by_answer[r["answer"].strip().lower()].append(r)
        for group in by_answer.values():
            rng.shuffle(group)

        counts = {a: len(g) for a, g in by_answer.items()}
        # Open-ended templates have near-unique answers; capping them by
        # fraction would delete almost everything, so only cap templates with a
        # small closed answer set.
        if len(counts) <= 8:
            for _ in range(100):
                total = sum(counts.values())
                cap = max(1, int(total * max(max_fraction, 1.0 / len(counts))))
                changed = False
                for a in counts:
                    if counts[a] > cap:
                        counts[a] = cap
                        changed = True
                if not changed:
                    break

        for answer, group in by_answer.items():
            kept.extend(group[: counts[answer]])
        report[template] = counts

    rng.shuffle(kept)
    return kept, report


def split_by_patch(
    records: List[Dict[str, Any]], val_fraction: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    patches = sorted({r["patch_id"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(patches)
    n_val = max(1, int(len(patches) * val_fraction))
    val_patches = set(patches[:n_val])
    train = [r for r in records if r["patch_id"] not in val_patches]
    val = [r for r in records if r["patch_id"] in val_patches]
    return train, val


def assert_quality(records: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    n = len(records)
    patches = {r["patch_id"] for r in records}
    answers = collections.Counter(r["answer"].strip().lower() for r in records)
    templates = collections.Counter(r["template"] for r in records)

    stats = {
        "records": n,
        "unique_patches": len(patches),
        "unique_answers": len(answers),
        "templates": dict(templates),
        "top_answers": answers.most_common(10),
    }

    problems: List[str] = []
    if n < MIN_RECORDS:
        problems.append(f"only {n} records (need >= {MIN_RECORDS})")
    if len(patches) < MIN_UNIQUE_PATCHES:
        problems.append(
            f"only {len(patches)} unique patches (need >= {MIN_UNIQUE_PATCHES})"
        )
    if len(answers) < MIN_UNIQUE_ANSWERS:
        problems.append(f"only {len(answers)} distinct answers (need >= {MIN_UNIQUE_ANSWERS})")
    if n:
        top_ans, top_cnt = answers.most_common(1)[0]
        frac = top_cnt / n
        stats["top_answer_fraction"] = round(frac, 4)
        if frac > MAX_TOP_ANSWER_FRACTION + 0.05:
            problems.append(
                f"answer {top_ans!r} is {frac:.1%} of the set (max ~{MAX_TOP_ANSWER_FRACTION:.0%})"
            )

    # Per-template skew is the failure mode global stats hide.
    for template in templates:
        rows = [r for r in records if r["template"] == template]
        counts = collections.Counter(r["answer"].strip().lower() for r in rows)
        if len(counts) == 2:
            top = counts.most_common(1)[0][1] / len(rows)
            if top > 0.70:
                problems.append(
                    f"template {template!r} is {top:.0%} one answer - the model "
                    f"will learn that question's prior instead of reading the image"
                )

    if problems:
        detail = "\n".join(f"  - {p}" for p in problems)
        _fail(
            f"Quality gate FAILED for '{label}':\n{detail}\n\n"
            f"Statistics: {json.dumps(stats, indent=2, default=str)}"
        )
    return stats


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):>7,} rows -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the BigEarthNet-MM optical+SAR dataset.")
    ap.add_argument("--out-dir", default="data/bigearthnet")
    ap.add_argument("--repo", default=None, help="Override the HF dataset id.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--inspect", action="store_true", help="Dump the real schema and exit.")
    ap.add_argument("--max-patches", type=int, default=12000)
    ap.add_argument("--max-qa-per-patch", type=int, default=4)
    ap.add_argument("--max-answer-fraction", type=float, default=0.30)
    ap.add_argument("--val-fraction", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    ds, repo = open_stream(args.repo, args.split)

    if args.inspect:
        info = describe_features(ds)
        info["repo"] = repo
        print(json.dumps(info, indent=2, default=str))
        (out_dir / "schema_report.json").write_text(
            json.dumps(info, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nWrote {out_dir / 'schema_report.json'}")
        return 0

    rng = random.Random(args.seed)
    plan: Optional[Dict[str, Any]] = None
    records: List[Dict[str, Any]] = []
    failures: collections.Counter = collections.Counter()
    kept_patches = 0

    print(f"\nStreaming up to {args.max_patches:,} patches from {repo} ...")
    for i, example in enumerate(ds):
        if kept_patches >= args.max_patches:
            break

        if plan is None:
            plan = resolve_columns(example)
            print(f"  detected layout: {plan['layout']}")
            print(f"  labels column  : {plan['labels']}")
            print(f"  id column      : {plan['id']}")

        labels = normalise_labels(example.get(plan["labels"]))
        if not labels:
            failures["no_usable_labels"] += 1
            continue

        patch_id = str(example.get(plan["id"]) or f"patch{i:08d}").replace("/", "_")

        try:
            opt_img, sar_img = extract_pair(example, plan)
        except Exception as exc:
            failures[f"extract:{type(exc).__name__}"] += 1
            continue

        if opt_img is None:
            failures["optical_render_failed"] += 1
            continue
        if sar_img is None:
            failures["sar_render_failed"] += 1
            continue
        if opt_img.size != sar_img.size:
            sar_img = sar_img.resize(opt_img.size, Image.BILINEAR)

        opt_path = image_dir / f"{patch_id}_opt.png"
        sar_path = image_dir / f"{patch_id}_sar.png"
        opt_img.save(opt_path)
        sar_img.save(sar_path)

        presence = semantic_presence(labels)
        candidates = generate_qa(labels, presence)
        rng.shuffle(candidates)
        for qa in candidates[: args.max_qa_per_patch]:
            records.append(
                {
                    "patch_id": patch_id,
                    "optical": str(opt_path),
                    "sar": str(sar_path),
                    "labels": labels,
                    **qa,
                }
            )

        kept_patches += 1
        if kept_patches % 500 == 0:
            print(f"    {kept_patches:,} patches / {len(records):,} QA pairs", flush=True)

    if not records:
        _fail(
            "No usable records were produced. Failure breakdown: "
            f"{dict(failures)}. Run with --inspect to see the real schema."
        )

    total_seen = kept_patches + sum(failures.values())
    print(f"\nKept {kept_patches:,} patches of {total_seen:,} seen; {len(records):,} QA pairs")
    for reason, count in failures.most_common():
        print(f"  skipped {reason}: {count:,}")

    print(f"\nBalancing each template to <= {args.max_answer_fraction:.0%} per answer ...")
    balanced, balance_report = balance_per_template(records, args.max_answer_fraction, args.seed)
    print(f"  {len(records):,} -> {len(balanced):,}")

    print("\nRunning quality gates ...")
    stats = assert_quality(balanced, "BigEarthNet-MM")
    print(
        f"  OK: {stats['records']:,} records, {stats['unique_patches']:,} patches, "
        f"{stats['unique_answers']:,} distinct answers, "
        f"top answer {stats.get('top_answer_fraction', 0):.1%}"
    )

    print("\nSplitting by patch and writing ...")
    train, val = split_by_patch(balanced, args.val_fraction, args.seed)
    overlap = {r["patch_id"] for r in train} & {r["patch_id"] for r in val}
    if overlap:
        _fail(f"train/val leakage: {len(overlap)} shared patches")

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)

    used = {r["patch_id"] for r in balanced}
    removed = 0
    for f in image_dir.glob("*_*.png"):
        stem = f.stem.rsplit("_", 1)[0]
        if stem not in used:
            f.unlink()
            removed += 1
    if removed:
        print(f"  pruned {removed:,} unused image files")

    report = {
        "repo": repo,
        "split": args.split,
        "layout": plan,
        "patches_kept": kept_patches,
        "skipped": dict(failures),
        "qa_before_balance": len(records),
        "qa_after_balance": len(balanced),
        "per_template_caps": balance_report,
        "stats": stats,
        "train": len(train),
        "val": len(val),
        "split_by": "patch_id",
        "seed": args.seed,
    }
    (out_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"\nWrote {out_dir / 'build_report.json'}")
    print("\nBuild complete. All quality gates passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except DatasetBuildError as exc:
        print(f"\n{'!' * 70}\nDATASET BUILD FAILED\n{'!' * 70}\n{exc}\n", file=sys.stderr)
        sys.exit(2)
