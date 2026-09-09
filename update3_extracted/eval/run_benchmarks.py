"""
Benchmark evaluation harness for SatQuery AI.

There was no way to score this system. Final judging uses prescribed public
test splits with normalised metrics, and until now the project could not state
a single number about its own accuracy - which also meant a collapsed adapter
could ship undetected.

This runs the held-out splits produced by `data_prep.py` and
`training/prep_cdvqa.py` through the real registry and reports:

  * exact match (normalised) and token F1
  * distinct-prediction ratio and most-common-prediction share - the
    degeneracy detectors. A model that answers "no" to everything can score
    respectably on exact match if the split is imbalanced; these two numbers
    expose it immediately.
  * grounding IoU@0.5 and mean IoU against VRSBench referring boxes
  * latency, and how many answers came from cache rather than live inference

Usage
-----
    python eval/run_benchmarks.py --task all --limit 200
    python eval/run_benchmarks.py --task vqa --limit 500 --no-cache
    python eval/run_benchmarks.py --task grounding --box-threshold 0.25
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import string
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_PUNCT = str.maketrans("", "", string.punctuation)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def normalize_answer(text: str) -> str:
    text = str(text).lower().strip().translate(_PUNCT)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def summarise_text_task(
    preds: List[str], golds: List[str], latencies: List[float], cache_hits: int
) -> Dict[str, Any]:
    n = len(preds)
    if n == 0:
        return {"n": 0}

    exact = sum(normalize_answer(p) == normalize_answer(g) for p, g in zip(preds, golds))
    f1 = sum(token_f1(p, g) for p, g in zip(preds, golds))

    norm_preds = [normalize_answer(p) for p in preds]
    counts = collections.Counter(norm_preds)
    top_pred, top_count = counts.most_common(1)[0]

    gold_counts = collections.Counter(normalize_answer(g) for g in golds)
    majority_baseline = gold_counts.most_common(1)[0][1] / n

    metrics = {
        "n": n,
        "exact_match": round(exact / n, 4),
        "token_f1": round(f1 / n, 4),
        "majority_class_baseline": round(majority_baseline, 4),
        "distinct_predictions": len(counts),
        "distinct_ratio": round(len(counts) / n, 4),
        "most_common_prediction": top_pred,
        "most_common_fraction": round(top_count / n, 4),
        "gold_distinct_ratio": round(len(gold_counts) / n, 4),
        "mean_latency_ms": round(sum(latencies) / n, 1) if latencies else 0.0,
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / n, 4),
    }

    # A model can beat exact match while being useless. Both checks matter.
    metrics["collapsed"] = bool(
        metrics["distinct_ratio"] < 0.10 or metrics["most_common_fraction"] > 0.60
    )
    metrics["beats_majority_baseline"] = bool(
        metrics["exact_match"] > metrics["majority_class_baseline"]
    )
    return metrics


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def read_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------


def run_vqa(registry, rows: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    preds, golds, lat = [], [], []
    cache_hits = 0
    failures = 0

    for i, row in enumerate(rows, 1):
        res = registry.run_single_vqa(row["image"], row["question"], params)
        if res.error:
            failures += 1
            continue
        preds.append(res.answer)
        golds.append(row["answer"])
        lat.append(res.latency_ms)
        cache_hits += int(res.cache_hit)
        if i % 25 == 0:
            print(f"    {i}/{len(rows)}", flush=True)

    out = summarise_text_task(preds, golds, lat, cache_hits)
    out["tool_failures"] = failures
    out["samples"] = [
        {"question": r["question"], "gold": g, "pred": p}
        for r, g, p in list(zip(rows, golds, preds))[:10]
    ]
    return out


def run_rsvqa(registry, rows: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    RSVQA, scored overall and per question type.

    The official protocol reports presence / count / comparison / rural-urban
    separately, and it matters: a model can look respectable overall while being
    at chance on counting, because presence questions dominate the split.
    """
    preds, golds, lat, types = [], [], [], []
    cache_hits = 0
    failures = 0

    for i, row in enumerate(rows, 1):
        res = registry.run_single_vqa(row["image"], row["question"], params)
        if res.error:
            failures += 1
            continue
        preds.append(res.answer)
        golds.append(row["answer"])
        types.append(row.get("type", "unknown"))
        lat.append(res.latency_ms)
        cache_hits += int(res.cache_hit)
        if i % 25 == 0:
            print(f"    {i}/{len(rows)}", flush=True)

    out = summarise_text_task(preds, golds, lat, cache_hits)
    out["tool_failures"] = failures

    by_type: Dict[str, Dict[str, List[str]]] = collections.defaultdict(
        lambda: {"pred": [], "gold": []}
    )
    for p, g, t in zip(preds, golds, types):
        by_type[t]["pred"].append(p)
        by_type[t]["gold"].append(g)

    out["per_type"] = {
        t: {
            "n": len(v["pred"]),
            "exact_match": round(
                sum(
                    normalize_answer(a) == normalize_answer(b)
                    for a, b in zip(v["pred"], v["gold"])
                )
                / max(1, len(v["pred"])),
                4,
            ),
        }
        for t, v in sorted(by_type.items())
    }
    out["samples"] = [
        {"question": r["question"], "gold": g, "pred": p}
        for r, g, p in list(zip(rows, golds, preds))[:10]
    ]
    return out


def run_fusion(registry, rows: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    """Cross-modal optical+SAR VQA on the BigEarthNet-MM held-out split."""
    preds, golds, lat = [], [], []
    cache_hits = 0
    failures = 0

    for i, row in enumerate(rows, 1):
        res = registry.run_optical_sar(row["optical"], row["sar"], row["question"], params)
        if res.error:
            failures += 1
            continue
        preds.append(res.answer)
        golds.append(row["answer"])
        lat.append(res.latency_ms)
        cache_hits += int(res.cache_hit)
        if i % 10 == 0:
            print(f"    {i}/{len(rows)}", flush=True)

    out = summarise_text_task(preds, golds, lat, cache_hits)
    out["tool_failures"] = failures
    out["samples"] = [
        {"question": r["question"], "gold": g, "pred": p}
        for r, g, p in list(zip(rows, golds, preds))[:10]
    ]
    return out


def run_caption(registry, rows: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    preds, golds, lat = [], [], []
    cache_hits = 0
    failures = 0

    for i, row in enumerate(rows, 1):
        res = registry.run_caption(row["image"], params)
        if res.error:
            failures += 1
            continue
        preds.append(res.answer)
        golds.append(row["answer"])
        lat.append(res.latency_ms)
        cache_hits += int(res.cache_hit)
        if i % 25 == 0:
            print(f"    {i}/{len(rows)}", flush=True)

    out = summarise_text_task(preds, golds, lat, cache_hits)
    # Exact match is meaningless for free-form captions; token F1 carries it.
    out["note"] = "Captioning: token F1 is the meaningful metric; exact match is near-zero by nature."
    out["tool_failures"] = failures
    out["samples"] = [
        {"gold": g[:200], "pred": p[:200]} for g, p in list(zip(golds, preds))[:6]
    ]
    return out


def run_change(registry, rows: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    preds, golds, lat = [], [], []
    cache_hits = 0
    failures = 0

    for i, row in enumerate(rows, 1):
        res = registry.run_change_vqa(row["t1"], row["t2"], row["question"], params)
        if res.error:
            failures += 1
            continue
        preds.append(res.answer)
        golds.append(row["suffix"])
        lat.append(res.latency_ms)
        cache_hits += int(res.cache_hit)
        if i % 10 == 0:
            print(f"    {i}/{len(rows)}", flush=True)

    out = summarise_text_task(preds, golds, lat, cache_hits)
    out["tool_failures"] = failures
    out["samples"] = [
        {"question": r["question"], "gold": g, "pred": p}
        for r, g, p in list(zip(rows, golds, preds))[:10]
    ]
    return out


def run_grounding(
    registry, rows: List[Dict[str, Any]], params: Dict[str, Any]
) -> Dict[str, Any]:
    ious: List[float] = []
    lat: List[float] = []
    no_detection = 0
    failures = 0

    backends: collections.Counter = collections.Counter()

    for i, row in enumerate(rows, 1):
        res = registry.run_grounding(row["image"], [row["phrase"]], params)
        if res.error:
            failures += 1
            continue
        lat.append(res.latency_ms)
        backends[res.applied_parameters.get("backend", "unknown")] += 1
        detections = res.evidence.get("detections", [])
        if not detections:
            no_detection += 1
            ious.append(0.0)
            continue

        # Annotations may be normalised, percent or per-mille rather than
        # absolute pixels. Scoring relative coordinates against pixel-space
        # predictions silently yields near-zero IoU, which reads as a broken
        # model rather than a unit mismatch.
        gold = list(row["bbox"])
        divisor = row.get("bbox_divisor")
        if divisor:
            size = res.applied_parameters.get("image_size")
            if size and len(size) == 2:
                w, h = size
                gold = [
                    gold[0] / divisor * w,
                    gold[1] / divisor * h,
                    gold[2] / divisor * w,
                    gold[3] / divisor * h,
                ]
        # Best IoU over returned boxes: the annotation names one region, and
        # any returned box matching it counts as a hit.
        ious.append(max(iou(d["box"], gold) for d in detections))
        if i % 25 == 0:
            print(f"    {i}/{len(rows)}", flush=True)

    n = len(ious)
    if n == 0:
        return {"n": 0, "tool_failures": failures}

    return {
        "n": n,
        "mean_iou": round(sum(ious) / n, 4),
        "acc@0.5": round(sum(1 for v in ious if v >= 0.5) / n, 4),
        "acc@0.25": round(sum(1 for v in ious if v >= 0.25) / n, 4),
        "no_detection_rate": round(no_detection / n, 4),
        "mean_latency_ms": round(sum(lat) / len(lat), 1) if lat else 0.0,
        "tool_failures": failures,
        "backends_used": dict(backends),
        "note": (
            "Check backends_used. 'detector' means stock GroundingDINO ran - a "
            "natural-image model on nadir imagery, i.e. the pre-adaptation "
            "baseline. 'vlm' means the remote-sensing-adapted PaliGemma detect "
            "adapter ran, which is the configuration the problem statement asks for."
        ),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def to_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# SatQuery AI - benchmark results",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Device: `{report['environment']['device']}` / `{report['environment']['dtype']}`",
        f"- Base VLM: `{report['environment']['base_vlm']}`",
        f"- Cache: {'enabled' if report['environment']['cache_enabled'] else 'DISABLED'}",
        "",
    ]

    for task, m in report["results"].items():
        lines.append(f"## {task}")
        lines.append("")
        if m.get("n", 0) == 0:
            lines.append("_No data. Build the split first._")
            lines.append("")
            continue

        lines.append("| metric | value |")
        lines.append("|---|---|")
        for k, v in m.items():
            if k in ("samples", "note"):
                continue
            lines.append(f"| {k} | {v} |")
        lines.append("")

        if m.get("collapsed"):
            lines.append(
                "> **COLLAPSED.** Predictions are near-constant across the split. "
                "This model has learned the answer prior, not the imagery. Do not ship it."
            )
            lines.append("")
        elif "beats_majority_baseline" in m and not m["beats_majority_baseline"]:
            lines.append(
                "> **Below the majority-class baseline.** Always answering the most "
                "common gold label would score at least as well as this model."
            )
            lines.append("")

        if m.get("note"):
            lines.append(f"_{m['note']}_")
            lines.append("")

        if m.get("samples"):
            lines.append("<details><summary>Sample predictions</summary>")
            lines.append("")
            for s in m["samples"]:
                if "question" in s:
                    lines.append(f"- **Q:** {s['question']}")
                lines.append(f"  - gold: `{s['gold']}`")
                lines.append(f"  - pred: `{s['pred']}`")
            lines.append("")
            lines.append("</details>")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Score SatQuery AI on held-out splits.")
    ap.add_argument(
        "--task",
        choices=["vqa", "rsvqa", "caption", "change", "fusion", "grounding", "all"],
        default="all",
    )
    ap.add_argument("--vrsbench-dir", default="data/vrsbench")
    ap.add_argument("--cdvqa-dir", default="training/cdvqa")
    ap.add_argument("--rsvqa-dir", default="data/rsvqa")
    ap.add_argument("--bigearthnet-dir", default="data/bigearthnet")
    ap.add_argument("--out-dir", default="eval/results")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--box-threshold", type=float, default=0.25)
    ap.add_argument("--text-threshold", type=float, default=0.20)
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="Force live inference. Use this for any number you intend to report - "
             "a cached run measures the cache, not the model.",
    )
    args = ap.parse_args()

    if args.no_cache:
        os.environ["SATQUERY_DISABLE_CACHE"] = "1"

    from modules.model_registry import BASE_VLM_ID, ModelRegistry

    registry = ModelRegistry()

    vrs = Path(args.vrsbench_dir)
    cdv = Path(args.cdvqa_dir)
    tasks = (
        ("vqa", "rsvqa", "caption", "change", "fusion", "grounding")
        if args.task == "all"
        else (args.task,)
    )

    gen_params = {"max_new_tokens": args.max_new_tokens, "num_beams": args.num_beams}
    ground_params = {
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "max_box_area_fraction": 0.92,
        "top_k": 8,
    }

    results: Dict[str, Any] = {}
    started = time.time()

    for task in tasks:
        print(f"\n=== {task} ===")
        if task == "vqa":
            rows = read_jsonl(vrs / "val_vqa.jsonl", args.limit)
            print(f"  {len(rows)} rows from {vrs / 'val_vqa.jsonl'}")
            results["single_image_vqa"] = run_vqa(registry, rows, gen_params) if rows else {"n": 0}

        elif task == "rsvqa":
            rsv = Path(args.rsvqa_dir)
            rows = read_jsonl(rsv / "test.jsonl", args.limit) or read_jsonl(
                rsv / "val.jsonl", args.limit
            )
            print(f"  {len(rows)} rows from {rsv}")
            results["rsvqa"] = run_rsvqa(registry, rows, gen_params) if rows else {"n": 0}

        elif task == "fusion":
            ben = Path(args.bigearthnet_dir)
            rows = read_jsonl(ben / "val.jsonl", args.limit)
            print(f"  {len(rows)} rows from {ben / 'val.jsonl'}")
            results["optical_sar_fusion"] = (
                run_fusion(registry, rows, gen_params) if rows else {"n": 0}
            )

        elif task == "caption":
            rows = read_jsonl(vrs / "val_caption.jsonl", args.limit)
            print(f"  {len(rows)} rows from {vrs / 'val_caption.jsonl'}")
            results["single_image_captioning"] = (
                run_caption(registry, rows, {"max_new_tokens": 96, "num_beams": 3})
                if rows
                else {"n": 0}
            )

        elif task == "change":
            rows = read_jsonl(cdv / "val.jsonl", args.limit)
            print(f"  {len(rows)} rows from {cdv / 'val.jsonl'}")
            results["bitemporal_change_vqa"] = (
                run_change(registry, rows, gen_params) if rows else {"n": 0}
            )

        elif task == "grounding":
            rows = read_jsonl(vrs / "grounding_eval.jsonl", args.limit)
            print(f"  {len(rows)} rows from {vrs / 'grounding_eval.jsonl'}")
            results["region_grounding"] = (
                run_grounding(registry, rows, ground_params) if rows else {"n": 0}
            )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "device": registry.device,
            "dtype": str(registry.dtype),
            "base_vlm": BASE_VLM_ID,
            "loaded_adapters": registry.loaded_adapters,
            "cache_enabled": registry.cache_enabled,
        },
        "configuration": {
            "limit": args.limit,
            "generation": gen_params,
            "grounding": ground_params,
        },
        "results": results,
        "total_seconds": round(time.time() - started, 1),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = out_dir / f"benchmark_{stamp}.json"
    md_path = out_dir / f"benchmark_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")

    print("\n" + "=" * 70)
    for task, m in results.items():
        if m.get("n", 0) == 0:
            print(f"{task:<28} no data")
            continue
        if "mean_iou" in m:
            print(f"{task:<28} IoU@0.5={m['acc@0.5']:.3f}  meanIoU={m['mean_iou']:.3f}")
        else:
            flag = "  <-- COLLAPSED" if m.get("collapsed") else ""
            print(
                f"{task:<28} EM={m['exact_match']:.3f}  F1={m['token_f1']:.3f}  "
                f"distinct={m['distinct_ratio']:.3f}{flag}"
            )
    print("=" * 70)
    print(f"\nWrote {json_path}\nWrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
