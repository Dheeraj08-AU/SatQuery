"""
VRSBench dataset builder for SatQuery AI.

Replaces the previous version, which used `item.get('question') or
item.get('caption') or '<hardcoded string>'` fallback chains. When the real
schema did not match, those chains silently manufactured 1000 identical
synthetic records instead of failing. This version:

  * discovers the real files in the HF repo instead of hardcoding names,
  * auto-detects the record schema and reports which one matched,
  * NEVER substitutes a placeholder for missing data - unparseable records
    are counted and reported, and the build aborts if too many fail,
  * verifies every referenced image actually exists on disk,
  * enforces hard diversity floors (unique images, unique answers, max
    frequency of the single most common answer),
  * splits train/val BY IMAGE so no image appears in both splits.

Usage
-----
    python data_prep.py --inspect          # dump real schemas, build nothing
    python data_prep.py                    # build the full dataset
    python data_prep.py --max-records 8000 # cap for a faster first run
    python data_prep.py --no-images        # skip the images zip (schema only)

Outputs (under --out-dir, default `data/vrsbench`):
    train_vqa.jsonl      {"image", "question", "answer", "task", "image_id"}
    val_vqa.jsonl
    train_caption.jsonl
    val_caption.jsonl
    grounding_eval.jsonl {"image", "phrase", "bbox": [x0,y0,x1,y1]}
    build_report.json    full provenance + statistics
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ID = "xiang709/VRSBench"
REPO_TYPE = "dataset"

# Strings the old broken pipeline injected. If any of these show up in a
# built record, the parse is wrong and we abort.
FORBIDDEN_ANSWERS = {
    "urban and natural land-cover.",
    "urban and natural land-cover",
}
FORBIDDEN_QUESTIONS = {
    "describe the visible features in this image.",
    "describe the visible features.",
}

# Quality floors. A build that cannot clear these is not usable for training.
MIN_RECORDS = 2000
MIN_UNIQUE_IMAGES = 300
MIN_UNIQUE_ANSWERS = 100
MAX_TOP_ANSWER_FRACTION = 0.35   # no single answer may exceed this share
MAX_PARSE_FAILURE_FRACTION = 0.05

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

CAPTION_CUES = (
    "describe",
    "caption",
    "provide a description",
    "give a description",
    "summarize the image",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatasetBuildError(RuntimeError):
    """Raised when the built dataset fails a quality gate."""


def _fail(msg: str) -> None:
    raise DatasetBuildError(msg)


# ---------------------------------------------------------------------------
# HF repo discovery
# ---------------------------------------------------------------------------


def list_repo_files() -> List[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    return sorted(files)


def pick(files: Iterable[str], *patterns: str) -> Optional[str]:
    """First file whose lowercased name matches every pattern (substring)."""
    for f in files:
        low = f.lower()
        if all(p.lower() in low for p in patterns):
            return f
    return None


def download(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    print(f"  downloading {filename} ...", flush=True)
    return hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type=REPO_TYPE)


# ---------------------------------------------------------------------------
# Schema handling
# ---------------------------------------------------------------------------


def load_json_records(path: str) -> List[Any]:
    """Load a JSON or JSONL file into a list of records."""
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
    elif stripped.startswith("{"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            # JSON Lines
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        if isinstance(obj, dict):
            # A dict wrapper: find the first list-valued key.
            for key in ("data", "annotations", "records", "questions", "items"):
                if isinstance(obj.get(key), list):
                    print(f"  (unwrapped list from key '{key}')")
                    return obj[key]
            list_keys = [k for k, v in obj.items() if isinstance(v, list)]
            if len(list_keys) == 1:
                print(f"  (unwrapped list from key '{list_keys[0]}')")
                return obj[list_keys[0]]
            _fail(
                f"{path} is a JSON object, not a list, and no obvious record list was "
                f"found. Top-level keys: {list(obj.keys())[:20]}"
            )
        data = obj
    else:
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    if not isinstance(data, list):
        _fail(f"{path} did not parse to a list (got {type(data).__name__}).")
    return data


def describe_schema(records: List[Any], name: str, n: int = 3) -> Dict[str, Any]:
    """Human- and machine-readable description of a record list."""
    info: Dict[str, Any] = {"file": name, "count": len(records)}
    if not records:
        info["error"] = "empty"
        return info

    types = collections.Counter(type(r).__name__ for r in records[:500])
    info["record_types"] = dict(types)

    if isinstance(records[0], dict):
        keys = collections.Counter()
        for r in records[:2000]:
            if isinstance(r, dict):
                keys.update(r.keys())
        info["keys"] = dict(keys.most_common())

    info["samples"] = []
    for r in records[:n]:
        s = json.dumps(r, ensure_ascii=False)
        info["samples"].append(s[:800] + (" ...[truncated]" if len(s) > 800 else ""))
    return info


_IMAGE_TAG = re.compile(r"<image>\s*", flags=re.IGNORECASE)


def _clean_prompt(text: str) -> str:
    return _IMAGE_TAG.sub("", str(text)).strip()


def _first_str(rec: Dict[str, Any], *keys: str) -> Optional[str]:
    """Return the first key present with a non-empty string value. No fallbacks."""
    for k in keys:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return str(v)
    return None


def normalize_record(rec: Any) -> List[Dict[str, str]]:
    """
    Turn one raw record into zero or more {image_id, question, answer} dicts.

    Returns [] when the record cannot be understood. It NEVER invents content.
    Supported shapes:
      A. LLaVA conversations: {"image", "conversations":[{from,value}, ...]}
      B. flat QA:             {"image"/"image_id", "question", "answer"}
      C. caption only:        {"image"/"image_id", "caption"}
    """
    if not isinstance(rec, dict):
        return []

    image_id = _first_str(
        rec, "image", "image_id", "img", "image_path", "file_name", "filename", "imgname"
    )
    if not image_id:
        return []
    image_id = os.path.basename(image_id.replace("\\", "/"))

    out: List[Dict[str, str]] = []

    # --- Shape A: LLaVA conversations ---
    convs = rec.get("conversations")
    if isinstance(convs, list) and convs:
        pending_q: Optional[str] = None
        for turn in convs:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", "")).lower()
            value = turn.get("value")
            if not isinstance(value, str) or not value.strip():
                continue
            if role in ("human", "user"):
                pending_q = _clean_prompt(value)
            elif role in ("gpt", "assistant", "model"):
                if pending_q:
                    out.append(
                        {
                            "image_id": image_id,
                            "question": pending_q,
                            "answer": value.strip(),
                        }
                    )
                    pending_q = None
        return out

    # --- Shape B: flat QA ---
    question = _first_str(rec, "question", "query", "instruction", "prompt", "text")
    answer = _first_str(rec, "answer", "ground_truth", "gt", "label", "response")
    if question and answer:
        out.append(
            {
                "image_id": image_id,
                "question": _clean_prompt(question),
                "answer": answer,
            }
        )
        return out

    # --- Shape C: caption only ---
    caption = _first_str(rec, "caption", "description", "captions")
    if caption:
        out.append(
            {
                "image_id": image_id,
                "question": "Describe the land cover and major objects visible in this image.",
                "answer": caption,
            }
        )
        return out

    return []


_BOX_IN_TEXT = re.compile(r"\{?\s*(?:<\s*-?\d+(?:\.\d+)?\s*>\s*){4}\}?")


def _numbers(text: str) -> List[float]:
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text)]


def normalize_grounding(rec: Any) -> List[Dict[str, Any]]:
    """
    Extract referring-expression boxes: {image_id, phrase, bbox [x0,y0,x1,y1]}.

    Handles the three ways these annotations appear: an explicit `bbox` list, a
    box embedded in an answer string, and VRSBench's LLaVA-style conversations
    where the human turn holds the referring phrase and the assistant turn holds
    `{<x0><y0><x1><y1>}`.
    """
    if not isinstance(rec, dict):
        return []
    image_id = _first_str(rec, "image", "image_id", "img", "file_name", "filename")
    if not image_id:
        return []
    image_id = os.path.basename(image_id.replace("\\", "/"))

    out: List[Dict[str, Any]] = []

    # --- conversations carrying boxes ---
    convs = rec.get("conversations")
    if isinstance(convs, list) and convs:
        pending: Optional[str] = None
        for turn in convs:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", "")).lower()
            value = turn.get("value")
            if not isinstance(value, str) or not value.strip():
                continue
            if role in ("human", "user"):
                pending = _clean_prompt(value)
            elif role in ("gpt", "assistant", "model") and pending:
                if _BOX_IN_TEXT.search(value) or "<" in value:
                    nums = _numbers(value)
                    if len(nums) >= 4:
                        out.append(
                            {
                                "image_id": image_id,
                                "phrase": pending,
                                "bbox": nums[:4],
                            }
                        )
                pending = None
        if out:
            return out

    # --- flat record ---
    phrase = _first_str(
        rec, "question", "referring_sentence", "sentence", "phrase", "expression", "caption"
    )
    bbox = None
    for key in ("bbox", "box", "boxes", "answer", "gt_box"):
        v = rec.get(key)
        if isinstance(v, (list, tuple)) and len(v) == 4 and all(
            isinstance(c, (int, float)) for c in v
        ):
            bbox = [float(c) for c in v]
            break
        if isinstance(v, str):
            nums = _numbers(v)
            if len(nums) >= 4:
                bbox = nums[:4]
                break
    if not phrase or bbox is None:
        return []
    return [{"image_id": image_id, "phrase": _clean_prompt(phrase), "bbox": bbox}]


def detect_bbox_convention(boxes: List[List[float]]) -> Dict[str, Any]:
    """
    Work out what coordinate system the boxes are in.

    VRSBench-style annotations appear variously as absolute pixels, per-mille
    (0-999), percent (0-100) or normalised (0-1). Guessing wrong silently
    trains the model on boxes that do not sit on the objects, which is
    invisible until the IoU comes back near zero, so the observed range is
    recorded in the build report.
    """
    if not boxes:
        return {"convention": "unknown", "max_value": None}
    flat = [c for b in boxes for c in b]
    hi = max(flat)
    lo = min(flat)
    if hi <= 1.5:
        convention, scale = "normalised_0_1", 1.0
    elif hi <= 100.5:
        convention, scale = "percent_0_100", 100.0
    elif hi <= 1000.5:
        convention, scale = "per_mille_0_999", 999.0
    else:
        convention, scale = "absolute_pixels", None
    return {
        "convention": convention,
        "normalising_divisor": scale,
        "min_value": round(lo, 3),
        "max_value": round(hi, 3),
        "note": (
            "Relative conventions are divided by the divisor and multiplied by the "
            "image's own width/height at training time. 'absolute_pixels' is used "
            "as-is."
        ),
    }


def is_caption_task(question: str, answer: str) -> bool:
    q = question.lower()
    if any(cue in q for cue in CAPTION_CUES):
        return True
    return len(answer.split()) >= 20


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def ensure_images(files: List[str], out_dir: Path, skip: bool) -> Tuple[Optional[Path], List[str]]:
    """Download+extract the training images zip. Returns (image_root, notes)."""
    notes: List[str] = []
    if skip:
        notes.append("image download skipped (--no-images)")
        return None, notes

    zip_name = (
        pick(files, "image", "train", ".zip")
        or pick(files, "images", ".zip")
        or pick(files, "image", ".zip")
    )
    if not zip_name:
        zips = [f for f in files if f.lower().endswith(".zip")]
        _fail(
            "Could not find an images zip in the repo. "
            f"Available .zip files: {zips or '(none)'}"
        )

    zip_path = download(zip_name)
    image_root = out_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    marker = image_root / ".extracted"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == zip_name:
        notes.append(f"images already extracted from {zip_name}")
        return image_root, notes

    print(f"  extracting {zip_name} -> {image_root} ...", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(image_root)
    marker.write_text(zip_name, encoding="utf-8")
    notes.append(f"extracted {zip_name}")
    return image_root, notes


def index_images(image_root: Optional[Path]) -> Dict[str, str]:
    """Map basename -> absolute path for every image under image_root."""
    if image_root is None:
        return {}
    index: Dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(image_root):
        for fn in filenames:
            if fn.lower().endswith(IMAGE_EXTS):
                index.setdefault(fn, os.path.join(dirpath, fn))
    return index


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------


def assert_quality(records: List[Dict[str, str]], label: str) -> Dict[str, Any]:
    """Hard gates. Raises DatasetBuildError on failure. Returns stats."""
    n = len(records)
    images = {r["image_id"] for r in records}
    answers = collections.Counter(r["answer"].strip().lower() for r in records)
    questions = collections.Counter(r["question"].strip().lower() for r in records)

    stats = {
        "records": n,
        "unique_images": len(images),
        "unique_answers": len(answers),
        "unique_questions": len(questions),
        "top_answers": answers.most_common(10),
        "top_questions": questions.most_common(5),
    }

    problems: List[str] = []

    if n < MIN_RECORDS:
        problems.append(f"only {n} records (need >= {MIN_RECORDS})")

    if len(images) < MIN_UNIQUE_IMAGES:
        problems.append(
            f"only {len(images)} unique images (need >= {MIN_UNIQUE_IMAGES}). "
            "Training on few images is what produced the degenerate adapter."
        )

    if len(answers) < MIN_UNIQUE_ANSWERS:
        problems.append(f"only {len(answers)} unique answers (need >= {MIN_UNIQUE_ANSWERS})")

    if n:
        top_ans, top_cnt = answers.most_common(1)[0]
        frac = top_cnt / n
        stats["top_answer_fraction"] = round(frac, 4)
        if frac > MAX_TOP_ANSWER_FRACTION:
            problems.append(
                f"answer {top_ans!r} is {frac:.1%} of the dataset "
                f"(max {MAX_TOP_ANSWER_FRACTION:.0%}); the model will learn the prior, not the image"
            )

    hit_forbidden_a = answers.keys() & FORBIDDEN_ANSWERS
    if hit_forbidden_a:
        problems.append(
            f"placeholder answer(s) present: {sorted(hit_forbidden_a)}. "
            "This means the source parse failed and fallbacks leaked in."
        )
    hit_forbidden_q = questions.keys() & FORBIDDEN_QUESTIONS
    if hit_forbidden_q:
        problems.append(f"placeholder question(s) present: {sorted(hit_forbidden_q)}")

    if problems:
        detail = "\n".join(f"  - {p}" for p in problems)
        _fail(
            f"Quality gate FAILED for '{label}':\n{detail}\n\n"
            f"Statistics: {json.dumps(stats, indent=2, default=str)}"
        )

    return stats


def balance_answers(
    records: List[Dict[str, str]], max_fraction: float, seed: int
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    Subsample over-represented answer classes so no single answer exceeds
    `max_fraction` of the set.

    VRSBench VQA is heavy on yes/no. Left unbalanced, the cheapest thing a
    model can learn is the marginal answer distribution - answer "yes" and be
    right a third of the time without looking at the image. Balancing removes
    that shortcut, and it also stops the quality gate below from failing on a
    dataset that is legitimate but skewed.
    """
    by_answer: Dict[str, List[Dict[str, str]]] = collections.defaultdict(list)
    for r in records:
        by_answer[r["answer"].strip().lower()].append(r)

    rng = random.Random(seed)
    for rows in by_answer.values():
        rng.shuffle(rows)

    # The cap depends on the total, and the total shrinks as we cap, so iterate.
    counts = {a: len(rows) for a, rows in by_answer.items()}
    for _ in range(100):
        total = sum(counts.values())
        cap = max(1, int(total * max_fraction))
        changed = False
        for a in counts:
            if counts[a] > cap:
                counts[a] = cap
                changed = True
        if not changed:
            break

    out: List[Dict[str, str]] = []
    for a, rows in by_answer.items():
        out.extend(rows[: counts[a]])
    rng.shuffle(out)
    return out, counts


def split_by_image(
    records: List[Dict[str, str]], val_fraction: float, seed: int
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Split so that no image appears in both halves. Splitting by QA pair (the
    previous behaviour) leaks: VRSBench/CDVQA carry many questions per image,
    so a random pair-level split puts the same picture on both sides.
    """
    images = sorted({r["image_id"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(images)
    n_val = max(1, int(len(images) * val_fraction))
    val_images = set(images[:n_val])
    train = [r for r in records if r["image_id"] not in val_images]
    val = [r for r in records if r["image_id"] in val_images]
    return train, val


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
    ap = argparse.ArgumentParser(description="Build the VRSBench training set.")
    ap.add_argument("--out-dir", default="data/vrsbench")
    ap.add_argument("--inspect", action="store_true", help="Dump real schemas and exit.")
    ap.add_argument("--no-images", action="store_true", help="Skip downloading the images zip.")
    ap.add_argument("--max-records", type=int, default=0, help="0 = no cap.")
    ap.add_argument("--max-answer-fraction", type=float, default=0.30,
                    help="Cap on the share of any single VQA answer.")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Listing files in {REPO_ID} ...")
    files = list_repo_files()
    for f in files:
        print(f"  {f}")

    train_json = pick(files, "train", ".json") or pick(files, "vrsbench", ".json")
    if not train_json:
        _fail(f"No training JSON found. Repo files: {files}")

    eval_vqa_json = pick(files, "eval", "vqa", ".json")
    eval_cap_json = pick(files, "eval", "cap", ".json")
    eval_ref_json = pick(files, "referring", ".json")

    print(f"\nSelected files:")
    print(f"  train      : {train_json}")
    print(f"  eval vqa   : {eval_vqa_json}")
    print(f"  eval cap   : {eval_cap_json}")
    print(f"  eval refer : {eval_ref_json}")

    # ---- inspect mode -----------------------------------------------------
    if args.inspect:
        report = {"repo_files": files, "schemas": []}
        for name in filter(None, [train_json, eval_vqa_json, eval_cap_json, eval_ref_json]):
            path = download(name)
            recs = load_json_records(path)
            info = describe_schema(recs, name)
            report["schemas"].append(info)
            print(f"\n{'=' * 70}\n{name}  ({info['count']:,} records)")
            print(f"keys: {info.get('keys')}")
            for s in info["samples"]:
                print(f"  {s}")
        (out_dir / "schema_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nWrote {out_dir / 'schema_report.json'}")
        return 0

    # ---- images -----------------------------------------------------------
    print("\nPreparing images ...")
    image_root, image_notes = ensure_images(files, out_dir, args.no_images)
    image_index = index_images(image_root)
    print(f"  indexed {len(image_index):,} image files")

    # ---- parse training records ------------------------------------------
    print(f"\nParsing {train_json} ...")
    raw = load_json_records(download(train_json))
    print(f"  {len(raw):,} raw records")

    schema_info = describe_schema(raw, train_json)
    print(f"  keys observed: {schema_info.get('keys')}")

    normalized: List[Dict[str, str]] = []
    failed = 0
    for rec in raw:
        got = normalize_record(rec)
        if not got:
            failed += 1
            continue
        normalized.extend(got)

    fail_frac = failed / len(raw) if raw else 1.0
    print(f"  normalized {len(normalized):,} QA pairs from {len(raw) - failed:,} records")
    print(f"  unparseable records: {failed:,} ({fail_frac:.1%})")

    if fail_frac > MAX_PARSE_FAILURE_FRACTION:
        _fail(
            f"{fail_frac:.1%} of records could not be parsed (max "
            f"{MAX_PARSE_FAILURE_FRACTION:.0%}). The schema is not what this script "
            f"expects. Run `python data_prep.py --inspect` and inspect "
            f"{out_dir / 'schema_report.json'}.\n"
            f"Observed keys: {schema_info.get('keys')}\n"
            f"Sample records:\n" + "\n".join(schema_info["samples"])
        )

    if not normalized:
        _fail("Zero records survived normalization.")

    # ---- resolve image paths ---------------------------------------------
    resolved: List[Dict[str, str]] = []
    missing_images: collections.Counter = collections.Counter()

    for r in normalized:
        if image_index:
            path = image_index.get(r["image_id"])
            if path is None:
                stem = Path(r["image_id"]).stem
                path = next(
                    (image_index[k] for k in image_index if Path(k).stem == stem),
                    None,
                )
            if path is None:
                missing_images[r["image_id"]] += 1
                continue
        else:
            path = r["image_id"]
        resolved.append({**r, "image": path})

    if image_index:
        miss_frac = 1.0 - (len(resolved) / len(normalized))
        print(f"  resolved {len(resolved):,} records to real image files ({miss_frac:.1%} dropped)")
        if miss_frac > 0.10:
            _fail(
                f"{miss_frac:.1%} of records reference images that are not on disk. "
                f"Examples: {missing_images.most_common(5)}"
            )
    else:
        print("  WARNING: no image index (--no-images); paths are unverified basenames")

    if args.max_records and len(resolved) > args.max_records:
        rng = random.Random(args.seed)
        rng.shuffle(resolved)
        resolved = resolved[: args.max_records]
        print(f"  capped to {len(resolved):,} records (--max-records)")

    # ---- split by task ----------------------------------------------------
    vqa = [r for r in resolved if not is_caption_task(r["question"], r["answer"])]
    caption = [r for r in resolved if is_caption_task(r["question"], r["answer"])]
    print(f"\n  VQA records     : {len(vqa):,}")
    print(f"  Caption records : {len(caption):,}")

    for r in vqa:
        r["task"] = "vqa"
    for r in caption:
        r["task"] = "caption"

    # ---- balance the VQA answer distribution ------------------------------
    print(f"\nBalancing VQA answers to <= {args.max_answer_fraction:.0%} each ...")
    top_before = collections.Counter(r["answer"].strip().lower() for r in vqa).most_common(5)
    vqa, answer_caps = balance_answers(vqa, args.max_answer_fraction, args.seed)
    top_after = collections.Counter(r["answer"].strip().lower() for r in vqa).most_common(5)
    print(f"  {len(vqa):,} records after balancing")
    print(f"  top answers before: {top_before}")
    print(f"  top answers after : {top_after}")

    # ---- quality gates ----------------------------------------------------
    print("\nRunning quality gates ...")
    vqa_stats = assert_quality(vqa, "VQA")
    print(f"  VQA OK: {vqa_stats['records']:,} records, "
          f"{vqa_stats['unique_images']:,} images, "
          f"{vqa_stats['unique_answers']:,} distinct answers, "
          f"top answer {vqa_stats.get('top_answer_fraction', 0):.1%}")

    caption_stats: Dict[str, Any] = {}
    if len(caption) >= MIN_RECORDS:
        caption_stats = assert_quality(caption, "Caption")
        print(f"  Caption OK: {caption_stats['records']:,} records, "
              f"{caption_stats['unique_images']:,} images")
    else:
        print(f"  Caption set too small to gate ({len(caption):,}); writing as-is.")

    # ---- split & write ----------------------------------------------------
    print("\nSplitting by image and writing ...")
    tr_vqa, va_vqa = split_by_image(vqa, args.val_fraction, args.seed)
    write_jsonl(out_dir / "train_vqa.jsonl", tr_vqa)
    write_jsonl(out_dir / "val_vqa.jsonl", va_vqa)

    overlap = {r["image_id"] for r in tr_vqa} & {r["image_id"] for r in va_vqa}
    if overlap:
        _fail(f"train/val image leakage: {len(overlap)} shared images")

    tr_cap: List[Dict[str, str]] = []
    va_cap: List[Dict[str, str]] = []
    if caption:
        tr_cap, va_cap = split_by_image(caption, args.val_fraction, args.seed)
        write_jsonl(out_dir / "train_caption.jsonl", tr_cap)
        write_jsonl(out_dir / "val_caption.jsonl", va_cap)

    # ---- grounding: train split from the training file --------------------
    def resolve_grounding(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not image_index:
            return [{**g, "image": g["image_id"]} for g in items]
        kept = []
        for g in items:
            p = image_index.get(g["image_id"])
            if p is None:
                stem = Path(g["image_id"]).stem
                p = next((image_index[k] for k in image_index if Path(k).stem == stem), None)
            if p:
                kept.append({**g, "image": p})
        return kept

    print("\nParsing grounding annotations from the training file ...")
    grounding_train_all: List[Dict[str, Any]] = []
    for rec in raw:
        grounding_train_all.extend(normalize_grounding(rec))
    grounding_train_all = resolve_grounding(grounding_train_all)
    print(f"  {len(grounding_train_all):,} referring expressions with boxes")

    train_convention = detect_bbox_convention([g["bbox"] for g in grounding_train_all])
    if grounding_train_all:
        print(f"  coordinate convention: {train_convention}")

    tr_ground: List[Dict[str, Any]] = []
    va_ground: List[Dict[str, Any]] = []
    if grounding_train_all:
        images_g = sorted({g["image_id"] for g in grounding_train_all})
        rng_g = random.Random(args.seed)
        rng_g.shuffle(images_g)
        n_val_g = max(1, int(len(images_g) * args.val_fraction))
        val_g = set(images_g[:n_val_g])
        tr_ground = [g for g in grounding_train_all if g["image_id"] not in val_g]
        va_ground = [g for g in grounding_train_all if g["image_id"] in val_g]
        for row in tr_ground + va_ground:
            row["bbox_convention"] = train_convention["convention"]
            row["bbox_divisor"] = train_convention["normalising_divisor"]
        write_jsonl(out_dir / "grounding_train.jsonl", tr_ground)
        write_jsonl(out_dir / "grounding_val.jsonl", va_ground)
    else:
        print(
            "  WARNING: no grounding boxes found in the training file. The "
            "grounding adapter cannot be trained without them - inspect the "
            "schema report and tell me the real annotation format."
        )

    # ---- grounding eval (held out, never trained on) ----------------------
    grounding: List[Dict[str, Any]] = []
    eval_convention: Dict[str, Any] = {}
    if eval_ref_json:
        print(f"\nParsing grounding benchmark from {eval_ref_json} ...")
        raw_ref = load_json_records(download(eval_ref_json))
        for rec in raw_ref:
            grounding.extend(normalize_grounding(rec))
        grounding = resolve_grounding(grounding)
        eval_convention = detect_bbox_convention([g["bbox"] for g in grounding])
        for row in grounding:
            row["bbox_convention"] = eval_convention.get("convention")
            row["bbox_divisor"] = eval_convention.get("normalising_divisor")
        print(f"  {len(grounding):,} referring expressions with boxes")
        if grounding:
            print(f"  coordinate convention: {eval_convention}")
            write_jsonl(out_dir / "grounding_eval.jsonl", grounding)
        else:
            print("  WARNING: no grounding boxes parsed; check the referring schema")

        leak = {g["image_id"] for g in grounding} & {g["image_id"] for g in tr_ground}
        if leak:
            print(
                f"  WARNING: {len(leak)} images appear in BOTH the grounding train "
                f"split and the benchmark. Reported IoU would be inflated."
            )

    # ---- report -----------------------------------------------------------
    report = {
        "repo": REPO_ID,
        "source_files": {
            "train": train_json,
            "eval_vqa": eval_vqa_json,
            "eval_caption": eval_cap_json,
            "eval_referring": eval_ref_json,
        },
        "image_notes": image_notes,
        "raw_records": len(raw),
        "unparseable_records": failed,
        "normalized_qa_pairs": len(normalized),
        "resolved_records": len(resolved),
        "observed_keys": schema_info.get("keys"),
        "vqa": {
            "stats": vqa_stats,
            "train": len(tr_vqa),
            "val": len(va_vqa),
            "answer_balancing": {
                "max_fraction": args.max_answer_fraction,
                "caps": answer_caps,
            },
        },
        "caption": {
            "stats": caption_stats,
            "train": len(tr_cap),
            "val": len(va_cap),
        },
        "grounding": {
            "train": len(tr_ground),
            "val": len(va_ground),
            "benchmark": len(grounding),
            "train_bbox_convention": train_convention,
            "benchmark_bbox_convention": eval_convention,
        },
        "split": {"by": "image_id", "val_fraction": args.val_fraction, "seed": args.seed},
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
