"""
CDVQA (bi-temporal change VQA) dataset builder for SatQuery AI.

The previous version produced 190 training samples covering only SIX unique
image pairs, with 65% of labels being bare "yes"/"no". A LoRA trained on that
learns the marginal answer prior and stops looking at the pixels entirely -
which is exactly the "degenerate fixed yes/no pattern regardless of content"
that was observed and then written off as an unexplained failure.

Four root causes, all fixed here:

  1. `except Exception: pass` inside the extraction loop silently discarded
     almost every sample. Failures are now counted, categorised and reported,
     and the build aborts if too many fail.
  2. `curl` via subprocess (unavailable/awkward on Windows, and a partial
     download passed the >100KB size check). Now uses `hf_hub_download`,
     which is atomic and checksum-verified.
  3. Train/val were split by QA pair. CDVQA carries ~40 questions per image
     pair, so a pair-level shuffle put the same imagery on both sides of the
     split. Now split BY IMAGE PAIR.
  4. No diversity floor and no class balancing. Now both, enforced.

It also stores T1 and T2 separately rather than pre-baking composites, so the
composite strategy can be changed without re-extracting the dataset.

Usage
-----
    python training/prep_cdvqa.py --inspect            # dump tar structure
    python training/prep_cdvqa.py --shards 24          # build
    python training/prep_cdvqa.py --shards 24 --max-qa-per-pair 12

Outputs (under --out-dir, default `training/cdvqa`):
    images/<pair_id>_t1.png, <pair_id>_t2.png
    train.jsonl   {"pair_id", "t1", "t2", "question", "answer"}
    val.jsonl
    build_report.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import random
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# Import the framing prefix from the single place that defines it, rather than
# re-typing it here. Training, inference and dataset construction must all use
# the byte-identical string; a copy in this file is a copy that can drift.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.composite import CHANGE_PREFIX as PROMPT_PREFIX  # noqa: E402

REPO_ID = "ljx620/CDVQA"
REPO_TYPE = "dataset"
SHARD_TEMPLATE = "train/train-{:05d}.tar"

# Boilerplate the CDVQA webdataset puts in front of every question.
BOILERPLATE = "Image 1: <image>\nImage 2: <image>\n"

# Quality floors.
MIN_RECORDS = 3000
MIN_UNIQUE_PAIRS = 300
MIN_UNIQUE_ANSWERS = 8
MAX_TOP_ANSWER_FRACTION = 0.35
MAX_EXTRACT_FAILURE_FRACTION = 0.10


class DatasetBuildError(RuntimeError):
    pass


def _fail(msg: str) -> None:
    raise DatasetBuildError(msg)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def fetch_shard(index: int) -> Optional[str]:
    """Download one shard. Returns None if the shard does not exist."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

    name = SHARD_TEMPLATE.format(index)
    try:
        return hf_hub_download(repo_id=REPO_ID, filename=name, repo_type=REPO_TYPE)
    except EntryNotFoundError:
        return None
    except HfHubHTTPError as exc:
        if "404" in str(exc):
            return None
        raise


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def group_members(tar: tarfile.TarFile) -> Dict[str, Dict[str, tarfile.TarInfo]]:
    """
    Group webdataset members by sample key.

    A member named `cdvqa-train-00000042.0.img` has key `cdvqa-train-00000042`
    and suffix `0.img`.
    """
    groups: Dict[str, Dict[str, tarfile.TarInfo]] = collections.defaultdict(dict)
    for m in tar.getmembers():
        if not m.isfile():
            continue
        name = os.path.basename(m.name)
        if "." not in name:
            continue
        key, suffix = name.split(".", 1)
        groups[key][suffix] = m
    return groups


def describe_tar(tar_path: str, n: int = 3) -> Dict[str, Any]:
    """Report the real structure of a shard without assuming anything."""
    with tarfile.open(tar_path, "r") as tar:
        groups = group_members(tar)
        suffixes = collections.Counter()
        for g in groups.values():
            suffixes.update(g.keys())

        samples = []
        for key in list(groups)[:n]:
            entry: Dict[str, Any] = {"key": key, "suffixes": sorted(groups[key].keys())}
            json_suffix = next(
                (s for s in groups[key] if s.endswith("json")), None
            )
            if json_suffix:
                try:
                    raw = tar.extractfile(groups[key][json_suffix]).read()
                    text = json.dumps(json.loads(raw), ensure_ascii=False)
                    entry["json"] = text[:600]
                except Exception as exc:
                    entry["json_error"] = str(exc)
            samples.append(entry)

    return {
        "tar": os.path.basename(tar_path),
        "sample_keys": len(groups),
        "suffix_counts": dict(suffixes.most_common()),
        "samples": samples,
    }


def _decode_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    return img.convert("RGB")


def extract_shard(
    tar_path: str,
    image_dir: Path,
    pair_registry: Dict[str, str],
    failures: collections.Counter,
) -> List[Dict[str, Any]]:
    """
    Pull every QA pair out of one shard.

    Image pairs are deduplicated by content hash: CDVQA repeats the same
    imagery across many questions, and writing one file per question is what
    inflated the old dataset to 200 files holding 6 distinct pictures.
    """
    samples: List[Dict[str, Any]] = []

    with tarfile.open(tar_path, "r") as tar:
        groups = group_members(tar)

        for key, members in groups.items():
            json_suffix = next((s for s in members if s.endswith("json")), None)
            t1_suffix = next((s for s in members if s.startswith("0.")), None)
            t2_suffix = next((s for s in members if s.startswith("1.")), None)

            if not json_suffix:
                failures["no_json_member"] += 1
                continue
            if not t1_suffix or not t2_suffix:
                failures["missing_image_member"] += 1
                continue

            try:
                meta = json.loads(tar.extractfile(members[json_suffix]).read())
            except Exception as exc:
                failures[f"json_decode:{type(exc).__name__}"] += 1
                continue

            convs = meta.get("conversations")
            if not isinstance(convs, list) or len(convs) < 2:
                failures["no_conversations"] += 1
                continue

            question = str(convs[0].get("value", "")).replace(BOILERPLATE, "").strip()
            answer = str(convs[1].get("value", "")).strip()
            if not question or not answer:
                failures["empty_qa"] += 1
                continue

            try:
                b1 = tar.extractfile(members[t1_suffix]).read()
                b2 = tar.extractfile(members[t2_suffix]).read()
            except Exception as exc:
                failures[f"image_read:{type(exc).__name__}"] += 1
                continue

            pair_id = hashlib.sha1(b1 + b2).hexdigest()[:16]

            if pair_id not in pair_registry:
                try:
                    img1 = _decode_image(b1)
                    img2 = _decode_image(b2)
                except Exception as exc:
                    failures[f"image_decode:{type(exc).__name__}"] += 1
                    continue
                if img1.size != img2.size:
                    img2 = img2.resize(img1.size, Image.BILINEAR)
                p1 = image_dir / f"{pair_id}_t1.png"
                p2 = image_dir / f"{pair_id}_t2.png"
                img1.save(p1)
                img2.save(p2)
                pair_registry[pair_id] = str(p1)

            samples.append(
                {
                    "pair_id": pair_id,
                    "t1": str(image_dir / f"{pair_id}_t1.png"),
                    "t2": str(image_dir / f"{pair_id}_t2.png"),
                    "question": question,
                    "answer": answer,
                    "source_key": key,
                }
            )

    return samples


# ---------------------------------------------------------------------------
# Balancing / splitting / gates
# ---------------------------------------------------------------------------


def cap_qa_per_pair(
    records: List[Dict[str, Any]], max_per_pair: int, seed: int
) -> List[Dict[str, Any]]:
    """Limit how many questions any single image pair contributes."""
    if max_per_pair <= 0:
        return records
    by_pair: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        by_pair[r["pair_id"]].append(r)
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for pair, rows in by_pair.items():
        rng.shuffle(rows)
        out.extend(rows[:max_per_pair])
    return out


def balance_answers(
    records: List[Dict[str, Any]], max_fraction: float, seed: int
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Subsample over-represented answer classes so no single answer exceeds
    `max_fraction` of the set. This is the direct fix for a model that learns
    P(answer) instead of reading the imagery.
    """
    by_answer: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        by_answer[r["answer"].strip().lower()].append(r)

    rng = random.Random(seed)
    for rows in by_answer.values():
        rng.shuffle(rows)

    # Iteratively shrink: the cap depends on the total, which shrinks as we cap.
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

    out: List[Dict[str, Any]] = []
    for a, rows in by_answer.items():
        out.extend(rows[: counts[a]])
    rng.shuffle(out)
    return out, counts


def split_by_pair(
    records: List[Dict[str, Any]], val_fraction: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pairs = sorted({r["pair_id"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_fraction))
    val_pairs = set(pairs[:n_val])
    train = [r for r in records if r["pair_id"] not in val_pairs]
    val = [r for r in records if r["pair_id"] in val_pairs]
    return train, val


def assert_quality(records: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    n = len(records)
    pairs = {r["pair_id"] for r in records}
    answers = collections.Counter(r["answer"].strip().lower() for r in records)

    stats = {
        "records": n,
        "unique_pairs": len(pairs),
        "unique_answers": len(answers),
        "qa_per_pair": round(n / len(pairs), 2) if pairs else 0,
        "top_answers": answers.most_common(12),
    }

    problems: List[str] = []
    if n < MIN_RECORDS:
        problems.append(f"only {n} records (need >= {MIN_RECORDS})")
    if len(pairs) < MIN_UNIQUE_PAIRS:
        problems.append(
            f"only {len(pairs)} unique image pairs (need >= {MIN_UNIQUE_PAIRS}). "
            "The previous build had 6, which is why the adapter went degenerate."
        )
    if len(answers) < MIN_UNIQUE_ANSWERS:
        problems.append(f"only {len(answers)} distinct answers (need >= {MIN_UNIQUE_ANSWERS})")
    if n:
        top_ans, top_cnt = answers.most_common(1)[0]
        frac = top_cnt / n
        stats["top_answer_fraction"] = round(frac, 4)
        if frac > MAX_TOP_ANSWER_FRACTION + 1e-6:
            problems.append(
                f"answer {top_ans!r} is {frac:.1%} of the set (max {MAX_TOP_ANSWER_FRACTION:.0%})"
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
            f.write(
                json.dumps(
                    {
                        "pair_id": row["pair_id"],
                        "t1": row["t1"],
                        "t2": row["t2"],
                        "question": row["question"],
                        "prefix": PROMPT_PREFIX + row["question"],
                        "suffix": row["answer"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"  wrote {len(rows):>7,} rows -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the CDVQA change-VQA dataset.")
    ap.add_argument("--out-dir", default="training/cdvqa")
    ap.add_argument("--shards", type=int, default=24, help="How many shards to pull.")
    ap.add_argument("--start-shard", type=int, default=0)
    ap.add_argument("--inspect", action="store_true", help="Dump one shard's structure and exit.")
    ap.add_argument("--max-qa-per-pair", type=int, default=12)
    ap.add_argument("--max-answer-fraction", type=float, default=0.30)
    ap.add_argument("--val-fraction", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # ---- inspect mode -----------------------------------------------------
    if args.inspect:
        path = fetch_shard(args.start_shard)
        if path is None:
            _fail(f"Shard {args.start_shard} does not exist in {REPO_ID}.")
        info = describe_tar(path)
        print(json.dumps(info, indent=2, ensure_ascii=False))
        (out_dir / "shard_report.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nWrote {out_dir / 'shard_report.json'}")
        return 0

    # ---- extract ----------------------------------------------------------
    pair_registry: Dict[str, str] = {}
    failures: collections.Counter = collections.Counter()
    all_samples: List[Dict[str, Any]] = []
    shards_used: List[str] = []

    print(f"Pulling shards {args.start_shard}..{args.start_shard + args.shards - 1} from {REPO_ID}")
    for i in range(args.start_shard, args.start_shard + args.shards):
        path = fetch_shard(i)
        if path is None:
            print(f"  shard {i:05d}: not present, stopping")
            break
        before_samples = len(all_samples)
        before_pairs = len(pair_registry)
        got = extract_shard(path, image_dir, pair_registry, failures)
        all_samples.extend(got)
        shards_used.append(SHARD_TEMPLATE.format(i))
        print(
            f"  shard {i:05d}: +{len(all_samples) - before_samples:,} QA, "
            f"+{len(pair_registry) - before_pairs:,} new pairs "
            f"(total {len(all_samples):,} QA / {len(pair_registry):,} pairs)"
        )

    if not shards_used:
        _fail("No shards were downloaded.")

    attempted = len(all_samples) + sum(failures.values())
    fail_frac = sum(failures.values()) / attempted if attempted else 1.0
    print(f"\nExtracted {len(all_samples):,} QA pairs over {len(pair_registry):,} unique image pairs")
    print(f"Extraction failures: {sum(failures.values()):,} ({fail_frac:.1%})")
    for reason, count in failures.most_common():
        print(f"  {reason}: {count:,}")

    if fail_frac > MAX_EXTRACT_FAILURE_FRACTION:
        _fail(
            f"{fail_frac:.1%} of samples failed to extract (max "
            f"{MAX_EXTRACT_FAILURE_FRACTION:.0%}). Breakdown: {dict(failures)}\n"
            f"Run `python training/prep_cdvqa.py --inspect` to see the real tar layout."
        )

    if not all_samples:
        _fail("Zero samples extracted.")

    # ---- cap, balance, gate ----------------------------------------------
    print(f"\nCapping to {args.max_qa_per_pair} questions per image pair ...")
    capped = cap_qa_per_pair(all_samples, args.max_qa_per_pair, args.seed)
    print(f"  {len(all_samples):,} -> {len(capped):,}")

    print(f"Balancing answers to <= {args.max_answer_fraction:.0%} each ...")
    balanced, caps = balance_answers(capped, args.max_answer_fraction, args.seed)
    print(f"  {len(capped):,} -> {len(balanced):,}")
    top_before = collections.Counter(r["answer"].strip().lower() for r in capped).most_common(5)
    top_after = collections.Counter(r["answer"].strip().lower() for r in balanced).most_common(5)
    print(f"  top answers before: {top_before}")
    print(f"  top answers after : {top_after}")

    print("\nRunning quality gates ...")
    stats = assert_quality(balanced, "CDVQA")
    print(
        f"  OK: {stats['records']:,} records, {stats['unique_pairs']:,} image pairs, "
        f"{stats['unique_answers']} distinct answers, "
        f"{stats['qa_per_pair']} QA/pair, top answer {stats['top_answer_fraction']:.1%}"
    )

    # ---- split & write ----------------------------------------------------
    print("\nSplitting by image pair and writing ...")
    train, val = split_by_pair(balanced, args.val_fraction, args.seed)
    overlap = {r["pair_id"] for r in train} & {r["pair_id"] for r in val}
    if overlap:
        _fail(f"train/val leakage: {len(overlap)} shared image pairs")

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)

    # Prune image files no longer referenced after capping/balancing.
    used_pairs = {r["pair_id"] for r in balanced}
    removed = 0
    for f in image_dir.glob("*_t?.png"):
        if f.stem.rsplit("_t", 1)[0] not in used_pairs:
            f.unlink()
            removed += 1
    if removed:
        print(f"  pruned {removed:,} unused image files")

    report = {
        "repo": REPO_ID,
        "shards": shards_used,
        "prompt_prefix": PROMPT_PREFIX,
        "extracted_qa": len(all_samples),
        "unique_pairs_extracted": len(pair_registry),
        "extraction_failures": dict(failures),
        "after_cap_per_pair": len(capped),
        "after_answer_balance": len(balanced),
        "answer_caps": caps,
        "stats": stats,
        "train": len(train),
        "val": len(val),
        "split": {"by": "pair_id", "val_fraction": args.val_fraction, "seed": args.seed},
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
