"""
RSVQA builder (RSVQA-LR / RSVQA-HR).

The problem statement names "VRSBench and RSVQA" as the benchmarks for
single-image captioning, grounding and visual question answering. Only VRSBench
was being built, so final scoring on the prescribed RSVQA test split had no code
path at all.

RSVQA is also qualitatively different from VRSBench and worth training on: its
questions are template-generated over OpenStreetMap-derived ground truth, with
four types - presence, count, comparison, and rural/urban - and short
closed-vocabulary answers. That complements VRSBench's free-form annotations and
gives the adapter a second supervision style.

Official layout (either downloaded from HF or pointed at with --data-dir):
    LR_split_train_questions.json   {"questions": [{id, img_id, type, question, answers_ids, active}]}
    LR_split_train_answers.json     {"answers":   [{id, question_id, answer, active}]}
    LR_split_train_images.json      {"images":    [{id, type, active}]}
    Images_LR/<img_id>.tif

Usage
-----
    python training/prep_rsvqa.py --inspect
    python training/prep_rsvqa.py --variant LR
    python training/prep_rsvqa.py --variant LR --data-dir D:/datasets/RSVQA-LR

Outputs (under --out-dir, default `data/rsvqa`):
    train.jsonl / val.jsonl / test.jsonl
        {"image", "image_id", "question", "answer", "type"}
    build_report.json
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CANDIDATE_REPOS = {
    "LR": ["jonathan-roberts1/RSVQA-LR", "RSVQA/RSVQA-LR", "flax-sentence-embeddings/RSVQA_LR"],
    "HR": ["jonathan-roberts1/RSVQA-HR", "RSVQA/RSVQA-HR"],
}

IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

# Answers appearing more often than this share of a question type are capped.
MAX_ANSWER_FRACTION_PER_TYPE = 0.35

MIN_RECORDS = 2000
MIN_UNIQUE_IMAGES = 100
MIN_UNIQUE_ANSWERS = 8


class DatasetBuildError(RuntimeError):
    pass


def _fail(msg: str) -> None:
    raise DatasetBuildError(msg)


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------


def fetch_from_hub(variant: str, dest: Path) -> Path:
    """Download and unpack an RSVQA release from the Hub into `dest`."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    errors: List[str] = []

    for repo in CANDIDATE_REPOS[variant]:
        try:
            files = api.list_repo_files(repo_id=repo, repo_type="dataset")
        except Exception as exc:
            errors.append(f"{repo}: {type(exc).__name__}: {exc}")
            continue

        print(f"  found {repo} ({len(files)} files)")
        dest.mkdir(parents=True, exist_ok=True)
        pulled = 0
        for name in files:
            low = name.lower()
            if not (low.endswith(".json") or low.endswith(".zip")):
                continue
            try:
                local = hf_hub_download(repo_id=repo, filename=name, repo_type="dataset")
            except Exception as exc:
                print(f"    skip {name}: {exc}")
                continue
            target = dest / Path(name).name
            if low.endswith(".zip"):
                marker = dest / (Path(name).stem + ".extracted")
                if not marker.exists():
                    print(f"    extracting {name} ...")
                    with zipfile.ZipFile(local, "r") as zf:
                        zf.extractall(dest)
                    marker.write_text("ok", encoding="utf-8")
            else:
                if not target.exists():
                    target.write_bytes(Path(local).read_bytes())
            pulled += 1

        if pulled:
            return dest
        errors.append(f"{repo}: no .json or .zip assets")

    _fail(
        f"Could not obtain RSVQA-{variant}. Tried:\n  " + "\n  ".join(errors)
        + "\n\nDownload it manually from https://rsvqa.sylvainlobry.com/ and pass "
        "--data-dir pointing at the extracted folder."
    )


def find_json(root: Path, variant: str, split: str, kind: str) -> Optional[Path]:
    """Locate e.g. LR_split_train_questions.json, tolerating naming variation."""
    wanted = (variant.lower(), split.lower(), kind.lower())
    best: Optional[Path] = None
    for path in root.rglob("*.json"):
        low = path.name.lower()
        if all(token in low for token in wanted):
            return path
        if split.lower() in low and kind.lower() in low and best is None:
            best = path
    return best


def index_images(root: Path) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            index.setdefault(path.stem, str(path))
    return index


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def load_block(path: Path, key: str) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if key in data and isinstance(data[key], list):
            return data[key]
        lists = [v for v in data.values() if isinstance(v, list)]
        if len(lists) == 1:
            return lists[0]
        _fail(
            f"{path.name} is an object with keys {list(data.keys())[:10]}; expected "
            f"a list under {key!r}."
        )
    _fail(f"{path.name} parsed to {type(data).__name__}, expected list or object.")
    return []


def build_split(
    questions_path: Path,
    answers_path: Path,
    image_index: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], collections.Counter]:
    questions = load_block(questions_path, "questions")
    answers = load_block(answers_path, "answers")

    # question id -> first active answer
    answer_by_qid: Dict[Any, str] = {}
    for a in answers:
        if not isinstance(a, dict):
            continue
        if a.get("active") is False:
            continue
        qid = a.get("question_id", a.get("questions_id"))
        text = a.get("answer")
        if qid is None or text is None:
            continue
        answer_by_qid.setdefault(qid, str(text).strip())

    records: List[Dict[str, Any]] = []
    failures: collections.Counter = collections.Counter()

    for q in questions:
        if not isinstance(q, dict):
            failures["not_a_dict"] += 1
            continue
        if q.get("active") is False:
            failures["inactive"] += 1
            continue

        qid = q.get("id")
        question = q.get("question")
        img_id = q.get("img_id", q.get("image_id"))
        qtype = str(q.get("type", "unknown")).strip().lower()

        if question is None or img_id is None:
            failures["missing_question_or_image"] += 1
            continue

        answer = answer_by_qid.get(qid)
        if answer is None:
            failures["no_answer_for_question"] += 1
            continue

        path = image_index.get(str(img_id))
        if path is None:
            failures["image_file_missing"] += 1
            continue

        records.append(
            {
                "image": path,
                "image_id": str(img_id),
                "question": str(question).strip(),
                "answer": answer,
                "type": qtype,
            }
        )

    return records, failures


# ---------------------------------------------------------------------------
# Balancing / gates
# ---------------------------------------------------------------------------


def balance_per_type(
    records: List[Dict[str, Any]], max_fraction: float, seed: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """
    Balance within each question type.

    RSVQA presence questions are heavily yes-skewed and comparison questions
    heavily no-skewed. Balancing globally hides both, because the two skews
    partially cancel while each individual question type stays trivially
    guessable.
    """
    by_type: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        by_type[r["type"]].append(r)

    rng = random.Random(seed)
    kept: List[Dict[str, Any]] = []
    report: Dict[str, Dict[str, int]] = {}

    for qtype, rows in by_type.items():
        by_answer: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in rows:
            by_answer[r["answer"].strip().lower()].append(r)
        for group in by_answer.values():
            rng.shuffle(group)

        counts = {a: len(g) for a, g in by_answer.items()}
        if 2 <= len(counts) <= 12:
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
        report[qtype] = counts

    rng.shuffle(kept)
    return kept, report


def assert_quality(records: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    n = len(records)
    images = {r["image_id"] for r in records}
    answers = collections.Counter(r["answer"].strip().lower() for r in records)
    types = collections.Counter(r["type"] for r in records)

    stats = {
        "records": n,
        "unique_images": len(images),
        "unique_answers": len(answers),
        "question_types": dict(types),
        "top_answers": answers.most_common(10),
    }

    problems: List[str] = []
    if n < MIN_RECORDS:
        problems.append(f"only {n} records (need >= {MIN_RECORDS})")
    if len(images) < MIN_UNIQUE_IMAGES:
        problems.append(f"only {len(images)} unique images (need >= {MIN_UNIQUE_IMAGES})")
    if len(answers) < MIN_UNIQUE_ANSWERS:
        problems.append(f"only {len(answers)} distinct answers (need >= {MIN_UNIQUE_ANSWERS})")
    if n:
        top_ans, top_cnt = answers.most_common(1)[0]
        stats["top_answer_fraction"] = round(top_cnt / n, 4)

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
    ap = argparse.ArgumentParser(description="Build the RSVQA dataset splits.")
    ap.add_argument("--variant", choices=["LR", "HR"], default="LR")
    ap.add_argument("--out-dir", default="data/rsvqa")
    ap.add_argument("--data-dir", default=None, help="Existing local RSVQA folder.")
    ap.add_argument("--cache-dir", default="data/_rsvqa_raw")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--max-answer-fraction", type=float, default=MAX_ANSWER_FRACTION_PER_TYPE)
    ap.add_argument("--balance-train-only", action="store_true", default=True,
                    help="Never rebalance the test split - it is the prescribed benchmark.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.data_dir:
        root = Path(args.data_dir)
        if not root.exists():
            _fail(f"--data-dir {root} does not exist.")
    else:
        root = Path(args.cache_dir) / args.variant
        print(f"Fetching RSVQA-{args.variant} ...")
        fetch_from_hub(args.variant, root)

    print(f"\nIndexing images under {root} ...")
    image_index = index_images(root)
    print(f"  {len(image_index):,} image files")

    found: Dict[str, Dict[str, Optional[Path]]] = {}
    for split in ("train", "val", "test"):
        found[split] = {
            "questions": find_json(root, args.variant, split, "questions"),
            "answers": find_json(root, args.variant, split, "answers"),
        }

    print("\nAnnotation files located:")
    for split, paths in found.items():
        q = paths["questions"].name if paths["questions"] else "MISSING"
        a = paths["answers"].name if paths["answers"] else "MISSING"
        print(f"  {split:<6} questions={q}  answers={a}")

    if args.inspect:
        report: Dict[str, Any] = {
            "root": str(root),
            "image_count": len(image_index),
            "sample_images": list(image_index.items())[:5],
            "annotation_files": {
                s: {k: (str(v) if v else None) for k, v in p.items()}
                for s, p in found.items()
            },
            "all_json_files": [str(p.relative_to(root)) for p in root.rglob("*.json")][:60],
        }
        qpath = found["test"]["questions"] or found["train"]["questions"]
        if qpath:
            block = load_block(qpath, "questions")
            report["question_sample"] = block[:3]
            report["question_count"] = len(block)
        apath = found["test"]["answers"] or found["train"]["answers"]
        if apath:
            block = load_block(apath, "answers")
            report["answer_sample"] = block[:3]
            report["answer_count"] = len(block)

        print(json.dumps(report, indent=2, default=str)[:6000])
        (out_dir / "schema_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nWrote {out_dir / 'schema_report.json'}")
        return 0

    if not any(p["questions"] and p["answers"] for p in found.values()):
        _fail(
            f"No usable question/answer file pair found under {root}. "
            f"Run with --inspect to list what is actually there."
        )

    summary: Dict[str, Any] = {}
    all_failures: Dict[str, Dict[str, int]] = {}
    balance_reports: Dict[str, Any] = {}

    for split in ("train", "val", "test"):
        qpath = found[split]["questions"]
        apath = found[split]["answers"]
        if not (qpath and apath):
            print(f"\n{split}: skipped (annotations not found)")
            continue

        print(f"\n{split}: parsing {qpath.name} + {apath.name} ...")
        records, failures = build_split(qpath, apath, image_index)
        all_failures[split] = dict(failures)
        print(f"  {len(records):,} records; skipped {sum(failures.values()):,}")
        for reason, count in failures.most_common(5):
            print(f"    {reason}: {count:,}")

        if not records:
            print(f"  {split}: nothing usable, skipping")
            continue

        # The test split is the prescribed benchmark. Rebalancing it would make
        # our reported numbers incomparable with everyone else's.
        if split == "test":
            print("  test split left unbalanced (prescribed benchmark)")
            final = records
        else:
            final, brep = balance_per_type(records, args.max_answer_fraction, args.seed)
            balance_reports[split] = brep
            print(f"  balanced per question type: {len(records):,} -> {len(final):,}")

        if split == "train":
            summary["train_stats"] = assert_quality(final, "RSVQA train")

        write_jsonl(out_dir / f"{split}.jsonl", final)
        summary[f"{split}_records"] = len(final)
        summary[f"{split}_images"] = len({r["image_id"] for r in final})
        summary[f"{split}_types"] = dict(
            collections.Counter(r["type"] for r in final)
        )

    report = {
        "variant": args.variant,
        "root": str(root),
        "images_indexed": len(image_index),
        "skipped": all_failures,
        "balancing": balance_reports,
        "max_answer_fraction_per_type": args.max_answer_fraction,
        **summary,
    }
    (out_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"\nWrote {out_dir / 'build_report.json'}")
    print("\nBuild complete.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except DatasetBuildError as exc:
        print(f"\n{'!' * 70}\nDATASET BUILD FAILED\n{'!' * 70}\n{exc}\n", file=sys.stderr)
        sys.exit(2)
