"""
Unified QLoRA fine-tuning for SatQuery AI on a Colab free-tier T4.

One code path trains every adapter, so the VQA and change stages cannot drift
apart. Composites and prompts come from `modules.composite`, the same module
inference uses.

T4-specific correctness
-----------------------
The T4 is compute capability 7.5 (Turing) and has NO bfloat16 support. The
previous scripts set `bnb_4bit_compute_dtype=torch.bfloat16` alongside
`fp16=True`, which is contradictory. Everything here is float16.

Anti-degeneracy
---------------
After training, the script measures how many DISTINCT predictions the model
produces across the validation set. A model that has learned the answer prior
instead of reading the imagery emits one or two strings for everything. That
is the exact failure the previous change adapter hit, so it is now measured
and reported as a first-class metric, and the run is marked FAILED if the
model collapses.

Usage (in Colab, after cloning the repo and running the data prep)
------------------------------------------------------------------
    python training/train_vlm.py --task vqa \
        --data-dir data/vrsbench --out modules/satquery_vqa_adapter \
        --epochs 1 --hub-repo <user>/satquery-vqa-adapter

    python training/train_vlm.py --task change \
        --data-dir training/cdvqa --out modules/satquery_change_adapter \
        --epochs 2 --hub-repo <user>/satquery-change-adapter

Free Colab disconnects. `--save-steps` writes checkpoints and `--resume`
picks up from the last one, so a dropped session costs minutes, not hours.
Always pass `--hub-repo`: it is the only thing that stops the weights from
living on exactly one machine.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import string
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make `modules` importable when run as `python training/train_vlm.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from torch.utils.data import Dataset

from modules.composite import (
    build_caption_prompt,
    build_change_prompt,
    build_vqa_prompt,
    make_pair_composite,
    make_single_image,
)

# LoRA target suffixes.
#   Gemma language model : q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
#   SigLIP vision tower  : q_proj k_proj v_proj out_proj fc1 fc2
LM_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
VISION_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class SatQueryDataset(Dataset):
    """Holds already-normalised records; images are opened lazily in collate."""

    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


def load_task_data(
    task: str, data_dir: Path, max_answer_words: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Normalise the on-disk JSONL into a single record shape."""

    def keep(answer: str) -> bool:
        return 0 < len(answer.split()) <= max_answer_words

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []

    if task == "vqa":
        for split, sink in (("train", train), ("val", val)):
            for row in read_jsonl(data_dir / f"{split}_vqa.jsonl"):
                if keep(row["answer"]):
                    sink.append(
                        {
                            "kind": "vqa",
                            "images": [row["image"]],
                            "question": row["question"],
                            "answer": row["answer"],
                            "key": row.get("image_id", row["image"]),
                        }
                    )
            for row in read_jsonl(data_dir / f"{split}_caption.jsonl"):
                if keep(row["answer"]):
                    sink.append(
                        {
                            "kind": "caption",
                            "images": [row["image"]],
                            "question": row["question"],
                            "answer": row["answer"],
                            "key": row.get("image_id", row["image"]),
                        }
                    )

    elif task == "change":
        for split, sink in (("train", train), ("val", val)):
            for row in read_jsonl(data_dir / f"{split}.jsonl"):
                if keep(row["suffix"]):
                    sink.append(
                        {
                            "kind": "change",
                            "images": [row["t1"], row["t2"]],
                            "question": row["question"],
                            "answer": row["suffix"],
                            "key": row["pair_id"],
                        }
                    )
    else:
        raise ValueError(f"Unknown task {task!r}")

    if not train:
        raise SystemExit(
            f"No training records found for task '{task}' under {data_dir}. "
            f"Run the dataset builder first."
        )
    return train, val


def build_prompt(record: Dict[str, Any]) -> str:
    kind = record["kind"]
    if kind == "vqa":
        return build_vqa_prompt(record["question"])
    if kind == "caption":
        return build_caption_prompt()
    if kind == "change":
        return build_change_prompt(record["question"])
    raise ValueError(f"Unknown record kind {kind!r}")


def render_image(record: Dict[str, Any], size: int) -> Image.Image:
    """
    Build the exact pixel input the model will see.

    Training data is PNG/JPEG (benchmark imagery), so PIL is correct here.
    Inference goes through modules.raster_io for GeoTIFF/SAR, then hands the
    result to these same composite functions.
    """
    paths = record["images"]
    if len(paths) == 1:
        return make_single_image(Image.open(paths[0]).convert("RGB"), size=size)
    img1 = Image.open(paths[0]).convert("RGB")
    img2 = Image.open(paths[1]).convert("RGB")
    return make_pair_composite(img1, img2, size=size, layout="horizontal")


def make_collate_fn(processor, image_size: int, max_length: int):
    def collate(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [build_prompt(ex) for ex in examples]
        answers = [ex["answer"] for ex in examples]
        images = [render_image(ex, image_size) for ex in examples]

        batch = processor(
            text=prompts,
            images=images,
            suffix=answers,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
        )
        # Forward every key the processor produced. PaliGemma needs
        # token_type_ids to mask the prompt out of the loss; the previous
        # collate dropped it.
        return {k: v for k, v in batch.items()}

    return collate


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def discover_lora_targets(model, include_vision: bool) -> Tuple[List[str], Dict[str, int]]:
    """
    Return the FULL module names to attach LoRA to, plus a per-suffix count.

    Full names, not suffixes, on purpose. PEFT matches bare suffixes against
    every module in the model, so a target list of ["q_proj", "k_proj",
    "v_proj", "o_proj"] - the previous configuration - also matched the SigLIP
    vision tower's attention, despite the surrounding code claiming the vision
    encoder was frozen. Freezing before `get_peft_model` does not stop LoRA
    from adding new trainable weights to those same modules.

    Adapting the vision tower is in fact desirable here (nadir satellite
    imagery is a long way from SigLIP's natural-image pretraining), but it has
    to be a deliberate, recorded choice rather than an accident. Full names
    make `--no-include-vision` actually exclude the vision tower, and make the
    adapter's real scope auditable from adapter_config.json.
    """
    matched: Dict[str, int] = collections.Counter()
    full_names: List[str] = []
    linear_names = ("Linear", "Linear4bit", "Linear8bitLt")

    for name, module in model.named_modules():
        if module.__class__.__name__ not in linear_names:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if "vision_tower" in name:
            if include_vision and leaf in VISION_TARGETS:
                matched[f"vision/{leaf}"] += 1
                full_names.append(name)
        elif "multi_modal_projector" in name:
            continue
        elif leaf in LM_TARGETS:
            matched[f"lm/{leaf}"] += 1
            full_names.append(name)

    return sorted(full_names), dict(matched)


def load_model(args, processor):
    from transformers import BitsAndBytesConfig, PaliGemmaForConditionalGeneration
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    # T4 (sm_75) has no bfloat16. float16 throughout.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print(f"Loading {args.model_id} in 4-bit NF4 (compute dtype float16) ...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    if args.init_from and args.init_from != "base":
        print(f"Continuing from existing adapter: {args.init_from}")
        model = PeftModel.from_pretrained(model, args.init_from, is_trainable=True)
    else:
        targets, matched = discover_lora_targets(model, args.include_vision)
        print(f"LoRA scope (include_vision={args.include_vision}):")
        for k in sorted(matched):
            print(f"    {k:<24} {matched[k]:>4} modules")
        print(f"  -> {len(targets)} modules targeted by full name")
        print(f"     e.g. {targets[:2]}")
        if not targets:
            raise SystemExit("No LoRA target modules matched. Aborting.")

        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(_PUNCT)
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


@torch.no_grad()
def evaluate(
    model,
    processor,
    records: List[Dict[str, Any]],
    image_size: int,
    max_new_tokens: int,
    limit: int,
) -> Dict[str, Any]:
    """Greedy-decode the validation set and score it."""
    model.eval()
    subset = records[:limit]
    if not subset:
        return {"n": 0}

    preds: List[str] = []
    golds: List[str] = []

    for i, ex in enumerate(subset):
        image = render_image(ex, image_size)
        prompt = build_prompt(ex)
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {
            k: (v.to(model.device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        with torch.autocast("cuda", dtype=torch.float16):
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        pred = processor.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        preds.append(pred)
        golds.append(ex["answer"])

        if i < 10:
            print(f"  [{i}] Q: {ex['question'][:80]}")
            print(f"      gold: {ex['answer'][:120]}")
            print(f"      pred: {pred[:120]}")

    exact = sum(normalize_answer(p) == normalize_answer(g) for p, g in zip(preds, golds))
    f1 = sum(token_f1(p, g) for p, g in zip(preds, golds))
    distinct = len({normalize_answer(p) for p in preds})

    counts = collections.Counter(normalize_answer(p) for p in preds)
    top_pred, top_count = counts.most_common(1)[0]

    metrics = {
        "n": len(subset),
        "exact_match": round(exact / len(subset), 4),
        "token_f1": round(f1 / len(subset), 4),
        "distinct_predictions": distinct,
        "distinct_ratio": round(distinct / len(subset), 4),
        "most_common_prediction": top_pred,
        "most_common_fraction": round(top_count / len(subset), 4),
        "gold_distinct_ratio": round(
            len({normalize_answer(g) for g in golds}) / len(subset), 4
        ),
    }

    # The degeneracy check that would have caught the previous adapter.
    metrics["collapsed"] = bool(
        metrics["distinct_ratio"] < 0.10
        or metrics["most_common_fraction"] > 0.60
    )
    model.train()
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="QLoRA fine-tune PaliGemma for SatQuery AI.")
    ap.add_argument("--task", choices=["vqa", "change"], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default="google/paligemma-3b-pt-224")
    ap.add_argument("--init-from", default="base", help="'base' or a path to an existing adapter.")

    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-answer-words", type=int, default=120)
    ap.add_argument("--include-vision", action="store_true", default=True,
                    help="Attach LoRA to the SigLIP vision tower too (recommended: "
                         "satellite imagery is far from the natural-image pretraining "
                         "distribution, so the visual features need adapting).")
    ap.add_argument("--no-include-vision", dest="include_vision", action="store_false")

    ap.add_argument("--save-steps", type=int, default=250)
    ap.add_argument("--logging-steps", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-limit", type=int, default=200)
    ap.add_argument("--eval-max-new-tokens", type=int, default=48)
    ap.add_argument("--hub-repo", default=None, help="Push the adapter here when done.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoProcessor, Trainer, TrainingArguments, set_seed

    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. This script requires a GPU (Colab T4).")
    gpu = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu} (sm_{cap[0]}{cap[1]})")
    if cap[0] >= 8:
        print("  note: this GPU supports bf16; the script still uses fp16 for T4 parity.")

    data_dir = Path(args.data_dir)
    train_records, val_records = load_task_data(args.task, data_dir, args.max_answer_words)

    train_keys = {r["key"] for r in train_records}
    val_keys = {r["key"] for r in val_records}
    leak = train_keys & val_keys
    if leak:
        raise SystemExit(f"train/val leakage: {len(leak)} shared image keys. Rebuild the dataset.")

    kinds = collections.Counter(r["kind"] for r in train_records)
    print(f"\nTrain records: {len(train_records):,}  {dict(kinds)}")
    print(f"Val records  : {len(val_records):,}")
    print(f"Unique train image keys: {len(train_keys):,}")
    gold_variety = len({r['answer'].strip().lower() for r in train_records})
    print(f"Distinct train answers  : {gold_variety:,}")
    if gold_variety < 20:
        raise SystemExit(
            f"Only {gold_variety} distinct answers in the training set. "
            "Refusing to train - this is the degenerate-data condition."
        )

    processor = AutoProcessor.from_pretrained(args.model_id)
    size_cfg = processor.image_processor.size
    image_size = int(size_cfg.get("height") or size_cfg.get("shortest_edge") or 224)
    print(f"Processor image size: {image_size}")

    model = load_model(args, processor)

    out_dir = Path(args.out)
    ckpt_dir = out_dir.parent / f"{out_dir.name}_checkpoints"

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        # T4: fp16 only. bf16 is unsupported on sm_75.
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SatQueryDataset(train_records),
        data_collator=make_collate_fn(processor, image_size, args.max_length),
    )

    resume = args.resume and ckpt_dir.exists() and any(ckpt_dir.glob("checkpoint-*"))
    print(f"\nStarting training (resume={resume}) ...")
    trainer.train(resume_from_checkpoint=resume)

    print(f"\nSaving adapter -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))

    # ---- evaluation ------------------------------------------------------
    metrics: Dict[str, Any] = {}
    if val_records:
        print(f"\nEvaluating on {min(len(val_records), args.eval_limit)} held-out records ...")
        metrics = evaluate(
            model, processor, val_records, image_size,
            args.eval_max_new_tokens, args.eval_limit,
        )
        print("\n" + json.dumps(metrics, indent=2))

        if metrics.get("collapsed"):
            print(
                "\n" + "!" * 70
                + "\nMODEL COLLAPSED: predictions are near-constant across the val set."
                f"\n  distinct_ratio={metrics['distinct_ratio']} "
                f"most_common={metrics['most_common_prediction']!r} "
                f"({metrics['most_common_fraction']:.0%})"
                "\nThis adapter has learned the answer prior, not the imagery."
                "\nDo NOT ship it. Check dataset diversity and lower the learning rate.\n"
                + "!" * 70
            )

    run_report = {
        "task": args.task,
        "model_id": args.model_id,
        "init_from": args.init_from,
        "image_size": image_size,
        "lora_rank": args.rank,
        "include_vision_tower": args.include_vision,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "unique_train_keys": len(train_keys),
        "distinct_train_answers": gold_variety,
        "hyperparameters": {
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "batch_size": args.bs,
            "grad_accum": args.accum,
            "effective_batch": args.bs * args.accum,
            "lr": args.lr,
            "scheduler": "cosine",
            "precision": "fp16 / 4-bit NF4",
        },
        "eval": metrics,
    }
    (out_dir / "training_report.json").write_text(
        json.dumps(run_report, indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote {out_dir / 'training_report.json'}")

    # ---- push to hub -----------------------------------------------------
    if args.hub_repo:
        from huggingface_hub import HfApi

        print(f"\nPushing adapter to https://huggingface.co/{args.hub_repo} ...")
        api = HfApi()
        api.create_repo(repo_id=args.hub_repo, exist_ok=True, private=True)
        api.upload_folder(repo_id=args.hub_repo, folder_path=str(out_dir))
        print("Pushed. The weights now exist somewhere other than this VM.")
    else:
        print(
            "\nWARNING: --hub-repo was not set, so these weights exist only in this "
            "Colab session. Download them or push them now."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
