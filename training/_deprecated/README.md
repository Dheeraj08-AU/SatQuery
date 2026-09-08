# Quarantined — do not run, do not train on

Everything in this folder is kept only as evidence of what went wrong. It is
superseded by `training/train_vlm.py`, `data_prep.py`, and
`training/prep_cdvqa.py`.

## `finetune_vlm.py`, `finetune_change_vlm.py`

Replaced by the single `training/train_vlm.py`.

Defects:

1. **Silent placeholder injection.** Both scripts built records with
   `item.get('question') or item.get('caption') or 'Describe the visible
   features.'` and `item.get('answer') or item.get('label') or 'Urban and
   natural land-cover.'`. When the real VRSBench schema did not match those
   keys, the `or` chain manufactured a synthetic dataset instead of raising.
   The result is preserved as `vrsbench_train_BROKEN.json`.
2. **bf16 on a T4.** `bnb_4bit_compute_dtype=torch.bfloat16` was combined with
   `fp16=True`. The T4 is compute capability 7.5 and has no bfloat16 support.
3. **Suffix-matched LoRA targets.** `target_modules=["q_proj","k_proj",
   "v_proj","o_proj"]` matched the SigLIP vision tower as well as the language
   model, so the vision encoder was adapted despite the code above it
   explicitly setting `requires_grad = False` on those parameters. Freezing
   before `get_peft_model` does not prevent LoRA from adding new trainable
   weights to the same modules.
4. **`token_type_ids` dropped.** The collate function forwarded only
   `input_ids`, `attention_mask`, `pixel_values` and `labels`. PaliGemma uses
   `token_type_ids` to mask the prompt out of the loss.
5. **Leaky split.** Train/val were split by QA pair, not by image. Both
   VRSBench and CDVQA carry many questions per image, so the same imagery
   landed on both sides of the split and validation numbers were inflated.

## `vrsbench_train_BROKEN.json`

1000 records. Distinct ids: **1**. Distinct images: **1**. Distinct
questions: **1**. Distinct answers: **1** — every row is
`{"id": "vrsbench_Final_Data/v1.2", "image": "images/Final_Data/v1.2.jpg",
"question": "Describe the visible features in this image.", "answer": "Urban
and natural land-cover."}`, i.e. 1000 copies of the fallback strings above,
pointing at an image path that does not exist.

## `cdvqa_formatted_BROKEN/`

200 composite JPEGs, of which only **6** are distinct images (checked by MD5:
45/45/43/34/23/10 duplicates). 190 training rows, 65% of them labelled bare
`yes` or `no`. Two epochs at `lr=2e-4` over that teaches the model the answer
prior — P(no) ≈ 0.58 — and nothing about the imagery. That is the origin of
the "degenerate fixed yes/no pattern regardless of content difference"
recorded in `run_change_analysis_experimental_multiprobe`.

Root causes, all fixed in the new `prep_cdvqa.py`: a bare
`except Exception: pass` swallowed nearly every sample; one composite file was
written per *question* rather than per *image pair*; and there was no
diversity floor or answer balancing.

## Safe to delete

Once the rebuilt datasets pass their quality gates, this whole folder can be
removed. It is retained now so the failure is documented rather than
rediscovered.
