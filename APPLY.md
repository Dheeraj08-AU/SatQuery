# SatQuery AI — update pack

Extract this archive **over the root of your `SatQuery-main` folder** on the
device that has the repo. The paths inside mirror the repo layout, so files
land in the right place and overwrite the old versions.

## What is in here

| Path | Status | Notes |
|---|---|---|
| `modules/raster_io.py` | **new** | GeoTIFF / multispectral / SAR pixel loader. Nothing existed before; every read went through `PIL.Image.open`. |
| `modules/composite.py` | **new** | Shared two-image composite + prompt strings. Imported by BOTH training and inference so they cannot drift. |
| `data_prep.py` | **replaced** | VRSBench builder. No fallback strings, hard quality gates, split by image. |
| `training/prep_cdvqa.py` | **replaced** | CDVQA builder. No silent `except: pass`, pair dedup, answer balancing, split by image pair. |
| `training/train_vlm.py` | **new** | Single QLoRA training script for both adapters. Replaces `finetune_vlm.py` and `finetune_change_vlm.py`. |
| `requirements.txt` | **replaced** | Pinned, with upper bounds. |
| `requirements-train.txt` | **new** | Colab-only extras (`bitsandbytes` is Linux/CUDA only). |
| `training/_deprecated/README.md` | **new** | Documents exactly what was wrong with the quarantined files. |
| `quarantine_broken.ps1` | **new** | Run once. Moves the broken scripts and degenerate datasets out of the way. |

Nothing else in the repo is touched. `app.py`, `modules/agent_controller.py`,
`modules/model_registry.py` and `modules/geo_validator.py` are **unchanged** —
those are the next batch of work.

## Steps

1. Back up your current folder (or commit what you have) before extracting.

2. Extract this archive over `SatQuery-main/`, allowing overwrites.

3. Run the quarantine script once, from the repo root. An archive cannot
   delete or move files, so this step is separate:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\quarantine_broken.ps1
   ```

   It moves `training/finetune_vlm.py`, `training/finetune_change_vlm.py`,
   `training/cdvqa_formatted/` and `data/vrsbench_train.json` into
   `training/_deprecated/`. It skips anything already moved and never
   overwrites. Nothing is deleted.

4. Install and inspect. Run both `--inspect` commands first — they take about
   a minute each and reveal the real dataset schemas, which is the one thing
   that could not be verified while writing these scripts:

   ```powershell
   pip install -r requirements-train.txt
   python data_prep.py --inspect
   python training\prep_cdvqa.py --inspect
   ```

5. Build the datasets:

   ```powershell
   python data_prep.py --max-answer-fraction 0.30
   python training\prep_cdvqa.py --shards 24 --max-qa-per-pair 12
   ```

6. Train on Colab (T4). Upload the repo or `git clone` it there, then:

   ```bash
   python training/train_vlm.py --task vqa \
     --data-dir data/vrsbench --out modules/satquery_vqa_adapter \
     --max-steps 1200 --save-steps 100 --resume \
     --hub-repo YOUR_HF_USER/satquery-vqa-adapter

   python training/train_vlm.py --task change \
     --data-dir training/cdvqa --out modules/satquery_change_adapter \
     --max-steps 800 --save-steps 100 --resume \
     --hub-repo YOUR_HF_USER/satquery-change-adapter
   ```

   `--hub-repo` is not optional. Your current adapter weights exist on exactly
   one device and are excluded by `.gitignore`; if that machine dies, the
   project dies with it.

## Important

None of this code has been executed. It was written and reviewed on a machine
with no Python interpreter available. If a script aborts, that is often the
quality gates working as intended — read the message before assuming a bug.
Send back the traceback plus the `--inspect` output and it can be corrected
against the real data structure.

## Expected failure modes, and what they mean

- **`DATASET BUILD FAILED ... unparseable records`** — the VRSBench schema is
  not what the normaliser expects. The message prints the observed keys and
  sample records. This is the check that the old pipeline lacked; it is why
  1000 identical fake rows were produced silently instead.

- **`DATASET BUILD FAILED ... only N unique image pairs`** — too few shards
  pulled. Raise `--shards`.

- **`MODEL COLLAPSED`** at the end of training — the adapter is emitting a
  near-constant answer. Do not ship it. Check `build_report.json` for answer
  diversity and lower `--lr`.
