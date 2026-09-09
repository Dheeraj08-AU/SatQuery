# SatQuery AI

An interactive, agentic vision-language assistant for multimodal remote-sensing
image analysis through natural-language queries.

**Smart India Hackathon 2026 · Problem Statement 26167 · ISRO / Department of Space**

---

## What it does

Upload a single satellite image, a bi-temporal pair, or a co-registered
optical–SAR pair; ask a question in plain English. An agentic controller
classifies the task, validates the inputs, selects and sequences the appropriate
specialist tools, and returns an evidence-grounded answer with visual overlays,
typed confidence, a full audit trail, and downloadable reports.

```
Query ──▶ Intent router ──▶ Input validation ──▶ Tool selection & sequencing ──▶ Fusion of outputs
          (cloud LLM or      (format, bands,      (VQA / caption / grounding /     (text + geometry +
           offline rules)     CRS, overlap,        change / optical-SAR)            confidence + trace)
                              co-registration)
```

## Architecture

| Layer | Module | Responsibility |
|---|---|---|
| UI | `app.py` | Streamlit web app: upload, preview, query, evidence, downloads |
| Orchestration | `modules/agent_controller.py` | Classify → validate → select → sequence → execute → merge |
| Routing | `modules/router.py`, `modules/intent.py` | Deterministic offline intent classifier; shared task schema |
| Tools | `modules/model_registry.py` | Lazy-loaded specialist models, honest descriptors, applied parameters |
| Imagery | `modules/raster_io.py` | GeoTIFF / multispectral / SAR loading, dB conversion, speckle filtering, reprojection |
| Composition | `modules/composite.py` | Canvas geometry, prompt strings, PaliGemma location tokens |
| Change | `modules/change_detection.py` | Radiometric normalisation → CVA → Otsu → morphology → regions |
| Cross-modal | `modules/sar_analysis.py` | Backscatter-based water / built-up extraction |
| Validation | `modules/geo_validator.py` | Format, metadata and co-registration checking |
| Reporting | `modules/report.py` | Self-contained HTML report |
| Evaluation | `eval/run_benchmarks.py` | Scoring on prescribed test splits |

### Specialist tools

| Tool | Model / algorithm | RS-adapted |
|---|---|---|
| Single-image VQA | PaliGemma-3B + LoRA (VRSBench, RSVQA) | yes |
| Scene description | PaliGemma-3B + LoRA, `caption en` prefix | yes |
| Region grounding | PaliGemma-3B + LoRA, native `detect` / `<loc>` tokens | yes |
| Region grounding (fallback) | GroundingDINO-tiny, zero-shot | **no** |
| Change VQA | PaliGemma-3B + LoRA (CDVQA), T1\|T2 composite | yes |
| Change map | CVA + Otsu + morphology, deterministic | n/a |
| Optical–SAR VQA | PaliGemma-3B + LoRA (BigEarthNet-MM) | yes |
| SAR surface classification | Relative backscatter percentiles + optical cross-check | n/a |

Every tool reports its real model identifier and an explicit
`remote_sensing_adapted` flag in the execution trace. When a tool falls back to
a substitute — the stock detector, or the VQA adapter standing in for a missing
task adapter — the trace says so rather than presenting the fallback as the
adapted path.

## Mapping to the problem statement

| Requirement | Where |
|---|---|
| Remote-sensing adaptation of a visual/VL component | `training/train_vlm.py`, four LoRA adapters |
| Single-image VQA (mandatory) | `ModelRegistry.run_single_vqa` |
| Second single-image task (captioning **and** grounding) | `run_caption`, `run_grounding_vlm` |
| Bi-temporal change description / change VQA (mandatory) | `run_change_vqa` |
| Spatial change map | `run_change_map` → `modules/change_detection.py` |
| Cross-modal optical–SAR extraction | `run_optical_sar` + `run_sar_evidence` |
| Agentic tool selection, sequencing, execution | `AgentController._execute` |
| Input upload and compatibility checking | `modules/geo_validator.py`, `modules/raster_io.py` |
| Visual evidence | Grounding boxes, change overlay, SAR classification overlay |
| Confidence information | Typed per tool; see *Confidence* below |
| Auditable execution summary | `ExecutionTrace.as_dict()` |
| Downloadable reports | GeoJSON, execution JSON, standalone HTML |
| GeoTIFF / TIFF + benchmark PNG/JPEG support | `modules/raster_io.py` |

## Setup

```bash
pip install -r requirements.txt
```

Training additionally needs `pip install -r requirements-train.txt` (Linux/CUDA;
`bitsandbytes` is not available on Windows).

Optional `.env`:

```
GEMINI_API_KEY=...          # cloud intent router; absent is fine
```

Without a key the app starts normally and uses the offline rule-based router.
There is no configuration in which routing fails outright.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | unset | Cloud intent router |
| `SATQUERY_GEMINI_MODEL` | `gemini-2.5-flash-lite` | Router model id |
| `SATQUERY_BASE_VLM` | `google/paligemma-3b-pt-224` | Base VLM |
| `SATQUERY_VQA_ADAPTER` | `modules/satquery_vqa_adapter` | Local path or HF repo id |
| `SATQUERY_CHANGE_ADAPTER` | `modules/satquery_change_adapter` | ” |
| `SATQUERY_FUSION_ADAPTER` | `modules/satquery_fusion_adapter` | ” |
| `SATQUERY_GROUNDING_ADAPTER` | `modules/satquery_grounding_adapter` | ” |
| `SATQUERY_DEVICE` | auto | `cpu` or `cuda` |
| `SATQUERY_DISABLE_CACHE` | unset | `1` forces live inference |

## Running

```bash
streamlit run app.py
```

```bash
pytest tests -q
```

The unit suite covers the deterministic core — raster maths, change detection,
SAR evidence, composite geometry, intent routing, co-registration — and needs no
model weights, no GPU and no network.

## Data pipeline

Every builder validates what it produced and **aborts loudly** rather than
emitting a degenerate dataset. Each enforces floors on unique images, unique
answers, and the share of the single most common answer, and each splits
train/val **by image** so the same scene never appears on both sides.

```bash
python data_prep.py --inspect            # dump the real VRSBench schema
python data_prep.py                      # VQA + captions + grounding boxes

python training/prep_rsvqa.py --variant LR
python training/prep_cdvqa.py --shards 24 --max-qa-per-pair 12
python training/prep_bigearthnet.py --max-patches 12000
```

`--inspect` on any builder dumps the real schema without building anything.

## Training

One script trains every adapter, so the tasks cannot drift apart. Targets a
free-tier Colab T4 (compute capability 7.5, **no bfloat16** — fp16 throughout,
4-bit NF4 QLoRA, gradient checkpointing).

```bash
python training/train_vlm.py --task vqa       --data-dir data/vrsbench   --out modules/satquery_vqa_adapter
python training/train_vlm.py --task change    --data-dir training/cdvqa  --out modules/satquery_change_adapter
python training/train_vlm.py --task fusion    --data-dir data/bigearthnet --out modules/satquery_fusion_adapter
python training/train_vlm.py --task grounding --data-dir data/vrsbench   --out modules/satquery_grounding_adapter
```

Always pass `--hub-repo <user>/<name>`; `--save-steps` plus `--resume` survive a
Colab disconnect.

After training, the script decodes the held-out split and reports
`distinct_ratio`, `most_common_fraction` and `collapsed`. A model that has
learned the answer prior instead of reading the imagery emits near-constant
predictions, and that is reported as a first-class failure rather than hidden
behind an accuracy number.

## Evaluation

```bash
python eval/run_benchmarks.py --task all --limit 200 --no-cache
```

Reports exact match, token F1, grounding IoU@0.5 / mean IoU, RSVQA per-question-type
accuracy, latency, and cache-hit rate. Writes JSON and Markdown to `eval/results/`.

Two guards worth noting:

- **Majority-class baseline** is reported alongside exact match. Beating the
  baseline is the bar; an imbalanced split can otherwise make a constant
  predictor look competent.
- **`--no-cache`** is set automatically for benchmark runs. A cached run
  measures the cache, not the model.

## Confidence

Confidence is not a single quantity, so it is never merged into one number:

| Type | Meaning |
|---|---|
| `sequence_likelihood` | `exp(mean per-token log-probability)`. A fluency measure, **not** a probability that the answer is correct. |
| `detector_score` | GroundingDINO phrase-matching score, comparable across boxes only. |
| `otsu_separability` | Between-class / total variance of the change-magnitude histogram. |
| `deterministic` | Algorithmic output with no probabilistic component. |

The UI displays the value together with its type and definition.

## Design notes

**Co-registration** is checked in *pixels*, not coordinate units, after
transforming both footprints into a common CRS. Five outcomes are distinguished;
only two of them claim "verified". Non-georeferenced benchmark imagery is
reported as *unverifiable* and allowed through, never as verified. Pairs on
different grids are reprojected rather than rejected.

**Composites** fit each scene into its own square cell and letterbox, rather
than pasting side by side into a wide canvas that the processor then squashes
2:1. `modules/composite.py` is imported by both training and inference, so the
model is never served a rendering it did not train on.

**SAR** is decibel-converted (20·log₁₀ for integer DN amplitude, 10·log₁₀ for
float power), Lee speckle-filtered, then percentile-stretched. A naive stretch
of raw SAR intensity yields a near-black image because the distribution is
heavy-tailed.

**Change detection** normalises T2 onto T1's radiometry using interquartile
anchors before differencing. Tail percentiles are not robust: a change region
covering 10% of a scene at extreme brightness sits inside the 98th percentile,
so the anchor gets computed from the change itself.

## Known limitations

- Grounding falls back to stock GroundingDINO when the grounding adapter is
  absent. That fallback is **not** remote-sensing adapted and the trace says so.
- SAR surface classification uses scene-relative percentiles, not calibrated
  σ⁰ thresholds; absolute dB cut-offs would require radiometrically calibrated
  input.
- CPU inference is minutes per query for the 3B VLM. Use a GPU for live demos.
- Change detection runs on the display-stretched rendering rather than raw
  radiance, which is why step 1 is a relative fit rather than absolute
  calibration.

## Repository layout

```
app.py                      Streamlit application
modules/                    Inference layer (see architecture table)
training/                   Dataset builders + unified QLoRA trainer
eval/run_benchmarks.py      Benchmark harness
tests/test_core.py          Unit suite for the deterministic core
_deprecated_scripts/        Superseded scripts, retained with migration notes
training/_deprecated/       Superseded training code and degenerate datasets
```
