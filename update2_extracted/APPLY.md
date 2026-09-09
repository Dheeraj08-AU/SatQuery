# SatQuery AI — update pack 2: the inference layer

Extract over the root of `SatQuery-main`, same as pack 1. This covers all six
items on the list: registry routed through raster_io/composite, the
`run_optical_sar` crash, the fabricated execution trace, the co-registration
tolerance and PNG bypass, real change detection, the offline router, and the
benchmark harness.

## Files

| Path | Status |
|---|---|
| `modules/model_registry.py` | **replaced** — all pixel I/O via raster_io, real tool descriptors, `applied_parameters`, typed confidence, signature fixed |
| `modules/agent_controller.py` | **replaced** — offline fallback routing, multi-tool sequencing, truthful trace |
| `modules/geo_validator.py` | **replaced** — pixel-scale tolerance, CRS transform, five honest statuses |
| `modules/change_detection.py` | **new** — radiometric normalisation → CVA → Otsu → morphology → regions |
| `modules/sar_analysis.py` | **new** — real optical/SAR cross-modal extraction |
| `modules/router.py` | **new** — deterministic offline intent router |
| `modules/intent.py` | **new** — shared task/intent schema |
| `modules/raster_io.py` | **updated** — public `box_mean` export |
| `app.py` | **replaced** — raster-aware previews, honest status, real geometry export |
| `eval/run_benchmarks.py` | **new** — the scoring harness |
| `tests/test_core.py` | **new** — 50+ unit tests, no weights/GPU/network needed |
| `_deprecated_scripts/README.md` | **new** — documents the API migration |
| `quarantine_stale_scripts.ps1` | **new** — run once |

## Steps

1. Extract over `SatQuery-main/`, allowing overwrites.

2. Move the stale scripts aside. About 30 root-level `test_*.py` / `run_*.py`
   files and the old `tests/` suite call the removed API and will now fail:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\quarantine_stale_scripts.ps1
   ```

3. **Run the unit suite.** This is the first thing in this project that can be
   verified without model weights:

   ```powershell
   pytest tests -q
   ```

   It exercises the box-mean filter against a naive convolution, Otsu against a
   known bimodal distribution, radiometric normalisation against a known gain
   and offset, change detection against a synthetic scene with a change in a
   known quadrant, SAR water classification, composite aspect-ratio
   preservation, the offline router against all five of the problem statement's
   representative queries, and co-registration across all five statuses.

   Anything failing here is a real bug — none of this code has been executed, so
   send me the output.

4. Launch the app:

   ```powershell
   streamlit run app.py
   ```

   It now starts without a Gemini key. The sidebar shows which router is active.

5. Once the retrained adapters exist, score them:

   ```powershell
   python eval\run_benchmarks.py --task all --limit 200 --no-cache
   ```

## Behaviour changes worth knowing

**Co-registration.** PNG/JPEG pairs no longer report "verified". They report
"alignment not verifiable" and proceed. Only coordinate-checked pairs say
verified. Real Cartosat/RISAT pairs that the old 1e-4 tolerance rejected now
pass, and mismatched grids are reprojected rather than refused.

**Optical–SAR now runs.** It previously raised `TypeError` on every invocation
from the app, because the controller passed three arguments to a four-argument
function. It also does real work now: SAR backscatter percentile classification
for water and built-up, cross-checked against optical, alongside the VLM.

**Change analysis runs two tools.** A deterministic spatial pass answering
"where and how much", then the VLM for "what". Their outputs are merged.

**The trace is true.** No "Dual-Branch Optical-SAR Fusion Network", no
"ChangeFormer". Each step reports its real model id and a
`remote_sensing_adapted` flag — which is `false` for GroundingDINO, because it
is still stock. That gap is now visible rather than hidden.

**Confidence carries its type.** No more merging a VLM sequence likelihood with
a detector box score, and no hardcoded 85% fallback. The UI shows what the
number means.

**Cache is visible.** Every result carries `cache_hit`, the UI says when an
answer came from cache, and `SATQUERY_DISABLE_CACHE=1` forces live inference.
`run_benchmarks.py --no-cache` sets it automatically.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | unset | Cloud router. Absent → offline router, no error. |
| `SATQUERY_GEMINI_MODEL` | `gemini-2.5-flash-lite` | Change if that id does not resolve. |
| `SATQUERY_VQA_ADAPTER` | `modules/satquery_vqa_adapter` | Local path or HF repo id. |
| `SATQUERY_CHANGE_ADAPTER` | `modules/satquery_change_adapter` | Local path or HF repo id. |
| `SATQUERY_DEVICE` | auto | `cpu` or `cuda`. |
| `SATQUERY_DISABLE_CACHE` | unset | `1` forces live inference. |

Once the adapters are on the Hub, point the two adapter variables at the repo
ids and the app pulls weights itself.

## Still open

**GroundingDINO is not remote-sensing adapted.** The problem statement says a
generic model without adaptation will not satisfy the requirements. The trace is
honest about it, but it needs fine-tuning on VRSBench referring expressions —
`data_prep.py` already emits `grounding_eval.jsonl` with the boxes, and
`run_benchmarks.py --task grounding` gives you the pre-adaptation baseline to
improve on.
