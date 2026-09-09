# SatQuery AI — update pack 3: everything remaining

Extract over the root of `SatQuery-main`. This closes every remaining gap
against the problem statement.

## New files

| Path | What |
|---|---|
| `training/prep_bigearthnet.py` | **BigEarthNet-MM builder — the single most important addition.** Real co-registered Sentinel-1 SAR + Sentinel-2 optical. |
| `training/prep_rsvqa.py` | RSVQA-LR/HR builder. RSVQA is a prescribed benchmark and had no code path at all. |
| `modules/report.py` | Self-contained HTML report (images embedded, opens offline, prints to PDF). |
| `README.md` | Architecture, PS requirement mapping, setup, pipeline, design notes, known limitations. |

## Modified files

| Path | Change |
|---|---|
| `modules/composite.py` | Box↔canvas transforms and PaliGemma `<loc>` token encode/decode. |
| `modules/model_registry.py` | Fusion + grounding adapters; `run_grounding_vlm`; grounding backend dispatch; fusion adapter selection. |
| `training/train_vlm.py` | Two new tasks: `fusion` and `grounding`. Grounding IoU eval. |
| `data_prep.py` | Grounding **train** split, box parsing from conversations, coordinate-convention detection. |
| `eval/run_benchmarks.py` | RSVQA (with per-question-type breakdown) and fusion tasks; grounding coordinate-unit handling; backend reporting. |
| `app.py` | HTML report download button. |
| `tests/test_core.py` | 10 more tests covering box geometry and location tokens. |

## Steps

```powershell
pytest tests -q
```

Then build the two new datasets (`--inspect` first — it dumps the real schema
and costs a minute):

```powershell
python training\prep_bigearthnet.py --inspect
python training\prep_bigearthnet.py --max-patches 12000
```

```powershell
python training\prep_rsvqa.py --inspect
python training\prep_rsvqa.py --variant LR
```

Then two more Colab runs:

```bash
python training/train_vlm.py --task fusion \
  --data-dir data/bigearthnet --out modules/satquery_fusion_adapter \
  --max-steps 900 --save-steps 100 --resume \
  --hub-repo YOUR_HF_USER/satquery-fusion-adapter

python training/train_vlm.py --task grounding \
  --data-dir data/vrsbench --out modules/satquery_grounding_adapter \
  --max-steps 900 --save-steps 100 --resume \
  --hub-repo YOUR_HF_USER/satquery-grounding-adapter
```

Then score everything:

```powershell
python eval\run_benchmarks.py --task all --limit 200 --no-cache
```

## Why the fusion adapter matters most

Your VQA adapter is trained on VRSBench — optical RGB. It has never seen a SAR
image. SAR is speckled, decibel-scaled and semantically inverted: water is
*dark* (specular scattering), buildings are *bright* (double-bounce). The
ISRO/SAC evaluation set is Cartosat-2S optical paired with RISAT SAR, so half of
it is a modality the model cannot currently read.

BigEarthNet-MM is the fix and is the dataset the problem statement names as
primary. `prep_bigearthnet.py` renders its SAR through the exact same
dB → Lee filter → percentile stretch path that `raster_io` uses at inference, so
the adapter trains on the rendering it will actually be served.

Until that adapter exists, `run_optical_sar` falls back to the VQA adapter and
**says so in the trace**, with a warning that radar-derived claims are
unreliable. It does not quietly pretend.

## Grounding is now remote-sensing adapted

Rather than fine-tuning GroundingDINO — awkward on a free T4 — this fine-tunes
PaliGemma's native `detect` task, which emits `<loc0000>`–`<loc1023>` tokens.
Same training script, same pipeline, and it satisfies the PS requirement that
specialist components be domain-adapted.

At inference `run_grounding` dispatches: the adapted VLM if its adapter is
loaded, stock GroundingDINO otherwise. The trace records which backend ran under
`applied_parameters.backend`, and the benchmark reports `backends_used` — so a
fallback can never be mistaken for the adapted path.

One subtlety handled: boxes are annotated in the source image's pixel space, but
the model sees a letterboxed square canvas. `box_to_canvas` / `box_from_canvas`
apply the transform in both directions. Skipping it would train on boxes that do
not sit on the objects and produce near-zero IoU with no visible error.

## Coordinate conventions

VRSBench referring boxes may be absolute pixels, per-mille (0–999), percent, or
normalised. `data_prep.py` detects which and records it in `build_report.json`;
training and evaluation both apply the divisor. **Check that field after
building** — a wrong guess yields near-zero IoU that reads as a broken model
rather than a unit mismatch.

## Still untested

Nothing in any of the three packs has been executed — there is no Python on the
machine these were written on. `pytest tests -q` is the fastest way to find real
bugs; send me the output.

The two `--inspect` commands resolve the remaining unknowns: the exact
BigEarthNet-MM column layout and the RSVQA file naming. Both builders auto-detect
and abort with a schema dump rather than guessing.

## After this

What remains is not code:

1. Verify the VQA adapter's `eval` block (`collapsed` must be `false`).
2. Finish the change adapter, train fusion + grounding.
3. Push all four adapters to the Hub.
4. Run the benchmarks and record real numbers.
5. Test on one real Sentinel-2 GeoTIFF and one real SAR GeoTIFF.
6. Build the demo pack.
