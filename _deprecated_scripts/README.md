# Quarantined scripts — do not run

These were ad-hoc debugging scripts written against the previous inference API.
They are kept as a record, not as working code. Every one of them will now fail
with `AttributeError` or `TypeError`, because the API they call no longer
exists.

## What changed underneath them

| Old API | New API |
|---|---|
| `registry.run_change_analysis(t1, t2, q, p)` | `registry.run_change_vqa(...)` plus `registry.run_change_map(...)` — two tools, sequenced by the controller |
| `registry.run_optical_sar(opt, sar, params)` (3 args) | `registry.run_optical_sar(opt, sar, query, params)` (4 args, as it always should have been) |
| returns `dict` with `answer` / `confidence` / `box` | returns `ToolResult` with `.answer`, `.confidence`, `.confidence_type`, `.tool`, `.applied_parameters`, `.preprocessing`, `.evidence`, `.images` |
| `registry.loaded_tools["vqa"]` → a display string | `registry.tool_descriptors()["vqa"]` → a `ToolDescriptor` with the real model id |
| `validator.verify_coregistration(a, b)` → `bool` | `validator.check_coregistration(a, b)` → `CoregistrationResult` with five distinct statuses. The boolean shim still exists but now returns `False` for unverifiable pairs instead of `True`. |
| `agent.route_and_configure(...)` → single-tool trace | still works (aliased to `agent.run`), but returns a multi-step `ExecutionTrace` |
| `PIL.Image.open(path)` for pixels | `modules.raster_io.load_as_rgb(path, modality=...)` |

## What replaces them

- **`tests/test_core.py`** — a real unit suite for the deterministic core
  (raster maths, change detection, SAR evidence, composites, routing,
  co-registration). Runs in seconds, needs no model weights, no GPU, no
  network. `pytest tests -q`.
- **`eval/run_benchmarks.py`** — scores the model-dependent paths on held-out
  splits with exact match, token F1, grounding IoU, and degeneracy detection.
  This is what `run_real_sanity*.py`, `test_vqa_real.py`,
  `test_multi_noun_evaluation.py` and friends were reaching for.

## `old_tests/`

The previous `tests/` directory. `test_agent.py` and
`test_agent_controller_e2e.py` mocked the Gemini call and asserted on the
old single-tool trace shape. `test_agent_controller_real_llm.py` required a
live API key, so it could not run in CI. `conftest.py` existed only to suppress
a transformers `FutureWarning`.

## `prewarm_cache.py`

Worth calling out separately. This pre-populated `vqa_cache.json` with the exact
images and queries used in demos, and its own comment noted change analysis took
"~5-6 minutes per pair" on CPU. Cached answers are legitimate engineering, but
presenting pre-computed results as live inference is not — and a judge handing
over a novel image would have hit the full six-minute latency on stage.

The cache still exists in the new registry, with two differences: every result
carries `cache_hit`, which the UI displays, and `SATQUERY_DISABLE_CACHE=1`
forces live inference. `eval/run_benchmarks.py --no-cache` sets it automatically,
because a cached benchmark run measures the cache rather than the model.

## Safe to delete

Once the rebuilt system is running, delete this folder. It is retained so the
API migration is documented rather than rediscovered.
