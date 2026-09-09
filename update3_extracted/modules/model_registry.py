"""
Specialist model / tool registry for SatQuery AI.

Every tool here reports what it actually is. The previous version advertised a
"Dual-Branch Optical-SAR Fusion Network" and "ChangeFormer (Active)" in the
auditable execution summary; neither existed anywhere in the codebase, and the
"permitted parameters" shown alongside them (fusion_strategy=cross_attention,
sar_decibel_norm, difference_threshold, temperature) were never read by any
function. In a system whose whole point is evidence-grounded, auditable output,
a fabricated audit trail is the most damaging possible defect.

So: `tool_descriptors()` returns real model identifiers, and every result
carries `applied_parameters` containing the values that were actually used -
not the values that were requested. Each descriptor also carries
`remote_sensing_adapted`, which is honest about the fact that GroundingDINO is
still stock and has not been fine-tuned on overhead imagery.

All pixel access goes through `modules.raster_io`, so multispectral GeoTIFF and
SAR inputs work. All composites and prompts come from `modules.composite`, the
same module the training scripts import.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

from modules.change_detection import detect_change
from modules.composite import (
    box_from_canvas,
    build_caption_prompt,
    build_change_prompt,
    build_detect_prompt,
    build_fusion_prompt,
    build_vqa_prompt,
    decode_loc_tokens,
    make_pair_composite,
    make_single_image,
)
from modules.raster_io import RasterReadError, align_pair, load_as_rgb
from modules.sar_analysis import analyse_optical_sar

BASE_VLM_ID = os.environ.get("SATQUERY_BASE_VLM", "google/paligemma-3b-pt-224")
GROUNDING_ID = os.environ.get("SATQUERY_GROUNDING_MODEL", "IDEA-Research/grounding-dino-tiny")

VQA_ADAPTER = os.environ.get("SATQUERY_VQA_ADAPTER", "modules/satquery_vqa_adapter")
CHANGE_ADAPTER = os.environ.get("SATQUERY_CHANGE_ADAPTER", "modules/satquery_change_adapter")
FUSION_ADAPTER = os.environ.get("SATQUERY_FUSION_ADAPTER", "modules/satquery_fusion_adapter")
GROUNDING_ADAPTER = os.environ.get("SATQUERY_GROUNDING_ADAPTER", "modules/satquery_grounding_adapter")

CACHE_FILE = os.environ.get("SATQUERY_CACHE_FILE", "vqa_cache.json")
CACHE_DISABLED = os.environ.get("SATQUERY_DISABLE_CACHE", "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ToolDescriptor:
    """What a tool really is. Surfaced verbatim in the execution trace."""

    key: str
    display_name: str
    model_id: str
    kind: str                          # "vlm" | "detector" | "algorithm"
    remote_sensing_adapted: bool
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    answer: str
    confidence: float
    # Confidence is not one quantity. A VLM sequence likelihood and a detector
    # box score are different things on different scales; merging them into one
    # "Model Confidence Score %" - as the previous UI did, alongside a
    # hardcoded 85.0 fallback - is misleading. The type travels with the value.
    confidence_type: str
    tool: ToolDescriptor
    applied_parameters: Dict[str, Any] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    images: Dict[str, Image.Image] = field(default_factory=dict)
    cache_hit: bool = False
    latency_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """Serialisable view. PIL images are named, not embedded."""
        return {
            "answer": self.answer,
            "confidence": round(self.confidence, 4),
            "confidence_type": self.confidence_type,
            "tool": self.tool.as_dict(),
            "applied_parameters": self.applied_parameters,
            "preprocessing": self.preprocessing,
            "evidence": self.evidence,
            "visual_evidence": sorted(self.images.keys()),
            "cache_hit": self.cache_hit,
            "latency_ms": round(self.latency_ms, 1),
            "warnings": self.warnings,
            "error": self.error,
        }


CONFIDENCE_TYPES = {
    "sequence_likelihood": (
        "exp(mean per-token log probability) of the generated answer. This "
        "measures how confident the decoder was in its own wording - it is a "
        "fluency measure, NOT a probability that the answer is correct."
    ),
    "detector_score": (
        "GroundingDINO objectness/phrase-matching score for the returned box, "
        "in [0, 1]. Comparable across boxes, not comparable with the VLM score."
    ),
    "otsu_separability": (
        "between-class variance / total variance of the change-magnitude "
        "histogram. High means the changed and unchanged populations are "
        "cleanly separated."
    ),
    "deterministic": "Algorithmic output with no probabilistic component.",
    "none": "No confidence could be computed.",
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ModelRegistry:
    def __init__(self) -> None:
        self.device, self.dtype = self._pick_device_dtype()

        self.cache_file = CACHE_FILE
        self.cache_enabled = not CACHE_DISABLED
        self.cache: Dict[str, Any] = {}
        if self.cache_enabled and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

        self.vlm_processor = None
        self.vlm_model = None
        self.vlm_load_error: Optional[str] = None
        self.loaded_adapters: List[str] = []

        self.grounding_processor = None
        self.grounding_model = None
        self.grounding_load_error: Optional[str] = None

        print(
            f"ModelRegistry initialised (device={self.device}, dtype={self.dtype}, "
            f"cache={'on' if self.cache_enabled else 'OFF'}, "
            f"{len(self.cache)} cached entries)"
        )

    # -- environment -------------------------------------------------------

    @staticmethod
    def _pick_device_dtype() -> Tuple[str, torch.dtype]:
        forced = os.environ.get("SATQUERY_DEVICE")
        if forced:
            device = forced
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        if device == "cuda":
            # T4 and older have no bf16; fp16 is safe everywhere CUDA is.
            dtype = torch.float16
        else:
            # bfloat16 halves the 3B model's CPU footprint to ~6 GB, which is
            # what keeps it inside a 16 GB machine.
            dtype = torch.bfloat16
        return device, dtype

    # -- descriptors -------------------------------------------------------

    def tool_descriptors(self) -> Dict[str, ToolDescriptor]:
        adapters = ", ".join(self.loaded_adapters) if self.loaded_adapters else "none loaded"
        return {
            "vqa": ToolDescriptor(
                key="vqa",
                display_name="Remote-sensing VQA (PaliGemma + LoRA)",
                model_id=f"{BASE_VLM_ID} + LoRA:{VQA_ADAPTER}",
                kind="vlm",
                remote_sensing_adapted=True,
                notes=f"LoRA fine-tuned on VRSBench. Adapters present: {adapters}.",
            ),
            "caption": ToolDescriptor(
                key="caption",
                display_name="Remote-sensing scene description (PaliGemma + LoRA)",
                model_id=f"{BASE_VLM_ID} + LoRA:{VQA_ADAPTER}",
                kind="vlm",
                remote_sensing_adapted=True,
                notes="Same adapter as VQA, invoked with PaliGemma's 'caption en' prefix.",
            ),
            "grounding": ToolDescriptor(
                key="grounding",
                display_name="Text-guided region grounding (GroundingDINO)",
                model_id=GROUNDING_ID,
                kind="detector",
                remote_sensing_adapted=False,
                notes=(
                    "Stock zero-shot detector, NOT fine-tuned on overhead imagery. "
                    "Trained on ground-level natural images, so nadir satellite "
                    "performance is materially weaker than its benchmark numbers. "
                    "Used only as a fallback when the RS-adapted grounding adapter "
                    "is unavailable."
                ),
            ),
            "grounding_vlm": ToolDescriptor(
                key="grounding_vlm",
                display_name="Text-guided region grounding (PaliGemma detect + LoRA)",
                model_id=f"{BASE_VLM_ID} + LoRA:{GROUNDING_ADAPTER}",
                kind="vlm",
                remote_sensing_adapted=True,
                notes=(
                    "PaliGemma's native <loc> detection head, LoRA fine-tuned on "
                    "VRSBench referring expressions. Adapting this rather than "
                    "fine-tuning a separate detector reuses the existing training "
                    "pipeline and fits a free-tier T4."
                ),
            ),
            "fusion_vlm": ToolDescriptor(
                key="fusion_vlm",
                display_name="Optical-SAR joint interpretation (PaliGemma + BigEarthNet-MM LoRA)",
                model_id=f"{BASE_VLM_ID} + LoRA:{FUSION_ADAPTER}",
                kind="vlm",
                remote_sensing_adapted=True,
                notes=(
                    "LoRA fine-tuned on BigEarthNet-MM: real co-registered "
                    "Sentinel-1 SAR and Sentinel-2 optical. This is the only "
                    "component that has been trained on genuine radar imagery."
                ),
            ),
            "change_vqa": ToolDescriptor(
                key="change_vqa",
                display_name="Bi-temporal change VQA (PaliGemma + LoRA)",
                model_id=f"{BASE_VLM_ID} + LoRA:{CHANGE_ADAPTER}",
                kind="vlm",
                remote_sensing_adapted=True,
                notes="LoRA fine-tuned on CDVQA over horizontal T1|T2 composites.",
            ),
            "change_map": ToolDescriptor(
                key="change_map",
                display_name="Spatial change map (CVA + Otsu)",
                model_id="modules.change_detection: relative radiometric normalisation -> CVA -> Otsu -> morphology",
                kind="algorithm",
                remote_sensing_adapted=True,
                notes="Deterministic. No learned weights, nothing to fine-tune.",
            ),
            "optical_sar_vlm": ToolDescriptor(
                key="optical_sar_vlm",
                display_name="Optical-SAR joint interpretation (PaliGemma + LoRA)",
                model_id=f"{BASE_VLM_ID} + LoRA:{VQA_ADAPTER}",
                kind="vlm",
                remote_sensing_adapted=True,
                notes="Reads a horizontal optical|SAR composite.",
            ),
            "sar_evidence": ToolDescriptor(
                key="sar_evidence",
                display_name="SAR backscatter surface classification",
                model_id="modules.sar_analysis: relative backscatter percentile classification",
                kind="algorithm",
                remote_sensing_adapted=True,
                notes=(
                    "Water from specular (dark) returns, built-up from bright "
                    "high-texture double-bounce, cross-checked against optical."
                ),
            ),
        }

    # -- lazy loading ------------------------------------------------------

    def _ensure_vlm_loaded(self) -> bool:
        if self.vlm_model is not None:
            return True
        if self.vlm_load_error is not None:
            return False

        try:
            from peft import PeftModel
            from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

            processor_src = VQA_ADAPTER if os.path.isdir(VQA_ADAPTER) else BASE_VLM_ID
            print(f"Loading VLM: {BASE_VLM_ID} ({self.dtype}) on {self.device} ...")
            self.vlm_processor = AutoProcessor.from_pretrained(processor_src)

            base = PaliGemmaForConditionalGeneration.from_pretrained(
                BASE_VLM_ID,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)

            model = PeftModel.from_pretrained(base, VQA_ADAPTER, adapter_name="vqa")
            self.loaded_adapters = ["vqa"]

            # Optional task adapters. A missing one is not fatal: the tool falls
            # back to the VQA adapter and records the substitution in the trace,
            # rather than failing or silently pretending it used the right one.
            for path, name in (
                (CHANGE_ADAPTER, "change"),
                (FUSION_ADAPTER, "fusion"),
                (GROUNDING_ADAPTER, "ground"),
            ):
                try:
                    model.load_adapter(path, adapter_name=name)
                    self.loaded_adapters.append(name)
                except Exception as exc:
                    print(f"  '{name}' adapter unavailable ({type(exc).__name__}); falling back to 'vqa'")

            model.eval()
            self.vlm_model = model
            print(f"  VLM ready. Adapters: {self.loaded_adapters}")
            return True

        except Exception as exc:
            self.vlm_load_error = (
                f"{type(exc).__name__}: {exc}. Check that {VQA_ADAPTER} contains "
                f"adapter_model.safetensors - .gitignore excludes *.safetensors, so "
                f"a fresh clone will not have the weights. Pull them from the Hub."
            )
            print(f"VLM load FAILED: {self.vlm_load_error}")
            return False

    def _ensure_grounding_loaded(self) -> bool:
        if self.grounding_model is not None:
            return True
        if self.grounding_load_error is not None:
            return False
        try:
            from transformers import AutoProcessor, GroundingDinoForObjectDetection

            print(f"Loading detector: {GROUNDING_ID} (float32, {self.device}) ...")
            self.grounding_processor = AutoProcessor.from_pretrained(GROUNDING_ID)
            self.grounding_model = (
                GroundingDinoForObjectDetection.from_pretrained(GROUNDING_ID)
                .to(self.device)
                .eval()
                .float()
            )
            return True
        except Exception as exc:
            self.grounding_load_error = f"{type(exc).__name__}: {exc}"
            print(f"Detector load FAILED: {self.grounding_load_error}")
            return False

    # -- helpers -----------------------------------------------------------

    @property
    def image_size(self) -> int:
        """Model input resolution, read from the processor rather than assumed."""
        if self.vlm_processor is None:
            return 224
        size = self.vlm_processor.image_processor.size
        return int(size.get("height") or size.get("shortest_edge") or 224)

    def _cache_key(self, paths: Sequence[str], prompt: str, extra: str = "") -> str:
        h = hashlib.sha256()
        for p in paths:
            try:
                with open(p, "rb") as f:
                    h.update(f.read())
            except Exception:
                h.update(str(p).encode("utf-8"))
        h.update(prompt.encode("utf-8"))
        h.update(extra.encode("utf-8"))
        return h.hexdigest()

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.cache_enabled:
            return None
        return self.cache.get(key)

    def _cache_put(self, key: str, payload: Dict[str, Any]) -> None:
        if not self.cache_enabled:
            return
        self.cache[key] = payload
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f)
        except Exception as exc:
            print(f"  cache write failed (non-fatal): {exc}")

    def _error(self, tool_key: str, message: str, started: float) -> ToolResult:
        return ToolResult(
            answer=message,
            confidence=0.0,
            confidence_type="none",
            tool=self.tool_descriptors()[tool_key],
            latency_ms=(time.time() - started) * 1000.0,
            error=message,
        )

    @torch.no_grad()
    def _generate(
        self,
        prompt: str,
        image: Image.Image,
        adapter: str,
        max_new_tokens: int,
        num_beams: int,
    ) -> Tuple[str, float]:
        """Greedy/beam decode with a sequence-likelihood confidence."""
        available = self.loaded_adapters or ["vqa"]
        chosen = adapter if adapter in available else available[0]
        self.vlm_model.set_adapter(chosen)

        inputs = self.vlm_processor(text=prompt, images=image, return_tensors="pt")
        moved: Dict[str, torch.Tensor] = {}
        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                continue
            # Only pixel_values becomes a float tensor. The previous code called
            # .to(torch.bfloat16) on the whole batch and then cast input_ids and
            # attention_mask back - leaving token_type_ids as bfloat16, which
            # PaliGemma needs as integers to mask the prompt.
            moved[k] = v.to(self.device, self.dtype) if k == "pixel_values" else v.to(self.device)

        out = self.vlm_model.generate(
            **moved,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

        prompt_len = moved["input_ids"].shape[1]
        tokens = out.sequences[0][prompt_len:]
        answer = self.vlm_processor.decode(tokens, skip_special_tokens=True).strip()

        confidence = self._sequence_likelihood(out, tokens)
        return answer, confidence

    def _sequence_likelihood(self, out, tokens: torch.Tensor) -> float:
        """exp(mean log p) over generated tokens, excluding padding after EOS."""
        scores = getattr(out, "scores", None)
        if not scores or tokens.numel() == 0:
            return 0.0

        eos_id = getattr(self.vlm_processor.tokenizer, "eos_token_id", None)
        logprobs: List[float] = []
        for step, step_scores in enumerate(scores):
            if step >= tokens.shape[0]:
                break
            token_id = tokens[step]
            if eos_id is not None and int(token_id) == int(eos_id):
                break
            probs = torch.softmax(step_scores[0].float(), dim=-1)
            p = float(probs[token_id].clamp_min(1e-12))
            logprobs.append(math.log(p))

        if not logprobs:
            return 0.0
        return float(math.exp(sum(logprobs) / len(logprobs)))

    def _load(self, path: str, modality: str) -> Tuple[Image.Image, Dict[str, Any]]:
        img, report = load_as_rgb(path, modality=modality)
        return img, report.as_dict()

    # ------------------------------------------------------------------
    # Single-image tools
    # ------------------------------------------------------------------

    def run_single_vqa(
        self, image_path: str, query: str, params: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        started = time.time()
        params = params or {}
        max_new_tokens = int(params.get("max_new_tokens", 64))
        num_beams = int(params.get("num_beams", 1))
        modality = str(params.get("modality", "auto"))

        prompt = build_vqa_prompt(query)
        key = self._cache_key([image_path], prompt, f"vqa|{max_new_tokens}|{num_beams}|{modality}")
        cached = self._cache_get(key)
        if cached:
            return ToolResult(
                answer=cached["answer"],
                confidence=cached["confidence"],
                confidence_type="sequence_likelihood",
                tool=self.tool_descriptors()["vqa"],
                applied_parameters=cached.get("applied_parameters", {}),
                preprocessing=cached.get("preprocessing", {}),
                cache_hit=True,
                latency_ms=(time.time() - started) * 1000.0,
            )

        if not self._ensure_vlm_loaded():
            return self._error("vqa", f"VLM unavailable. {self.vlm_load_error}", started)

        try:
            raw, report = self._load(image_path, modality)
        except RasterReadError as exc:
            return self._error("vqa", f"Could not read image: {exc}", started)

        image = make_single_image(raw, size=self.image_size)
        answer, confidence = self._generate(prompt, image, "vqa", max_new_tokens, num_beams)

        applied = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "do_sample": False,
            "adapter": "vqa",
            "input_resolution": self.image_size,
            "prompt": prompt,
        }
        preprocessing = {"image": report}

        self._cache_put(
            key,
            {
                "answer": answer,
                "confidence": confidence,
                "applied_parameters": applied,
                "preprocessing": preprocessing,
            },
        )

        return ToolResult(
            answer=answer,
            confidence=confidence,
            confidence_type="sequence_likelihood",
            tool=self.tool_descriptors()["vqa"],
            applied_parameters=applied,
            preprocessing=preprocessing,
            images={"input": image},
            latency_ms=(time.time() - started) * 1000.0,
        )

    def run_caption(
        self, image_path: str, params: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        started = time.time()
        params = params or {}
        max_new_tokens = int(params.get("max_new_tokens", 96))
        num_beams = int(params.get("num_beams", 3))
        modality = str(params.get("modality", "auto"))

        prompt = build_caption_prompt()
        key = self._cache_key([image_path], prompt, f"cap|{max_new_tokens}|{num_beams}|{modality}")
        cached = self._cache_get(key)
        if cached:
            return ToolResult(
                answer=cached["answer"],
                confidence=cached["confidence"],
                confidence_type="sequence_likelihood",
                tool=self.tool_descriptors()["caption"],
                applied_parameters=cached.get("applied_parameters", {}),
                preprocessing=cached.get("preprocessing", {}),
                cache_hit=True,
                latency_ms=(time.time() - started) * 1000.0,
            )

        if not self._ensure_vlm_loaded():
            return self._error("caption", f"VLM unavailable. {self.vlm_load_error}", started)

        try:
            raw, report = self._load(image_path, modality)
        except RasterReadError as exc:
            return self._error("caption", f"Could not read image: {exc}", started)

        image = make_single_image(raw, size=self.image_size)
        answer, confidence = self._generate(prompt, image, "vqa", max_new_tokens, num_beams)

        applied = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "adapter": "vqa",
            "input_resolution": self.image_size,
            "prompt": prompt,
        }
        preprocessing = {"image": report}
        self._cache_put(
            key,
            {
                "answer": answer,
                "confidence": confidence,
                "applied_parameters": applied,
                "preprocessing": preprocessing,
            },
        )
        return ToolResult(
            answer=answer,
            confidence=confidence,
            confidence_type="sequence_likelihood",
            tool=self.tool_descriptors()["caption"],
            applied_parameters=applied,
            preprocessing=preprocessing,
            images={"input": image},
            latency_ms=(time.time() - started) * 1000.0,
        )

    def run_grounding(
        self,
        image_path: str,
        queries: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        Dispatch grounding to the remote-sensing-adapted VLM when its adapter is
        loaded, otherwise to the stock detector.

        `backend` may be "auto" (default), "vlm" or "detector". Whichever runs
        is named in the trace, so a fallback is never mistaken for the adapted
        path - which matters, because the problem statement requires domain
        adaptation and stock GroundingDINO does not provide it.
        """
        params = params or {}
        backend = str(params.get("backend", "auto")).lower()

        if backend == "vlm" or (
            backend == "auto"
            and (self.vlm_model is not None or os.path.isdir(GROUNDING_ADAPTER))
            and self._grounding_adapter_available()
        ):
            return self.run_grounding_vlm(image_path, queries, params)
        return self.run_grounding_detector(image_path, queries, params)

    def _grounding_adapter_available(self) -> bool:
        if self.vlm_model is None:
            return os.path.isdir(GROUNDING_ADAPTER)
        return "ground" in self.loaded_adapters

    def run_grounding_vlm(
        self,
        image_path: str,
        queries: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        Region grounding via PaliGemma's native `detect` task.

        The model emits <locNNNN> tokens over the letterboxed square canvas it
        was shown, so boxes are mapped back through `box_from_canvas` into the
        source image's own pixel space before they leave this method. Skipping
        that inverse transform would place every box in the wrong spot by the
        letterbox offset.
        """
        started = time.time()
        params = params or {}
        max_new_tokens = int(params.get("max_new_tokens", 64))
        modality = str(params.get("modality", "auto"))
        max_area = float(params.get("max_box_area_fraction", 0.92))

        if not self._ensure_vlm_loaded():
            return self._error("grounding_vlm", f"VLM unavailable. {self.vlm_load_error}", started)

        try:
            raw, report = self._load(image_path, modality)
        except RasterReadError as exc:
            return self._error("grounding_vlm", f"Could not read image: {exc}", started)

        phrases: List[str] = (
            list(queries) if isinstance(queries, (list, tuple)) else [str(queries)]
        )
        phrases = [str(p).strip().rstrip(".") for p in phrases if str(p).strip()]
        if not phrases:
            return self._error("grounding_vlm", "No search phrase was supplied.", started)

        size = self.image_size
        canvas = make_single_image(raw, size=size)
        prompt = build_detect_prompt(phrases)

        adapter = "ground" if "ground" in self.loaded_adapters else "vqa"
        warnings: List[str] = []
        if adapter != "ground":
            warnings.append(
                "The remote-sensing grounding adapter is not loaded; the VQA "
                "adapter was used, which was not trained on the detection task."
            )

        answer_text, confidence = self._generate(prompt, canvas, adapter, max_new_tokens, 1)

        img_area = float(raw.size[0] * raw.size[1])
        detections: List[Dict[str, Any]] = []
        for canvas_box, label in decode_loc_tokens(answer_text, canvas_size=size):
            box = box_from_canvas(canvas_box, raw.size, size)
            area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            frac = area / img_area if img_area else 1.0
            if frac <= 0.0:
                continue
            if frac > max_area:
                warnings.append(
                    f"Rejected a box covering {frac:.0%} of the frame (limit "
                    f"{max_area:.0%}); a whole-image box is not a localisation."
                )
                continue
            detections.append(
                {
                    "box": [round(c, 2) for c in box],
                    "score": round(confidence, 4),
                    "label": label or phrases[0],
                    "phrase": label or phrases[0],
                    "area_fraction": round(frac, 4),
                }
            )

        applied = {
            "backend": "vlm",
            "adapter": adapter,
            "max_new_tokens": max_new_tokens,
            "max_box_area_fraction": max_area,
            "search_phrases": phrases,
            "input_resolution": size,
            "prompt": prompt,
            "raw_output": answer_text,
            "coordinate_mapping": "loc tokens -> canvas px -> source image px",
            "image_size": list(raw.size),
        }

        if not detections:
            return ToolResult(
                answer=f"No region matching {', '.join(phrases)} was localised.",
                confidence=0.0,
                confidence_type="sequence_likelihood",
                tool=self.tool_descriptors()["grounding_vlm"],
                applied_parameters=applied,
                preprocessing={"image": report},
                evidence={"detections": [], "detection_count": 0},
                images={"input": raw},
                latency_ms=(time.time() - started) * 1000.0,
                warnings=warnings,
            )

        best = detections[0]
        plural = "s" if len(detections) > 1 else ""
        answer = (
            f"Localised {len(detections)} region{plural} matching '{best['label']}'. "
            f"Primary detection at [{best['box'][0]:.0f}, {best['box'][1]:.0f}, "
            f"{best['box'][2]:.0f}, {best['box'][3]:.0f}]."
        )

        return ToolResult(
            answer=answer,
            confidence=confidence,
            confidence_type="sequence_likelihood",
            tool=self.tool_descriptors()["grounding_vlm"],
            applied_parameters=applied,
            preprocessing={"image": report},
            evidence={
                "detections": detections,
                "detection_count": len(detections),
                "best": best,
            },
            images={"input": raw},
            latency_ms=(time.time() - started) * 1000.0,
            warnings=warnings,
        )

    def run_grounding_detector(
        self,
        image_path: str,
        queries: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        Text-guided region grounding via the stock zero-shot detector.

        Returns every box above threshold, not just the single best one - "the
        water bodies" is legitimately plural. The oversized-box filter is a
        declared parameter (`max_box_area_fraction`) instead of the previous
        hidden hardcoded whitelist of "scene-wide terms".
        """
        started = time.time()
        params = params or {}
        box_threshold = float(params.get("box_threshold", 0.25))
        text_threshold = float(params.get("text_threshold", 0.20))
        max_area = float(params.get("max_box_area_fraction", 0.92))
        top_k = int(params.get("top_k", 8))
        modality = str(params.get("modality", "auto"))

        if not self._ensure_grounding_loaded():
            return self._error("grounding", f"Detector unavailable. {self.grounding_load_error}", started)

        try:
            image, report = self._load(image_path, modality)
        except RasterReadError as exc:
            return self._error("grounding", f"Could not read image: {exc}", started)

        phrases: List[str] = (
            list(queries) if isinstance(queries, (list, tuple)) else [str(queries)]
        )
        phrases = [p if p.strip().endswith(".") else f"{p.strip()}." for p in phrases if str(p).strip()]
        if not phrases:
            return self._error("grounding", "No search phrase was supplied.", started)

        detections: List[Dict[str, Any]] = []
        warnings: List[str] = []
        img_area = float(image.size[0] * image.size[1])

        for phrase in phrases:
            inputs = self.grounding_processor(images=image, text=phrase, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.grounding_model(**inputs)

            # Only text_threshold and target_sizes are passed. The score
            # threshold keyword has been renamed across transformers releases
            # (box_threshold -> threshold), so filtering by score is done
            # explicitly below instead of relying on a keyword that may not
            # exist in the installed version.
            processed = self.grounding_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]

            labels = processed.get("text_labels", processed.get("labels", []))
            for score, box, label in zip(processed["scores"], processed["boxes"], labels):
                s = float(score)
                if s < box_threshold:
                    continue
                coords = [float(c) for c in box.tolist()]
                area = max(0.0, coords[2] - coords[0]) * max(0.0, coords[3] - coords[1])
                frac = area / img_area if img_area else 1.0
                if frac > max_area:
                    warnings.append(
                        f"Rejected a box covering {frac:.0%} of the frame for "
                        f"'{phrase}' (limit {max_area:.0%}); a whole-image box is "
                        f"not a localisation."
                    )
                    continue
                detections.append(
                    {
                        "box": coords,
                        "score": round(s, 4),
                        "label": str(label),
                        "phrase": phrase,
                        "area_fraction": round(frac, 4),
                    }
                )

        detections.sort(key=lambda d: d["score"], reverse=True)
        detections = detections[:top_k]

        applied = {
            "backend": "detector",
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "max_box_area_fraction": max_area,
            "top_k": top_k,
            "search_phrases": phrases,
            "image_size": list(image.size),
        }

        if not detections:
            return ToolResult(
                answer=(
                    f"No region matching {', '.join(phrases)} was detected above a "
                    f"score of {box_threshold:.2f}."
                ),
                confidence=0.0,
                confidence_type="detector_score",
                tool=self.tool_descriptors()["grounding"],
                applied_parameters=applied,
                preprocessing={"image": report},
                evidence={"detections": [], "detection_count": 0},
                images={"input": image},
                cache_hit=False,
                latency_ms=(time.time() - started) * 1000.0,
                warnings=warnings,
            )

        best = detections[0]
        plural = "s" if len(detections) > 1 else ""
        answer = (
            f"Localised {len(detections)} region{plural} matching "
            f"'{best['label']}'. Highest-scoring detection at "
            f"[{best['box'][0]:.0f}, {best['box'][1]:.0f}, {best['box'][2]:.0f}, "
            f"{best['box'][3]:.0f}] with a detector score of {best['score']:.2f}."
        )

        return ToolResult(
            answer=answer,
            confidence=best["score"],
            confidence_type="detector_score",
            tool=self.tool_descriptors()["grounding"],
            applied_parameters=applied,
            preprocessing={"image": report},
            evidence={
                "detections": detections,
                "detection_count": len(detections),
                "best": best,
            },
            images={"input": image},
            latency_ms=(time.time() - started) * 1000.0,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Bi-temporal tools
    # ------------------------------------------------------------------

    def run_change_vqa(
        self,
        t1_path: str,
        t2_path: str,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        started = time.time()
        params = params or {}
        max_new_tokens = int(params.get("max_new_tokens", 64))
        num_beams = int(params.get("num_beams", 1))

        prompt = build_change_prompt(query)
        key = self._cache_key([t1_path, t2_path], prompt, f"chg|{max_new_tokens}|{num_beams}")
        cached = self._cache_get(key)
        if cached:
            return ToolResult(
                answer=cached["answer"],
                confidence=cached["confidence"],
                confidence_type="sequence_likelihood",
                tool=self.tool_descriptors()["change_vqa"],
                applied_parameters=cached.get("applied_parameters", {}),
                preprocessing=cached.get("preprocessing", {}),
                cache_hit=True,
                latency_ms=(time.time() - started) * 1000.0,
            )

        if not self._ensure_vlm_loaded():
            return self._error("change_vqa", f"VLM unavailable. {self.vlm_load_error}", started)

        try:
            img1, img2, align = align_pair(t1_path, t2_path, "auto", "auto")
        except RasterReadError as exc:
            return self._error("change_vqa", f"Could not read image pair: {exc}", started)

        composite = make_pair_composite(img1, img2, size=self.image_size, layout="horizontal")

        warnings: List[str] = []
        adapter = "change"
        if "change" not in self.loaded_adapters:
            adapter = "vqa"
            warnings.append(
                "The dedicated change adapter is not loaded; the VQA adapter was "
                "used instead. Change-specific accuracy will be lower."
            )

        answer, confidence = self._generate(prompt, composite, adapter, max_new_tokens, num_beams)

        applied = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "adapter": adapter,
            "composite_layout": "horizontal (T1 left, T2 right)",
            "input_resolution": self.image_size,
            "alignment_method": align.get("method"),
            "prompt": prompt,
        }
        preprocessing = {"alignment": align}

        self._cache_put(
            key,
            {
                "answer": answer,
                "confidence": confidence,
                "applied_parameters": applied,
                "preprocessing": preprocessing,
            },
        )

        return ToolResult(
            answer=answer,
            confidence=confidence,
            confidence_type="sequence_likelihood",
            tool=self.tool_descriptors()["change_vqa"],
            applied_parameters=applied,
            preprocessing=preprocessing,
            images={"composite": composite, "t1": img1, "t2": img2},
            latency_ms=(time.time() - started) * 1000.0,
            warnings=warnings,
        )

    def run_change_map(
        self,
        t1_path: str,
        t2_path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        started = time.time()
        params = params or {}
        open_radius = int(params.get("open_radius", 1))
        close_radius = int(params.get("close_radius", 2))
        min_region_fraction = float(params.get("min_region_fraction", 0.0005))
        max_regions = int(params.get("max_regions", 8))

        try:
            img1, img2, align = align_pair(t1_path, t2_path, "auto", "auto")
        except RasterReadError as exc:
            return self._error("change_map", f"Could not read image pair: {exc}", started)

        result = detect_change(
            img1,
            img2,
            open_radius=open_radius,
            close_radius=close_radius,
            min_region_fraction=min_region_fraction,
            max_regions=max_regions,
        )

        applied = {
            "open_radius": open_radius,
            "close_radius": close_radius,
            "min_region_fraction": min_region_fraction,
            "max_regions": max_regions,
            "alignment_method": align.get("method"),
            **result.method,
        }

        return ToolResult(
            answer=result.summary,
            confidence=result.separability,
            confidence_type="otsu_separability",
            tool=self.tool_descriptors()["change_map"],
            applied_parameters=applied,
            preprocessing={"alignment": align},
            evidence=result.stats(),
            images={"change_overlay": result.overlay, "t1": img1, "t2": img2},
            latency_ms=(time.time() - started) * 1000.0,
            warnings=result.warnings,
        )

    # ------------------------------------------------------------------
    # Cross-modal tools
    # ------------------------------------------------------------------

    def run_optical_sar(
        self,
        optical_path: str,
        sar_path: str,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        Signature note: this takes FOUR arguments. The previous controller
        called it with three (`optical, sar, params`), so `params` bound to
        `query` and `params` was missing entirely - a guaranteed TypeError the
        moment the optical-SAR path ran from the app. The standalone test
        passed four arguments, which is why the crash was never observed.
        """
        started = time.time()
        params = params or {}
        max_new_tokens = int(params.get("max_new_tokens", 64))
        num_beams = int(params.get("num_beams", 1))

        prompt = build_fusion_prompt(query)
        key = self._cache_key([optical_path, sar_path], prompt, f"fus|{max_new_tokens}|{num_beams}")
        cached = self._cache_get(key)
        if cached:
            return ToolResult(
                answer=cached["answer"],
                confidence=cached["confidence"],
                confidence_type="sequence_likelihood",
                tool=self.tool_descriptors()["optical_sar_vlm"],
                applied_parameters=cached.get("applied_parameters", {}),
                preprocessing=cached.get("preprocessing", {}),
                cache_hit=True,
                latency_ms=(time.time() - started) * 1000.0,
            )

        if not self._ensure_vlm_loaded():
            return self._error("optical_sar_vlm", f"VLM unavailable. {self.vlm_load_error}", started)

        try:
            # Modality is declared, not guessed: the UI has separate upload
            # slots, so we know which file is which. That matters because SAR
            # needs dB conversion and speckle filtering and optical does not.
            opt, sar, align = align_pair(optical_path, sar_path, "optical", "sar")
        except RasterReadError as exc:
            return self._error("optical_sar_vlm", f"Could not read image pair: {exc}", started)

        composite = make_pair_composite(opt, sar, size=self.image_size, layout="horizontal")

        warnings: List[str] = []
        adapter = "fusion" if "fusion" in self.loaded_adapters else "vqa"
        if adapter != "fusion":
            warnings.append(
                "The BigEarthNet-MM fusion adapter is not loaded; the VQA adapter "
                "was used instead. That adapter was trained only on optical "
                "imagery and has not seen SAR, so treat radar-derived claims in "
                "this answer with caution."
            )

        answer, confidence = self._generate(prompt, composite, adapter, max_new_tokens, num_beams)

        applied = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "adapter": adapter,
            "composite_layout": "horizontal (optical left, SAR right)",
            "input_resolution": self.image_size,
            "sar_preprocessing": "decibel conversion + Lee speckle filter + percentile stretch",
            "alignment_method": align.get("method"),
            "prompt": prompt,
        }
        preprocessing = {"alignment": align}

        self._cache_put(
            key,
            {
                "answer": answer,
                "confidence": confidence,
                "applied_parameters": applied,
                "preprocessing": preprocessing,
            },
        )

        return ToolResult(
            answer=answer,
            confidence=confidence,
            confidence_type="sequence_likelihood",
            tool=self.tool_descriptors()[
                "fusion_vlm" if adapter == "fusion" else "optical_sar_vlm"
            ],
            applied_parameters=applied,
            preprocessing=preprocessing,
            images={"composite": composite, "optical": opt, "sar": sar},
            latency_ms=(time.time() - started) * 1000.0,
            warnings=warnings,
        )

    def run_sar_evidence(
        self,
        optical_path: str,
        sar_path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        started = time.time()
        params = params or {}
        water_pct = float(params.get("water_percentile", 15.0))
        builtup_pct = float(params.get("builtup_percentile", 90.0))
        texture_pct = float(params.get("texture_percentile", 55.0))

        try:
            opt, sar, align = align_pair(optical_path, sar_path, "optical", "sar")
        except RasterReadError as exc:
            return self._error("sar_evidence", f"Could not read image pair: {exc}", started)

        ev = analyse_optical_sar(
            opt, sar, water_pct=water_pct, builtup_pct=builtup_pct, texture_pct=texture_pct
        )

        applied = {
            "water_percentile": water_pct,
            "builtup_percentile": builtup_pct,
            "texture_percentile": texture_pct,
            "alignment_method": align.get("method"),
            **ev.method,
        }

        return ToolResult(
            answer=ev.summary,
            confidence=0.0,
            confidence_type="deterministic",
            tool=self.tool_descriptors()["sar_evidence"],
            applied_parameters=applied,
            preprocessing={"alignment": align},
            evidence=ev.stats(),
            images={"sar_overlay": ev.overlay, "optical": opt, "sar": sar},
            latency_ms=(time.time() - started) * 1000.0,
            warnings=ev.warnings,
        )


__all__ = ["ModelRegistry", "ToolResult", "ToolDescriptor", "CONFIDENCE_TYPES"]
