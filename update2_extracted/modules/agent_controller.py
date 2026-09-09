"""
Agentic controller for SatQuery AI.

Responsibilities, in the order the problem statement lists them:
  * interpret the query and classify the requested task
  * check the number, modality, format, metadata and compatibility of inputs
  * select one or more tools from the registry
  * configure only permitted parameters and execute the workflow
  * combine textual and spatial outputs, estimate confidence, return evidence
  * produce an auditable execution summary

Two things are different from the previous version, and both matter.

1.  Routing no longer depends on the network. The Gemini classifier is tried
    first; if there is no API key, no connectivity, a rate limit, or an
    unrecognised model id, the deterministic offline router in
    `modules.router` takes over. The trace always records which backend
    actually decided, and why the other one was not used. Previously the
    constructor raised `EnvironmentError` without a key - inside
    `@st.cache_resource`, so the app died at startup - and any API failure
    turned every query into "Classification Error".

2.  The execution summary is true. Tool names come from
    `ModelRegistry.tool_descriptors()` and are real model identifiers;
    parameters reported are the ones the tools actually consumed, returned
    back from each tool as `applied_parameters`. Nothing is invented.

Paired tasks now execute a genuine multi-tool sequence - a deterministic
spatial tool plus the vision-language model - and merge their outputs, which is
what "select, sequence and execute" is asking for.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from modules.geo_validator import SatQueryValidator
from modules.intent import (
    IntentClassification,
    InputValidationError,
    PAIR_TASKS,
    TASK_KEY_MAP,
    TASK_KEYS,
    TaskType,
)
from modules.model_registry import ModelRegistry, ToolDescriptor, ToolResult
from modules.router import LocalIntentRouter, extract_entities

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_MODEL = os.environ.get("SATQUERY_GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_TIMEOUT_RETRIES = 2


INTENT_SYSTEM_PROMPT = """\
You are the intent-routing component of SatQuery AI, an agentic vision-language
assistant for remote-sensing image analysis.

Classify the user's natural-language query into exactly ONE task:

- "vqa"                -> Single-image visual question answering: counting,
                          identifying, attributes, yes/no questions about one image.
- "caption"            -> Single-image scene description: open-ended "describe
                          the land-cover and major objects" style requests.
- "grounding"          -> Text-guided region grounding: locating, highlighting or
                          bounding-boxing a specific entity in one image.
- "change_analysis"    -> Bi-temporal change detection and change VQA over two
                          images of the same area at different times.
- "optical_sar_fusion" -> Joint interpretation of a co-registered optical and
                          SAR pair.

You are given image_count (1 or 2) and input_mode ("single", "bi_temporal",
"cross_modal").

Rules:
1. With 1 image, NEVER return "change_analysis" or "optical_sar_fusion".
2. input_mode "bi_temporal" strongly favours "change_analysis".
3. input_mode "cross_modal" strongly favours "optical_sar_fusion".
4. "grounding" requires an explicit request to locate/highlight/mark something.
5. For "grounding", search_nouns must be short noun phrases ending in a period,
   e.g. ["water body.", "road."].

Return only the structured JSON.
"""


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


@dataclass
class ExecutionStep:
    order: int
    purpose: str
    result: ToolResult

    def as_dict(self) -> Dict[str, Any]:
        d = self.result.as_dict()
        return {"order": self.order, "purpose": self.purpose, **d}


@dataclass
class ExecutionTrace:
    query: str
    input_mode: str
    input_count: int
    selected_task: TaskType = TaskType.UNKNOWN
    router: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    steps: List[ExecutionStep] = field(default_factory=list)
    answer: str = ""
    confidence: float = 0.0
    confidence_type: str = "none"
    status: str = "pending"
    total_latency_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)

    # -- convenience -------------------------------------------------------

    @property
    def primary_result(self) -> Optional[ToolResult]:
        return self.steps[-1].result if self.steps else None

    @property
    def selected_tools(self) -> List[ToolDescriptor]:
        return [s.result.tool for s in self.steps]

    @property
    def selected_tool(self) -> str:
        """Human-readable list of the tools that actually ran."""
        if not self.steps:
            return "None"
        return " -> ".join(s.result.tool.display_name for s in self.steps)

    def images(self) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for step in self.steps:
            for name, img in step.result.images.items():
                merged.setdefault(name, img)
        return merged

    def as_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "input": {"mode": self.input_mode, "image_count": self.input_count},
            "selected_task": self.selected_task.value,
            "router": self.router,
            "input_validation": self.validation,
            "execution_plan": [
                {
                    "order": s.order,
                    "purpose": s.purpose,
                    "tool": s.result.tool.display_name,
                    "model_id": s.result.tool.model_id,
                    "remote_sensing_adapted": s.result.tool.remote_sensing_adapted,
                }
                for s in self.steps
            ],
            "steps": [s.as_dict() for s in self.steps],
            "answer": self.answer,
            "confidence": round(self.confidence, 4),
            "confidence_type": self.confidence_type,
            "status": self.status,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class AgentController:
    def __init__(self, registry: Optional[ModelRegistry] = None) -> None:
        self.registry = registry or ModelRegistry()
        self.validator = SatQueryValidator()
        self.local_router = LocalIntentRouter()

        self.gemini_client = None
        self.gemini_unavailable_reason: Optional[str] = None

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            # Not fatal. The offline router handles everything.
            self.gemini_unavailable_reason = "GEMINI_API_KEY is not set"
        else:
            try:
                from google import genai

                self.gemini_client = genai.Client(api_key=api_key)
            except Exception as exc:
                self.gemini_unavailable_reason = f"{type(exc).__name__}: {exc}"

        if self.gemini_unavailable_reason:
            print(
                f"Agent: cloud classifier unavailable ({self.gemini_unavailable_reason}). "
                f"Using offline rule-based router."
            )

    # -- classification ----------------------------------------------------

    def _call_gemini(self, query: str, input_config: Dict[str, Any]) -> IntentClassification:
        from google.genai import types

        user_message = (
            f'User query: "{query}"\n'
            f"Input context: image_count={input_config.get('image_count')}, "
            f"input_mode={input_config.get('input_mode')}"
        )
        schema = IntentClassification.model_json_schema()

        last_exc: Optional[Exception] = None
        for attempt in range(GEMINI_TIMEOUT_RETRIES + 1):
            try:
                try:
                    config = types.GenerateContentConfig(
                        system_instruction=INTENT_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_json_schema=schema,
                        temperature=0.0,
                    )
                except TypeError:
                    # Older/newer SDKs name this field differently. Falling back
                    # to plain JSON output still works; the response is parsed
                    # and validated locally either way.
                    config = types.GenerateContentConfig(
                        system_instruction=INTENT_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        temperature=0.0,
                    )

                response = self.gemini_client.models.generate_content(
                    model=GEMINI_MODEL, contents=user_message, config=config
                )
                payload = json.loads(response.text)
                classification = IntentClassification.model_validate(payload)
                classification.backend = f"gemini:{GEMINI_MODEL}"
                classification.rationale = "cloud LLM structured-output classification"
                return classification

            except Exception as exc:
                last_exc = exc
                if "429" in str(exc) and attempt < GEMINI_TIMEOUT_RETRIES:
                    time.sleep(2 * (attempt + 1))
                    continue
                break

        raise RuntimeError(f"{type(last_exc).__name__}: {last_exc}")

    def classify(self, query: str, input_config: Dict[str, Any]) -> IntentClassification:
        """Cloud first, offline fallback. Always returns something usable."""
        started = time.time()
        fallback_reason = self.gemini_unavailable_reason

        if self.gemini_client is not None:
            try:
                cls = self._call_gemini(query, input_config)
                if cls.task not in TASK_KEYS:
                    raise ValueError(f"model returned unknown task {cls.task!r}")
                cls = self._repair(cls, query, input_config)
                cls.rationale = (
                    f"{cls.rationale} (latency {int((time.time() - started) * 1000)} ms)"
                )
                return cls
            except Exception as exc:
                fallback_reason = f"{type(exc).__name__}: {exc}"
                logger.warning("Cloud classifier failed, using offline router: %s", exc)

        cls = self.local_router.classify(query, input_config)
        cls = self._repair(cls, query, input_config)
        if fallback_reason:
            cls.rationale = f"{cls.rationale} | cloud fallback reason: {fallback_reason}"
        return cls

    @staticmethod
    def _repair(
        cls: IntentClassification, query: str, input_config: Dict[str, Any]
    ) -> IntentClassification:
        """
        Reconcile the classification with what was actually uploaded, rather
        than rejecting the query outright.

        The previous controller raised InputValidationError and returned an
        empty trace whenever the classifier picked a two-image task for one
        image. Downgrading to the nearest viable single-image task and saying
        so is more useful and just as auditable.
        """
        image_count = int(input_config.get("image_count", 0) or 0)
        notes: List[str] = []

        if cls.task in PAIR_TASKS and image_count < 2:
            cls.task = "vqa"
            notes.append(
                f"downgraded to single-image VQA: the query implies a paired task "
                f"but {image_count} image(s) were supplied"
            )

        if cls.task == "grounding" and not cls.search_nouns:
            entities = extract_entities(query)
            if entities:
                cls.search_nouns = [f"{e}." for e in entities]
                cls.target_entity = cls.target_entity or entities[0]
                notes.append("search_nouns derived locally from the query")
            else:
                cls.task = "vqa"
                notes.append(
                    "downgraded to VQA: grounding was requested but no locatable "
                    "entity could be identified in the query"
                )

        if notes:
            cls.rationale = "; ".join(filter(None, [cls.rationale, *notes]))
        return cls

    # -- validation --------------------------------------------------------

    def _validate_inputs(
        self, task: str, image_paths: List[str]
    ) -> Dict[str, Any]:
        if not image_paths:
            raise InputValidationError("No image was supplied.")

        report: Dict[str, Any] = {
            "images": [self.validator.extract_metadata(p).as_dict() for p in image_paths]
        }

        for meta in report["images"]:
            if not meta["is_valid"]:
                raise InputValidationError(meta["error"] or "unreadable input")

        if task in PAIR_TASKS:
            if len(image_paths) < 2:
                raise InputValidationError(
                    f"Task '{task}' needs two images; {len(image_paths)} supplied."
                )
            coreg = self.validator.check_coregistration(image_paths[0], image_paths[1])
            report["coregistration"] = coreg.as_dict()
            if not coreg.ok_to_proceed:
                raise InputValidationError(coreg.message)

        return report

    # -- execution ---------------------------------------------------------

    def run(
        self, query: str, image_paths: List[str], input_mode: str = "single"
    ) -> ExecutionTrace:
        started = time.time()
        trace = ExecutionTrace(
            query=query, input_mode=input_mode, input_count=len(image_paths)
        )

        input_config = {"image_count": len(image_paths), "input_mode": input_mode}

        cls = self.classify(query, input_config)
        trace.router = {
            "backend": cls.backend,
            "task_key": cls.task,
            "classification_confidence": cls.classifier_self_reported_confidence,
            "target_entity": cls.target_entity,
            "search_nouns": cls.search_nouns,
            "rationale": cls.rationale,
        }
        trace.selected_task = TASK_KEY_MAP.get(cls.task, TaskType.UNKNOWN)

        try:
            trace.validation = self._validate_inputs(cls.task, image_paths)
        except InputValidationError as exc:
            trace.status = "input_validation_failed"
            trace.answer = str(exc)
            trace.total_latency_ms = (time.time() - started) * 1000.0
            return trace

        coreg = trace.validation.get("coregistration")
        if coreg and not coreg.get("verified"):
            trace.warnings.append(coreg["message"])

        try:
            self._execute(trace, cls, image_paths, query)
            trace.status = "executed"
        except Exception as exc:
            logger.exception("Execution failed")
            trace.status = "execution_error"
            trace.answer = f"Execution failed: {type(exc).__name__}: {exc}"

        for step in trace.steps:
            trace.warnings.extend(step.result.warnings)
            if step.result.error:
                trace.status = "tool_error"

        trace.total_latency_ms = (time.time() - started) * 1000.0
        return trace

    # Legacy name kept so existing callers and tests keep working.
    route_and_configure = run

    def _add(self, trace: ExecutionTrace, purpose: str, result: ToolResult) -> ToolResult:
        trace.steps.append(ExecutionStep(len(trace.steps) + 1, purpose, result))
        return result

    def _execute(
        self,
        trace: ExecutionTrace,
        cls: IntentClassification,
        paths: List[str],
        query: str,
    ) -> None:
        task = cls.task
        reg = self.registry

        if task == "vqa":
            r = self._add(
                trace,
                "Answer the question from the single image",
                reg.run_single_vqa(paths[0], query, {"max_new_tokens": 64, "num_beams": 1}),
            )
            trace.answer = r.answer
            trace.confidence = r.confidence
            trace.confidence_type = r.confidence_type

        elif task == "caption":
            r = self._add(
                trace,
                "Describe the scene",
                reg.run_caption(paths[0], {"max_new_tokens": 96, "num_beams": 3}),
            )
            trace.answer = r.answer
            trace.confidence = r.confidence
            trace.confidence_type = r.confidence_type

        elif task == "grounding":
            r = self._add(
                trace,
                f"Localise: {', '.join(cls.search_nouns or [])}",
                reg.run_grounding(
                    paths[0],
                    cls.search_nouns,
                    {
                        "box_threshold": 0.25,
                        "text_threshold": 0.20,
                        "max_box_area_fraction": 0.92,
                        "top_k": 8,
                    },
                ),
            )
            trace.answer = r.answer
            trace.confidence = r.confidence
            trace.confidence_type = r.confidence_type

        elif task == "change_analysis":
            # Two tools in sequence: a deterministic spatial pass that answers
            # "where and how much", then the VLM for "what". Neither alone
            # answers the problem statement's representative query.
            spatial = self._add(
                trace,
                "Compute the spatial change map (where, and how much)",
                reg.run_change_map(paths[0], paths[1], {}),
            )
            semantic = self._add(
                trace,
                "Describe the change in natural language (what)",
                reg.run_change_vqa(paths[0], paths[1], query, {"max_new_tokens": 64}),
            )
            trace.answer = f"{semantic.answer}\n\nSpatial evidence: {spatial.answer}"
            trace.confidence = semantic.confidence
            trace.confidence_type = semantic.confidence_type

        elif task == "optical_sar_fusion":
            evidence = self._add(
                trace,
                "Extract water and built-up masks from SAR backscatter",
                reg.run_sar_evidence(paths[0], paths[1], {}),
            )
            semantic = self._add(
                trace,
                "Interpret the optical and SAR pair jointly",
                reg.run_optical_sar(paths[0], paths[1], query, {"max_new_tokens": 64}),
            )
            trace.answer = f"{semantic.answer}\n\n{evidence.answer}"
            trace.confidence = semantic.confidence
            trace.confidence_type = semantic.confidence_type

        else:
            trace.answer = (
                "The query could not be mapped to a supported task. Supported: "
                "single-image VQA, scene description, region grounding, bi-temporal "
                "change analysis, and optical-SAR joint analysis."
            )
            trace.status = "unsupported_query"


__all__ = [
    "AgentController",
    "ExecutionTrace",
    "ExecutionStep",
    "TaskType",
    "InputValidationError",
]
