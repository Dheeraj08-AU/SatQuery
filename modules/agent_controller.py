"""
Agentic Controller for SatQuery AI.

Replaces the old keyword-based if/else intent classifier with an LLM-powered
intent router that calls the Gemini API with structured JSON output.

The controller:
1. Classifies the user's query intent via Gemini (structured output).
2. Validates the classified task against the actual input configuration
   (e.g. rejects "change_analysis" if only one image was uploaded).
3. Routes to the appropriate model/tool in the ModelRegistry.
4. Returns a fully auditable ExecutionTrace.
"""

import os
import json
import logging
from enum import Enum
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()  # Load .env file if present

from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from modules.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task & Schema Definitions
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    SINGLE_VQA = "Single-Image Visual Question Answering"
    SINGLE_GROUNDING = "Text-Guided Region Grounding"
    CHANGE_ANALYSIS = "Bi-Temporal Change Detection & VQA"
    OPTICAL_SAR_FUSION = "Optical-SAR Cross-Modal Analysis"
    UNKNOWN = "Unknown / Unsupported Query"


# Maps the short task key returned by the LLM to the TaskType enum.
TASK_KEY_MAP: Dict[str, TaskType] = {
    "vqa": TaskType.SINGLE_VQA,
    "grounding": TaskType.SINGLE_GROUNDING,
    "change_analysis": TaskType.CHANGE_ANALYSIS,
    "optical_sar_fusion": TaskType.OPTICAL_SAR_FUSION,
}


class IntentClassification(BaseModel):
    """Schema the LLM must return — enforced via Gemini structured output."""
    task: str = Field(
        description=(
            "The classified remote-sensing task. Must be exactly one of: "
            "vqa, grounding, change_analysis, optical_sar_fusion"
        )
    )
    target_entity: Optional[str] = Field(
        default=None,
        description=(
            "The specific object, region, or phenomenon the query refers to, "
            "e.g. 'water body', 'built-up area', 'vegetation'. "
            "None if the query is a general question."
        ),
    )
    search_nouns: Optional[List[str]] = Field(
        default=None,
        description=(
            "A list of period-separated noun phrases extracted from the user's query, "
            "formatted specifically for GroundingDINO (e.g. ['built-up area.', 'road.']). "
            "Must be provided if task is 'grounding'."
        ),
    )
    classifier_self_reported_confidence: float = Field(
        description="Self-reported confidence score between 0.0 and 1.0 for the classification. Be honest about ambiguity."
    )


class ExecutionTrace(BaseModel):
    selected_task: TaskType
    input_count: int
    selected_tool: str
    permitted_parameters: Dict[str, Any]
    status: str
    reasoning_summary: str
    execution_result: Optional[Any] = None


class InputValidationError(Exception):
    """Raised when the classified task is incompatible with the uploaded images."""
    pass


# ---------------------------------------------------------------------------
# System Prompt for Intent Classification
# ---------------------------------------------------------------------------

INTENT_SYSTEM_PROMPT = """\
You are the intent-routing component of SatQuery AI, an agentic vision-language
assistant for remote-sensing image analysis.

Your job is to classify the user's natural-language query into exactly ONE of the
following remote-sensing tasks:

TASKS:
- "vqa"               → Single-image visual question answering AND scene description.
                         (e.g. counting objects, identifying features, or open-ended
                         descriptions like "describe the land-cover").
- "grounding"         → Text-guided region grounding — locating or highlighting a
                         specific object/region referred to in the query on a single image
                         (e.g. "highlight the water body", "locate the airport").
- "change_analysis"   → Bi-temporal change detection & VQA — identifying, describing,
                         or answering questions about what changed between two images
                         acquired at different times (e.g. "what changed", "has the
                         built-up area increased").
- "optical_sar_fusion" → Cross-modal optical + SAR analysis — jointly interpreting
                         co-registered optical and SAR images to extract complementary
                         information (e.g. "use optical and SAR together to identify
                         built-up and water regions").

INPUT CONTEXT (provided with each query):
- image_count: how many images the user uploaded (1 or 2).
- input_mode: one of "single", "bi_temporal", or "cross_modal".

CLASSIFICATION RULES:
1. If only 1 image is available, NEVER classify as "change_analysis" or "optical_sar_fusion".
2. If input_mode is "bi_temporal", prefer "change_analysis" for ambiguous 2-image queries.
3. If input_mode is "cross_modal", prefer "optical_sar_fusion" for ambiguous 2-image queries.
4. "grounding" requires the query to explicitly ask for locating, highlighting, or
   bounding-boxing a specific entity.

Return ONLY the structured JSON. Do not add any explanation.
"""


# ---------------------------------------------------------------------------
# AgentController
# ---------------------------------------------------------------------------

class AgentController:
    def __init__(self):
        self.registry = ModelRegistry()

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set. "
                "The agentic controller requires a Gemini API key for intent classification."
            )
        self.client = genai.Client(api_key=api_key)

    # ---- LLM-based intent classifier ----

    def classify_intent(
        self,
        query: str,
        input_config: Dict[str, Any],
    ) -> IntentClassification:
        """
        Call Gemini to classify the user's query into one of the defined
        remote-sensing tasks, returning structured JSON.

        Parameters
        ----------
        query : str
            The user's natural-language query.
        input_config : dict
            Must contain:
              - "image_count" (int): number of uploaded images.
              - "input_mode" (str): one of "single", "bi_temporal", "cross_modal".

        Returns
        -------
        IntentClassification
            Pydantic model with task, target_entity, confidence.

        Raises
        ------
        InputValidationError
            If the classified task is incompatible with the available images.
        """
        image_count = input_config.get("image_count", 0)
        input_mode = input_config.get("input_mode", "single")

        user_message = (
            f"User query: \"{query}\"\n"
            f"Input context: image_count={image_count}, input_mode={input_mode}"
        )

        # Retry with backoff to handle free-tier rate limits (5 RPM)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        system_instruction=INTENT_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_json_schema=IntentClassification.model_json_schema(),
                        temperature=0.0,
                    ),
                )
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    import time
                    wait_time = (attempt + 1) * 5  # 5s, 10s, 15s
                    logger.warning(f"Rate limited. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise

        classification = IntentClassification.model_validate_json(response.text)

        # --- Validate against actual inputs ---
        self._validate_task_against_inputs(classification, input_config)

        return classification

    @staticmethod
    def _validate_task_against_inputs(
        classification: IntentClassification,
        input_config: Dict[str, Any],
    ) -> None:
        """
        Ensure the LLM's classified task is physically possible given
        the images the user actually uploaded.
        """
        image_count = input_config.get("image_count", 0)
        task = classification.task

        if task in ("change_analysis", "optical_sar_fusion") and image_count < 2:
            raise InputValidationError(
                f"Task '{task}' requires 2 images, but only {image_count} "
                f"image(s) were uploaded. Please upload a second image or "
                f"rephrase your query for single-image analysis."
            )

        if task in ("vqa", "captioning", "grounding") and image_count == 0:
            raise InputValidationError(
                f"Task '{task}' requires at least 1 image, but none were uploaded."
            )

        if task == "grounding" and not classification.search_nouns:
            raise InputValidationError(
                "Task 'grounding' requires 'search_nouns' to be provided by the classifier."
            )

        valid_tasks = set(TASK_KEY_MAP.keys())
        if task not in valid_tasks:
            logger.error(f"HARD VALIDATION ERROR: LLM returned invalid task '{task}'. Expected one of {valid_tasks}")
            raise InputValidationError(
                f"LLM returned unknown task '{task}'. "
                f"Valid tasks are: {valid_tasks}"
            )

    # ---- Route to the correct tool ----

    def route_and_configure(
        self,
        query: str,
        image_paths: List[str],
        input_mode: str = "single",
    ) -> ExecutionTrace:
        """
        Full agentic pipeline:
        1. Classify intent via LLM.
        2. Validate against inputs.
        3. Select tool + parameters.
        4. Execute and return auditable trace.
        """
        num_images = len(image_paths)
        input_config = {
            "image_count": num_images,
            "input_mode": input_mode,
        }

        # --- Step 1 & 2: Classify + Validate ---
        try:
            classification = self.classify_intent(query, input_config)
        except InputValidationError as e:
            return ExecutionTrace(
                selected_task=TaskType.UNKNOWN,
                input_count=num_images,
                selected_tool="None",
                permitted_parameters={},
                status=f"Validation Error: {e}",
                reasoning_summary=str(e),
            )
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return ExecutionTrace(
                selected_task=TaskType.UNKNOWN,
                input_count=num_images,
                selected_tool="None",
                permitted_parameters={},
                status=f"Classification Error: {e}",
                reasoning_summary=f"LLM intent classification failed: {e}",
            )

        task_type = TASK_KEY_MAP.get(classification.task, TaskType.UNKNOWN)
        result = None
        conf_score = classification.classifier_self_reported_confidence

        # --- Step 3 & 4: Select tool, configure params, execute ---
        if task_type == TaskType.SINGLE_VQA:
            tool = self.registry.loaded_tools["vqa"]
            params = {"max_new_tokens": 128, "temperature": 0.2}
            summary = (
                f"Routed single image to Remote Sensing VQA module. "
                f"Target entity: {classification.target_entity or 'general'}. "
                f"Self-reported confidence: {conf_score:.2f}"
            )
            if image_paths:
                result = self.registry.run_single_vqa(image_paths[0], query, params)

        elif task_type == TaskType.SINGLE_GROUNDING:
            tool = self.registry.loaded_tools["grounding"]
            params = {"box_threshold": 0.35, "text_threshold": 0.25}
            summary = (
                f"Routed query to Text-Guided Spatial Grounding tool. "
                f"Target entity: {classification.target_entity or 'unknown'}. "
                f"Self-reported confidence: {conf_score:.2f}"
            )
            if image_paths:
                result = self.registry.run_grounding(image_paths[0], classification.search_nouns, params)

        elif task_type == TaskType.CHANGE_ANALYSIS:
            tool = self.registry.loaded_tools["change"]
            params = {"difference_threshold": 0.5, "generate_change_map": True}
            summary = (
                f"Bi-temporal image pair routed to Change-VQA pipeline. "
                f"Target entity: {classification.target_entity or 'general change'}. "
                f"Self-reported confidence: {conf_score:.2f}"
            )
            if len(image_paths) == 2:
                result = self.registry.run_change_analysis(
                    image_paths[0], image_paths[1], query, params
                )

        elif task_type == TaskType.OPTICAL_SAR_FUSION:
            tool = "Dual-Branch Optical-SAR Fusion Network"
            params = {"fusion_strategy": "cross_attention", "sar_decibel_norm": True}
            summary = (
                f"Co-registered Optical and SAR images routed to multi-modal extractor. "
                f"Target entity: {classification.target_entity or 'joint analysis'}. "
                f"Self-reported confidence: {conf_score:.2f}"
            )
            if len(image_paths) == 2:
                result = self.registry.run_optical_sar(
                    image_paths[0], image_paths[1], params
                )

        else:
            tool = "None"
            params = {}
            summary = "Task routing failed. Input configuration or query intent invalid."

        return ExecutionTrace(
            selected_task=task_type,
            input_count=num_images,
            selected_tool=tool,
            permitted_parameters=params,
            status="Executed" if tool != "None" else "Failed",
            reasoning_summary=summary,
            execution_result=result,
        )