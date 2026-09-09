"""
Shared intent schema for SatQuery AI.

Kept in its own module so the cloud classifier (`agent_controller`) and the
offline rule-based classifier (`router`) can both produce and consume the same
objects without importing each other.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    SINGLE_VQA = "Single-Image Visual Question Answering"
    SINGLE_CAPTION = "Single-Image Scene Description"
    SINGLE_GROUNDING = "Text-Guided Region Grounding"
    CHANGE_ANALYSIS = "Bi-Temporal Change Detection & VQA"
    OPTICAL_SAR_FUSION = "Optical-SAR Cross-Modal Analysis"
    UNKNOWN = "Unknown / Unsupported Query"


TASK_KEY_MAP: Dict[str, TaskType] = {
    "vqa": TaskType.SINGLE_VQA,
    "caption": TaskType.SINGLE_CAPTION,
    "grounding": TaskType.SINGLE_GROUNDING,
    "change_analysis": TaskType.CHANGE_ANALYSIS,
    "optical_sar_fusion": TaskType.OPTICAL_SAR_FUSION,
}

TASK_KEYS = tuple(TASK_KEY_MAP.keys())

SINGLE_IMAGE_TASKS = ("vqa", "caption", "grounding")
PAIR_TASKS = ("change_analysis", "optical_sar_fusion")


class IntentClassification(BaseModel):
    """The routing decision, whoever made it."""

    task: str = Field(
        description=(
            "The classified remote-sensing task. Exactly one of: "
            "vqa, caption, grounding, change_analysis, optical_sar_fusion"
        )
    )
    target_entity: Optional[str] = Field(
        default=None,
        description=(
            "The object, region or phenomenon the query refers to, e.g. "
            "'water body', 'built-up area'. Null for a general question."
        ),
    )
    search_nouns: Optional[List[str]] = Field(
        default=None,
        description=(
            "Noun phrases formatted for GroundingDINO, each ending in a period, "
            "e.g. ['water body.', 'road.']. Required when task is 'grounding'."
        ),
    )
    classifier_self_reported_confidence: float = Field(
        default=0.5,
        description="Confidence in the CLASSIFICATION, 0.0-1.0. Not an answer confidence.",
    )
    # Populated by the controller, not by the classifier itself.
    backend: Optional[str] = None
    rationale: Optional[str] = None


class InputValidationError(Exception):
    """The classified task is incompatible with the images actually supplied."""


__all__ = [
    "TaskType",
    "TASK_KEY_MAP",
    "TASK_KEYS",
    "SINGLE_IMAGE_TASKS",
    "PAIR_TASKS",
    "IntentClassification",
    "InputValidationError",
]
