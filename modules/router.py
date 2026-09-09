"""
Offline rule-based intent router for SatQuery AI.

Why this exists
---------------
The agentic controller previously depended on a Gemini API call for every
query, and raised `EnvironmentError` at construction when no key was present -
inside `@st.cache_resource`, so the whole app died at startup with a traceback.
With no network at judging time, or a rate limit, or a wrong model id, every
single query returned "Classification Error". Routing is a mandatory component
of this problem statement, so putting it behind an external service with no
fallback is a single point of failure on a live demo.

This module classifies the same five tasks deterministically, offline, in
microseconds. It is used as the fallback whenever the cloud classifier is
unavailable or fails, and the execution trace always records which backend
actually made the decision.

It is not a toy. Scoring is weighted phrase matching over a remote-sensing
lexicon, with hard constraints from the input configuration applied afterwards,
and grounding noun-phrase extraction driven by an entity vocabulary sized for
GroundingDINO prompts.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from modules.intent import IntentClassification, TASK_KEYS

# ---------------------------------------------------------------------------
# Remote-sensing entity vocabulary
# ---------------------------------------------------------------------------
# surface form -> canonical GroundingDINO noun phrase.
# Longer keys are matched first so "built-up area" wins over "area".
ENTITY_LEXICON: Dict[str, str] = {
    # water
    "water body": "water body",
    "water bodies": "water body",
    "waterbody": "water body",
    "water": "water body",
    "river": "river",
    "lake": "lake",
    "pond": "pond",
    "reservoir": "reservoir",
    "canal": "canal",
    "sea": "sea",
    "ocean": "sea",
    "coastline": "coastline",
    "shoreline": "coastline",
    "harbour": "harbor",
    "harbor": "harbor",
    "port": "harbor",
    "flood": "flooded area",
    "flooded area": "flooded area",
    # built environment
    "built-up area": "built-up area",
    "built up area": "built-up area",
    "builtup": "built-up area",
    "urban area": "urban area",
    "urban": "urban area",
    "settlement": "built-up area",
    "residential area": "residential area",
    "industrial area": "industrial area",
    "building": "building",
    "buildings": "building",
    "house": "house",
    "houses": "house",
    "rooftop": "rooftop",
    "stadium": "stadium",
    "swimming pool": "swimming pool",
    # transport
    "road": "road",
    "roads": "road",
    "highway": "highway",
    "freeway": "highway",
    "motorway": "highway",
    "street": "road",
    "bridge": "bridge",
    "railway": "railway",
    "railroad": "railway",
    "runway": "runway",
    "airport": "airport",
    "airfield": "airport",
    "parking lot": "parking lot",
    "car park": "parking lot",
    "roundabout": "roundabout",
    "intersection": "intersection",
    "junction": "intersection",
    # vegetation and land use
    "forest": "forest",
    "woodland": "forest",
    "tree": "tree",
    "trees": "tree",
    "vegetation": "vegetation",
    "farmland": "farmland",
    "cropland": "farmland",
    "agricultural land": "farmland",
    "agriculture": "farmland",
    "field": "farmland",
    "grassland": "grassland",
    "meadow": "grassland",
    "park": "park",
    "playground": "playground",
    "golf course": "golf course",
    "orchard": "orchard",
    "deforestation": "forest",
    # terrain
    "bare land": "bare land",
    "bare soil": "bare land",
    "barren": "bare land",
    "sand": "sand",
    "desert": "desert",
    "mountain": "mountain",
    "hill": "mountain",
    "beach": "beach",
    "island": "island",
    "snow": "snow",
    "glacier": "glacier",
    "cloud": "cloud",
    # discrete objects
    "ship": "ship",
    "ships": "ship",
    "boat": "boat",
    "vessel": "ship",
    "vehicle": "vehicle",
    "vehicles": "vehicle",
    "car": "car",
    "cars": "car",
    "truck": "truck",
    "aircraft": "airplane",
    "airplane": "airplane",
    "aeroplane": "airplane",
    "plane": "airplane",
    "helicopter": "helicopter",
    "storage tank": "storage tank",
    "oil tank": "storage tank",
    "tank": "storage tank",
    "silo": "silo",
    "solar panel": "solar panel",
    "wind turbine": "wind turbine",
    "container": "container",
    "crane": "crane",
    "quarry": "quarry",
    "mine": "quarry",
    "dam": "dam",
    "greenhouse": "greenhouse",
}

# Sorted longest-first so multi-word entities win.
_ENTITY_KEYS = sorted(ENTITY_LEXICON, key=len, reverse=True)

# ---------------------------------------------------------------------------
# Task cues: (phrase, weight)
# ---------------------------------------------------------------------------

GROUNDING_CUES: List[Tuple[str, float]] = [
    ("highlight", 3.0), ("locate", 3.0), ("pinpoint", 3.0), ("delineate", 3.0),
    ("outline", 2.5), ("bounding box", 3.0), ("draw a box", 3.0), ("mark the", 2.5),
    ("point out", 2.5), ("where is", 2.5), ("where are", 2.5), ("show me the", 2.0),
    ("show the location", 3.0), ("find the", 1.8), ("detect the", 1.8),
    ("segment", 2.0), ("which region", 1.5), ("region referred", 3.0),
]

CHANGE_CUES: List[Tuple[str, float]] = [
    ("what changed", 4.0), ("what has changed", 4.0), ("changed", 3.0), ("change", 2.0),
    ("difference between", 3.0), ("differences", 2.0),
    ("increased", 3.0), ("decreased", 3.0), ("expanded", 2.5), ("shrunk", 2.5),
    ("grown", 2.0), ("grew", 2.0), ("reduced", 2.0), ("remained unchanged", 3.5),
    ("before and after", 3.5), ("two dates", 3.5), ("over time", 3.0),
    ("temporal", 2.5), ("bi-temporal", 4.0), ("bitemporal", 4.0),
    ("pre-event", 3.0), ("post-event", 3.0), ("t1", 2.0), ("t2", 2.0),
    ("new construction", 3.0), ("deforestation", 2.0), ("since", 1.2),
    ("first image", 1.5), ("second image", 1.5), ("earlier", 1.5), ("later", 1.2),
]

FUSION_CUES: List[Tuple[str, float]] = [
    ("optical and sar", 4.5), ("sar and optical", 4.5), ("optical-sar", 4.5),
    ("sar", 3.0), ("radar", 3.0), ("backscatter", 3.5), ("microwave", 2.5),
    ("cross-modal", 3.5), ("both modalities", 3.5), ("use both", 2.5),
    ("combine both", 2.5), ("together to identify", 2.5), ("complementary", 2.5),
    ("all-weather", 2.0), ("through cloud", 2.0),
]

CAPTION_CUES: List[Tuple[str, float]] = [
    ("describe", 3.0), ("description", 2.5), ("caption", 3.5),
    ("summarise", 2.5), ("summarize", 2.5), ("overview", 2.0),
    ("what do you see", 3.0), ("tell me about", 2.5),
    ("land-cover and major objects", 3.5), ("land cover and major objects", 3.5),
]

VQA_CUES: List[Tuple[str, float]] = [
    ("how many", 3.0), ("count", 2.5), ("number of", 2.5),
    ("is there", 2.5), ("are there", 2.5), ("does the", 2.0), ("do the", 1.8),
    ("what is the", 2.0), ("what are the", 2.0), ("what type", 2.5),
    ("which", 1.5), ("what colour", 2.5), ("what color", 2.5),
    ("is the", 1.5), ("can you see", 2.0),
]

CUE_SETS: Dict[str, List[Tuple[str, float]]] = {
    "grounding": GROUNDING_CUES,
    "change_analysis": CHANGE_CUES,
    "optical_sar_fusion": FUSION_CUES,
    "caption": CAPTION_CUES,
    "vqa": VQA_CUES,
}

# Bonus applied when the upload slots already declare the input configuration.
MODE_BONUS: Dict[str, Dict[str, float]] = {
    "bi_temporal": {"change_analysis": 3.0},
    "cross_modal": {"optical_sar_fusion": 3.0},
    "single": {},
}


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def extract_entities(query: str, limit: int = 4) -> List[str]:
    """
    Canonical entity phrases mentioned in the query, longest match first,
    without overlapping. Returned in the order they appear.
    """
    q = _normalise(query)
    found: List[Tuple[int, str]] = []
    consumed = [False] * len(q)

    for key in _ENTITY_KEYS:
        start = 0
        while True:
            idx = q.find(key, start)
            if idx == -1:
                break
            end = idx + len(key)
            # Whole-word match only, and not inside an already-claimed span.
            before_ok = idx == 0 or not q[idx - 1].isalnum()
            after_ok = end >= len(q) or not q[end].isalnum()
            if before_ok and after_ok and not any(consumed[idx:end]):
                for i in range(idx, end):
                    consumed[i] = True
                found.append((idx, ENTITY_LEXICON[key]))
            start = end

    found.sort(key=lambda p: p[0])
    out: List[str] = []
    for _, canonical in found:
        if canonical not in out:
            out.append(canonical)
        if len(out) >= limit:
            break
    return out


def score_tasks(query: str, input_mode: str) -> Dict[str, float]:
    q = _normalise(query)
    scores = {k: 0.0 for k in TASK_KEYS}

    for task, cues in CUE_SETS.items():
        for phrase, weight in cues:
            if phrase in q:
                scores[task] += weight

    for task, bonus in MODE_BONUS.get(input_mode, {}).items():
        scores[task] += bonus

    # A bare question with no cue at all is still a question about the image.
    if max(scores.values()) == 0.0:
        scores["vqa"] = 1.0

    return scores


class LocalIntentRouter:
    """Deterministic offline classifier. Same output type as the cloud path."""

    name = "local-rules-v1"

    def classify(self, query: str, input_config: Dict[str, object]) -> IntentClassification:
        image_count = int(input_config.get("image_count", 0) or 0)
        input_mode = str(input_config.get("input_mode", "single") or "single")

        scores = score_tasks(query, input_mode)
        entities = extract_entities(query)

        # Hard constraint: paired tasks need two images.
        if image_count < 2:
            for task in ("change_analysis", "optical_sar_fusion"):
                scores[task] = 0.0

        # Grounding without an identifiable target is not actionable.
        if not entities:
            scores["grounding"] = 0.0

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_task, top_score = ranked[0]
        runner_score = ranked[1][1] if len(ranked) > 1 else 0.0

        if top_score <= 0.0:
            top_task, top_score = "vqa", 1.0

        # Confidence from the margin: a clear winner scores high, a tie does not.
        if top_score <= 0:
            confidence = 0.35
        else:
            margin = (top_score - runner_score) / top_score
            confidence = 0.45 + 0.50 * max(0.0, min(1.0, margin))

        search_nouns = (
            [f"{e}." for e in entities] if top_task == "grounding" else None
        )

        rationale = (
            f"rule-based scores: "
            + ", ".join(f"{k}={v:.1f}" for k, v in ranked if v > 0)
            + (f"; entities={entities}" if entities else "; no entity matched")
            + f"; input_mode={input_mode}, image_count={image_count}"
        )

        return IntentClassification(
            task=top_task,
            target_entity=entities[0] if entities else None,
            search_nouns=search_nouns,
            classifier_self_reported_confidence=round(confidence, 3),
            backend=self.name,
            rationale=rationale,
        )


__all__ = ["LocalIntentRouter", "extract_entities", "score_tasks", "ENTITY_LEXICON"]
