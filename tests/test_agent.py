"""
Tests for the SatQuery AI Agentic Controller.

Validates that the LLM-based intent classifier correctly routes the 5
representative queries from SIH PS 26167 to the right tasks, and that
input validation catches mismatches (e.g. change_analysis with 1 image).

NOTE: These tests make real Gemini API calls. On the free tier (5 RPM for
gemini-3.5-flash-lite), a short delay is inserted between tests to avoid
hitting rate limits.
"""

import os
import sys
import time
from unittest.mock import MagicMock, patch
import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.agent_controller import (
    AgentController,
    InputValidationError,
    TaskType,
    TASK_KEY_MAP,
)

# Delay between API calls to stay within free-tier rate limits
API_CALL_DELAY = 4  # seconds


@pytest.fixture(scope="module")
def controller():
    """Instantiate the AgentController once for all tests in this module."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set — skipping live LLM tests.")
    return AgentController()


@pytest.fixture(autouse=True)
def rate_limit_delay():
    """Insert a delay after each test to respect free-tier rate limits."""
    yield
    time.sleep(API_CALL_DELAY)


# =========================================================================
# PS Representative Query Tests
# =========================================================================

class TestPSRepresentativeQueries:
    """
    The problem statement lists 5 representative queries.
    Each must be classified to the correct task.
    """

    def test_query_1_describe_land_cover(self, controller):
        """'Describe the land-cover and major objects visible in this image.'
        -> Should classify as vqa for an open-ended description.
        """
        result = controller.classify_intent(
            query="Describe the land-cover and major objects visible in this image.",
            input_config={"image_count": 1, "input_mode": "single"},
        )
        assert result.task == "vqa", (
            f"Expected 'vqa', got '{result.task}'"
        )
        assert result.classifier_self_reported_confidence > 0.5
        print(f"  -> task={result.task}, entity={result.target_entity}, conf={result.classifier_self_reported_confidence}")

    def test_query_2_highlight_water_body(self, controller):
        """'Highlight the water body referred to in the query.'
        → Should classify as grounding (locating a region on the image).
        """
        result = controller.classify_intent(
            query="Highlight the water body referred to in the query.",
            input_config={"image_count": 1, "input_mode": "single"},
        )
        assert result.task == "grounding", (
            f"Expected 'grounding', got '{result.task}'"
        )
        assert result.target_entity is not None, "Should identify 'water body' as target"
        assert result.classifier_self_reported_confidence > 0.5
        print(f"  -> task={result.task}, entity={result.target_entity}, conf={result.classifier_self_reported_confidence}")

    def test_query_3_change_between_dates(self, controller):
        """'What changed between these two dates, and where did the change occur?'
        → Should classify as change_analysis for a bi-temporal pair.
        """
        result = controller.classify_intent(
            query="What changed between these two dates, and where did the change occur?",
            input_config={"image_count": 2, "input_mode": "bi_temporal"},
        )
        assert result.task == "change_analysis", (
            f"Expected 'change_analysis', got '{result.task}'"
        )
        assert result.classifier_self_reported_confidence > 0.5
        print(f"  -> task={result.task}, entity={result.target_entity}, conf={result.classifier_self_reported_confidence}")

    def test_query_4_optical_sar_fusion(self, controller):
        """'Use the optical and SAR images together to identify built-up and water-covered regions.'
        → Should classify as optical_sar_fusion for a cross-modal pair.
        """
        result = controller.classify_intent(
            query="Use the optical and SAR images together to identify built-up and water-covered regions.",
            input_config={"image_count": 2, "input_mode": "cross_modal"},
        )
        assert result.task == "optical_sar_fusion", (
            f"Expected 'optical_sar_fusion', got '{result.task}'"
        )
        assert result.classifier_self_reported_confidence > 0.5
        print(f"  -> task={result.task}, entity={result.target_entity}, conf={result.classifier_self_reported_confidence}")

    def test_query_5_built_up_area_change(self, controller):
        """'Has the built-up area increased, decreased, or remained unchanged?'
        → Should classify as change_analysis for a bi-temporal pair.
        """
        result = controller.classify_intent(
            query="Has the built-up area increased, decreased, or remained unchanged?",
            input_config={"image_count": 2, "input_mode": "bi_temporal"},
        )
        assert result.task == "change_analysis", (
            f"Expected 'change_analysis', got '{result.task}'"
        )
        assert result.target_entity is not None, "Should identify 'built-up area' as target"
        assert result.classifier_self_reported_confidence > 0.5
        print(f"  -> task={result.task}, entity={result.target_entity}, conf={result.classifier_self_reported_confidence}")


# =========================================================================
# Input Validation Tests
# =========================================================================

class TestInputValidation:
    """Ensure the controller rejects tasks that are impossible given the inputs."""

    @patch("google.genai.Client.models")
    def test_change_analysis_with_single_image_rejected(self, mock_models, controller):
        """A change_analysis query with only 1 image must raise InputValidationError."""
        # Mock the LLM to force it to return change_analysis despite the 1 image context
        mock_response = MagicMock()
        mock_response.text = '{"task": "change_analysis", "target_entity": null, "classifier_self_reported_confidence": 0.9}'
        mock_models.generate_content.return_value = mock_response

        with pytest.raises(InputValidationError):
            controller.classify_intent(
                query="What changed between these two dates?",
                input_config={"image_count": 1, "input_mode": "single"},
            )

    @patch("google.genai.Client.models")
    def test_optical_sar_with_single_image_rejected(self, mock_models, controller):
        """An optical_sar_fusion query with only 1 image must raise InputValidationError."""
        mock_response = MagicMock()
        mock_response.text = '{"task": "optical_sar_fusion", "target_entity": null, "classifier_self_reported_confidence": 0.9}'
        mock_models.generate_content.return_value = mock_response

        with pytest.raises(InputValidationError):
            controller.classify_intent(
                query="Use the optical and SAR images together to identify built-up regions.",
                input_config={"image_count": 1, "input_mode": "single"},
            )

    def test_vqa_with_zero_images_rejected(self, controller):
        """Any analysis query with 0 images must raise InputValidationError."""
        with pytest.raises(InputValidationError):
            controller.classify_intent(
                query="Describe this satellite image.",
                input_config={"image_count": 0, "input_mode": "single"},
            )


# =========================================================================
# Edge Case / Robustness Tests
# =========================================================================

class TestEdgeCases:
    """Test rephrased and tricky queries to ensure the LLM router is robust."""

    def test_rephrased_vqa_query(self, controller):
        """A factual question about one image, phrased differently."""
        result = controller.classify_intent(
            query="How many buildings can you count in this satellite photograph?",
            input_config={"image_count": 1, "input_mode": "single"},
        )
        assert result.task == "vqa", f"Expected 'vqa', got '{result.task}'"
        print(f"  -> task={result.task}, entity={result.target_entity}, conf={result.classifier_self_reported_confidence}")

    def test_rephrased_grounding_query(self, controller):
        """A grounding request phrased differently."""
        result = controller.classify_intent(
            query="Can you locate and draw a box around the airport in this image?",
            input_config={"image_count": 1, "input_mode": "single"},
        )
        assert result.task == "grounding", f"Expected 'grounding', got '{result.task}'"
        print(f"  -> task={result.task}, entity={result.target_entity}, conf={result.classifier_self_reported_confidence}")

    def test_ambiguous_two_image_bi_temporal(self, controller):
        """An ambiguous query with 2 bi-temporal images should default to change_analysis."""
        result = controller.classify_intent(
            query="Tell me about these two images.",
            input_config={"image_count": 2, "input_mode": "bi_temporal"},
        )
        assert result.task == "change_analysis", (
            f"Expected 'change_analysis' for ambiguous bi-temporal, got '{result.task}'"
        )
        print(f"  -> task={result.task}, entity={result.target_entity}, conf={result.classifier_self_reported_confidence}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
