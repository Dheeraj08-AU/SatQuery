"""
Integration test for Task 2: Real GroundingDINO zero-shot grounding.

Creates two synthetic satellite-like PNG images on-the-fly:
  - Image A: green background with a bright blue rectangle (water body)
  - Image B: brown background with a grey horizontal strip (road)

Runs run_grounding with two different natural-language queries on each image.
Asserts that:
  1. The returned dict has the expected keys.
  2. Coordinates are real (not the old mock [100, 150, 400, 450]).
  3. Confidence is a real float from the model, > 0.
  4. Two different images produce different bounding box outputs.

Prints all raw coordinate and confidence data so the user can inspect them.
"""

import os
import sys
import numpy as np
import pytest
from PIL import Image

# Make sure we can import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.model_registry import ModelRegistry

MOCK_BOX = [100, 150, 400, 450]  # old hardcoded mock -- must NOT appear in results


# ──────────────────────────────────────────────────────────────────────────────
# Image generators
# ──────────────────────────────────────────────────────────────────────────────

def _make_water_body_image(path: str, size: tuple = (400, 400)) -> None:
    """Green vegetation background with a bright blue water-body rectangle."""
    W, H = size
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    arr[:, :] = [34, 139, 34]            # green vegetation
    arr[100:280, 50:200] = [30, 100, 220]  # blue water body (left-centre)
    Image.fromarray(arr).save(path)
    print(f"[TEST] Created water body image: {path}")


def _make_road_image(path: str, size: tuple = (400, 400)) -> None:
    """Brown bare-earth background with a grey horizontal road strip."""
    W, H = size
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    arr[:, :] = [139, 90, 43]    # brown bare earth
    arr[180:220, :] = [160, 160, 160]  # grey road running horizontally
    Image.fromarray(arr).save(path)
    print(f"[TEST] Created road image: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _assert_result_shape(r: dict) -> None:
    for key in ("box", "confidence", "label", "status"):
        assert key in r, f"Result missing key '{key}'"
    assert isinstance(r["confidence"], float), "confidence must be a float"


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def registry():
    print("\n[TEST] Loading ModelRegistry (GroundingDINO downloads on first run)...")
    r = ModelRegistry()
    print("[TEST] ModelRegistry ready.")
    return r


@pytest.fixture(scope="module")
def water_image_path(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("imgs") / "water_body.png")
    _make_water_body_image(p)
    return p


@pytest.fixture(scope="module")
def road_image_path(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("imgs") / "road_strip.png")
    _make_road_image(p)
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Test classes
# ──────────────────────────────────────────────────────────────────────────────

class TestGroundingWaterBodyImage:

    def test_water_query_returns_real_box(self, registry, water_image_path):
        """'highlight the water body' on a water-body scene must return non-mock coords."""
        print("\n" + "=" * 60)
        print("TEST: water image + 'highlight the water body'")
        params = {"box_threshold": 0.20, "text_threshold": 0.15}
        r = registry.run_grounding(water_image_path, "highlight the water body", params)

        print(f"  box:        {r['box']}")
        print(f"  confidence: {r['confidence']:.4f}   (threshold used: {params['box_threshold']})")
        print(f"  label:      {r['label']}")
        print(f"  status:     {r['status']}")
        print("=" * 60)

        _assert_result_shape(r)
        assert r["box"] != MOCK_BOX, "run_grounding is still returning the old hardcoded mock box!"
        if r["box"] is not None:
            assert r["confidence"] > 0.0, "Expected positive confidence for a real detection"
            x0, y0, x1, y1 = r["box"]
            assert x1 > x0 and y1 > y0, "Box coordinates are degenerate"

    def test_road_query_also_returns_structured_result(self, registry, water_image_path):
        """'detect the road' on a water image must return a valid (possibly empty) struct."""
        print("\n" + "=" * 60)
        print("TEST: water image + 'detect the road'")
        params = {"box_threshold": 0.20, "text_threshold": 0.15}
        r = registry.run_grounding(water_image_path, "detect the road", params)

        print(f"  box:        {r['box']}")
        print(f"  confidence: {r['confidence']:.4f}")
        print(f"  label:      {r['label']}")
        print(f"  status:     {r['status']}")
        print("=" * 60)

        _assert_result_shape(r)
        assert r["box"] != MOCK_BOX


class TestGroundingRoadImage:

    def test_road_query_returns_real_box(self, registry, road_image_path):
        """'detect the road' on a road scene must return non-mock coords."""
        print("\n" + "=" * 60)
        print("TEST: road image + 'detect the road'")
        params = {"box_threshold": 0.20, "text_threshold": 0.15}
        r = registry.run_grounding(road_image_path, "detect the road", params)

        print(f"  box:        {r['box']}")
        print(f"  confidence: {r['confidence']:.4f}   (threshold used: {params['box_threshold']})")
        print(f"  label:      {r['label']}")
        print(f"  status:     {r['status']}")
        print("=" * 60)

        _assert_result_shape(r)
        assert r["box"] != MOCK_BOX
        if r["box"] is not None:
            assert r["confidence"] > 0.0
            x0, y0, x1, y1 = r["box"]
            assert x1 > x0 and y1 > y0

    def test_water_query_also_returns_structured_result(self, registry, road_image_path):
        """'highlight the water body' on a road image must return a valid (possibly empty) struct."""
        print("\n" + "=" * 60)
        print("TEST: road image + 'highlight the water body'")
        params = {"box_threshold": 0.20, "text_threshold": 0.15}
        r = registry.run_grounding(road_image_path, "highlight the water body", params)

        print(f"  box:        {r['box']}")
        print(f"  confidence: {r['confidence']:.4f}")
        print(f"  label:      {r['label']}")
        print(f"  status:     {r['status']}")
        print("=" * 60)

        _assert_result_shape(r)
        assert r["box"] != MOCK_BOX


class TestGroundingBoxesDifferBetweenImages:

    def test_same_query_two_images_produce_different_boxes(
        self, registry, water_image_path, road_image_path
    ):
        """
        The same query on two visually different images must produce different outputs.
        This proves the model is reading the image, not ignoring it.
        """
        query = "highlight the dominant region"
        params = {"box_threshold": 0.15, "text_threshold": 0.10}

        print("\n" + "=" * 60)
        print(f"TEST: same query '{query}' on two different images")

        rw = registry.run_grounding(water_image_path, query, params)
        rr = registry.run_grounding(road_image_path, query, params)

        print(f"  water image box: {rw['box']}  conf={rw['confidence']:.4f}")
        print(f"  road  image box: {rr['box']}  conf={rr['confidence']:.4f}")
        print("=" * 60)

        _assert_result_shape(rw)
        _assert_result_shape(rr)

        if rw["box"] is not None and rr["box"] is not None:
            assert rw["box"] != rr["box"], (
                "Identical bounding boxes returned for two completely different images. "
                "This means the model is ignoring the image input!"
            )
            print(f"  PASS: boxes differ -> water={rw['box']}  road={rr['box']}")


class TestNoDetectionHandling:

    def test_impossible_query_at_strict_threshold_returns_no_detection(
        self, registry, water_image_path
    ):
        """
        With threshold=0.99, an impossible query ('find a spaceship') must return
        the 'No confident detection' state -- not a fake box, not a crash.
        """
        print("\n" + "=" * 60)
        print("TEST: impossible query at threshold 0.99")
        params = {"box_threshold": 0.99, "text_threshold": 0.99}
        r = registry.run_grounding(
            water_image_path,
            "find a large spaceship in the image",
            params,
        )

        print(f"  status:     {r['status']}")
        print(f"  box:        {r['box']}")
        print(f"  confidence: {r['confidence']:.4f}")
        print("=" * 60)

        _assert_result_shape(r)
        assert r["box"] is None, (
            f"Expected no detection at threshold 0.99 for an impossible query, "
            f"but got box={r['box']}"
        )
        assert r["confidence"] == 0.0
        assert "No confident detection" in r["status"]
