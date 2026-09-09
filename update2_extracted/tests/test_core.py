"""
Unit tests for SatQuery AI's deterministic core.

These need no model weights, no GPU and no network, so they run in seconds on
any machine and in CI. They cover the parts of the system that can be checked
against a known-correct answer: raster I/O maths, change detection, SAR
evidence extraction, composite geometry, intent routing, and input validation.

The model-dependent paths are scored by `eval/run_benchmarks.py` instead,
because "is the VQA answer right" is a benchmark question, not a unit test.

Run with:  pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.change_detection import (
    binary_close,
    binary_open,
    detect_change,
    otsu_threshold,
    radiometric_normalize,
)
from modules.composite import make_pair_composite, make_single_image
from modules.geo_validator import CoregistrationStatus, SatQueryValidator
from modules.raster_io import (
    box_mean,
    choose_rgb_bands,
    infer_modality,
    lee_filter,
    load_as_rgb,
    percentile_stretch,
    to_decibels,
)
from modules.router import LocalIntentRouter, extract_entities
from modules.sar_analysis import analyse_optical_sar

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def write_raster(
    path,
    array,
    crs="EPSG:32643",
    origin=(500000.0, 2000000.0),
    res=10.0,
    dtype=None,
):
    """Write a (bands, H, W) array as a GeoTIFF."""
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None, ...]
    dtype = dtype or array.dtype
    transform = from_origin(origin[0], origin[1], res, res)
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=array.shape[1],
        width=array.shape[2],
        count=array.shape[0],
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(array.astype(dtype))
    return str(path)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


# ---------------------------------------------------------------------------
# raster_io maths
# ---------------------------------------------------------------------------


def test_box_mean_matches_naive_convolution(rng):
    a = rng.random((17, 23))
    radius = 2
    got = box_mean(a, radius)

    padded = np.pad(a, radius, mode="reflect")
    k = 2 * radius + 1
    want = np.empty_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            want[i, j] = padded[i : i + k, j : j + k].mean()

    assert np.allclose(got, want, atol=1e-12)


def test_box_mean_of_constant_is_constant():
    a = np.full((8, 8), 3.5)
    assert np.allclose(box_mean(a, 3), 3.5)


def test_percentile_stretch_spans_full_range(rng):
    band = rng.normal(1000, 200, size=(64, 64))
    out, (lo, hi) = percentile_stretch(band, lo_pct=2, hi_pct=98)
    assert out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255
    assert lo < hi


def test_percentile_stretch_handles_constant_band():
    out, _ = percentile_stretch(np.full((10, 10), 7.0))
    assert out.shape == (10, 10)
    assert np.isfinite(out).all()


def test_to_decibels_integer_uses_20log10():
    arr = np.array([[10.0, 100.0]])
    db, label, _ = to_decibels(arr, is_integer_dn=True)
    assert label == "20log10"
    # 20*log10(100) - 20*log10(10) == 20 dB
    assert db[0, 1] - db[0, 0] == pytest.approx(20.0, abs=1e-6)


def test_to_decibels_float_uses_10log10():
    arr = np.array([[0.01, 0.1]])
    db, label, _ = to_decibels(arr, is_integer_dn=False)
    assert label == "10log10"
    assert db[0, 1] - db[0, 0] == pytest.approx(10.0, abs=1e-6)


def test_to_decibels_passes_through_existing_db():
    arr = np.array([[-20.0, -5.0]])
    _, label, _ = to_decibels(arr, is_integer_dn=False)
    assert label == "already_db"


def test_lee_filter_reduces_speckle_variance(rng):
    clean = np.full((64, 64), 100.0)
    speckled = clean * rng.gamma(shape=1.0, scale=1.0, size=clean.shape)
    filtered = lee_filter(speckled, radius=2)
    assert filtered.std() < speckled.std()


def test_rgba_png_is_not_band_swapped():
    """
    A 4-band RGBA raster must map to bands (1,2,3), not the B,G,R,NIR layout.
    Getting this wrong silently swaps red and blue on every PNG input.
    """
    idx, _ = choose_rgb_bands(4, [], ["red", "green", "blue", "alpha"])
    assert idx == (1, 2, 3)


def test_four_band_multispectral_without_colorinterp_uses_bgrn_layout():
    idx, _ = choose_rgb_bands(4, [], [])
    assert idx == (3, 2, 1)


def test_sentinel2_twelve_band_picks_b04_b03_b02():
    idx, _ = choose_rgb_bands(12, [], [])
    assert idx == (4, 3, 2)


def test_sentinel2_band_descriptions_win_over_count():
    names = ["B01", "B02", "B03", "B04", "B08"]
    idx, _ = choose_rgb_bands(5, names, [])
    assert idx == (4, 3, 2)


def test_infer_modality_from_filename():
    assert infer_modality(1, "float32", "/data/S1A_IW_GRDH_VV.tif") == "sar"
    assert infer_modality(12, "uint16", "/data/S2A_MSI.tif") == "optical"


# ---------------------------------------------------------------------------
# raster_io end to end
# ---------------------------------------------------------------------------


def test_multispectral_geotiff_loads_as_rgb(tmp_path, rng):
    """A 12-band uint16 scene must render - PIL alone cannot do this."""
    data = rng.integers(0, 4000, size=(12, 40, 50), dtype=np.uint16)
    path = write_raster(tmp_path / "s2.tif", data, dtype="uint16")

    img, report = load_as_rgb(path, modality="optical")
    assert img.size == (50, 40)
    assert img.mode == "RGB"
    assert report.band_count == 12
    assert report.bands_used == (4, 3, 2)
    assert report.modality == "optical"


def test_single_band_float_sar_loads_and_is_speckle_filtered(tmp_path, rng):
    power = rng.gamma(shape=1.0, scale=0.05, size=(40, 40)).astype(np.float32)
    path = write_raster(tmp_path / "sar.tif", power, dtype="float32")

    img, report = load_as_rgb(path, modality="sar")
    assert img.size == (40, 40)
    assert report.db_conversion == "10log10"
    assert report.speckle_filter is not None
    assert "lee" in report.speckle_filter


def test_nodata_is_excluded_from_the_stretch(tmp_path):
    data = np.full((1, 20, 20), 500.0, dtype="float32")
    data[0, :5, :] = -9999.0
    path = str(tmp_path / "nd.tif")
    transform = from_origin(0, 0, 1, 1)
    with rasterio.open(
        path, "w", driver="GTiff", height=20, width=20, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data)

    _, report = load_as_rgb(path, modality="optical")
    assert report.nodata_fraction == pytest.approx(0.25, abs=0.01)


# ---------------------------------------------------------------------------
# change detection
# ---------------------------------------------------------------------------


def test_otsu_separates_a_clean_bimodal_distribution():
    values = np.concatenate([np.full(5000, 0.1), np.full(5000, 0.8)])
    t, separability = otsu_threshold(values)
    assert 0.1 < t < 0.8
    assert separability > 0.9


def test_otsu_separability_is_low_for_unimodal_data(rng):
    values = np.clip(rng.normal(0.5, 0.02, size=10000), 0, 1)
    _, separability = otsu_threshold(values)
    assert separability < 0.9


def test_radiometric_normalize_is_identity_on_identical_images(rng):
    ref = rng.random((32, 32, 3)) * 255
    out, params = radiometric_normalize(ref, ref.copy())
    assert np.allclose(out, ref, atol=1e-6)
    for a, b in params:
        assert a == pytest.approx(1.0, abs=1e-3)
        assert b == pytest.approx(0.0, abs=1e-2)


def test_radiometric_normalize_undoes_a_gain_and_offset(rng):
    ref = rng.random((48, 48, 3)) * 200 + 20
    dimmed = ref * 0.6 + 15
    out, _ = radiometric_normalize(ref, dimmed)
    # Should recover the reference far better than the uncorrected version.
    assert np.abs(out - ref).mean() < np.abs(dimmed - ref).mean() / 4


def test_binary_open_removes_isolated_speckle():
    mask = np.zeros((32, 32), dtype=bool)
    mask[16, 16] = True                 # single pixel
    mask[4:12, 4:12] = True             # solid block
    opened = binary_open(mask, radius=1)
    assert not opened[16, 16]
    assert opened[7, 7]


def test_binary_close_fills_a_pinhole():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    mask[16, 16] = False
    closed = binary_close(mask, radius=1)
    assert closed[16, 16]


def test_no_change_pair_reports_no_significant_change(rng):
    base = (rng.random((96, 96, 3)) * 255).astype(np.uint8)
    img = Image.fromarray(base)
    result = detect_change(img, img.copy())
    assert result.changed_fraction < 0.02
    assert "No significant change" in result.summary


def test_illumination_shift_alone_is_not_reported_as_change(rng):
    """
    The old pixel-diff flagged the whole frame when brightness shifted. The
    radiometric normalisation step exists precisely to stop that.
    """
    # Range chosen so the gain does not clip at 255 - clipping is a real
    # difference, and would make this test measure the wrong thing.
    base = (rng.random((96, 96, 3)) * 140 + 30).astype(np.uint8)
    brighter = np.clip(base.astype(np.float64) * 1.25 + 12, 0, 255).astype(np.uint8)
    result = detect_change(Image.fromarray(base), Image.fromarray(brighter))
    assert result.changed_fraction < 0.15


def test_a_real_change_is_detected_in_the_correct_quadrant(rng):
    base = (rng.random((128, 128, 3)) * 60 + 90).astype(np.uint8)
    after = base.copy()
    # Bright block in the lower-right quadrant.
    after[80:120, 80:120] = 250

    result = detect_change(Image.fromarray(base), Image.fromarray(after))

    assert result.changed_fraction > 0.03
    assert result.regions, "expected at least one change region"
    biggest = result.regions[0]
    cx, cy = biggest.centroid
    assert cx > 64 and cy > 64
    assert "southern" in biggest.direction and "eastern" in biggest.direction


def test_change_overlay_keeps_the_underlying_imagery():
    """The overlay must be drawn over T2, not on a black field."""
    base = np.full((32, 32, 3), 120, dtype=np.uint8)
    after = base.copy()
    after[0:8, 0:8] = 255
    result = detect_change(Image.fromarray(base), Image.fromarray(after))
    overlay = np.asarray(result.overlay)
    # An unchanged corner should still show the original grey, not black.
    assert overlay[28, 28].mean() > 50


def test_mismatched_sizes_are_resized_with_a_warning(rng):
    a = Image.fromarray((rng.random((64, 64, 3)) * 255).astype(np.uint8))
    b = Image.fromarray((rng.random((32, 48, 3)) * 255).astype(np.uint8))
    result = detect_change(a, b)
    assert result.mask.shape == (64, 64)
    assert any("common grid" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# SAR evidence
# ---------------------------------------------------------------------------


def test_dark_sar_region_is_classified_as_water(rng):
    # The dark patch must be larger than the water percentile (default 15%),
    # otherwise the threshold lands inside the bright population and the test
    # asserts nothing meaningful. 32x32 of 64x64 is 25%.
    optical = np.full((64, 64, 3), 90, dtype=np.uint8)
    sar = (rng.random((64, 64)) * 40 + 180).astype(np.uint8)
    sar[0:32, 0:32] = 5                      # radar-dark (specular) patch
    optical[0:32, 0:32] = 30                 # dark in optical too

    ev = analyse_optical_sar(
        Image.fromarray(optical),
        Image.fromarray(np.stack([sar] * 3, axis=-1)),
    )
    assert ev.water_fraction > 0.05
    assert ev.water_mask[8, 8]
    assert not ev.water_mask[50, 50]


def test_radar_dark_but_optically_bright_is_rejected_as_water():
    """Radar shadow and smooth pavement are radar-dark but not water."""
    optical = np.full((64, 64, 3), 50, dtype=np.uint8)
    sar = np.full((64, 64), 200, dtype=np.uint8)
    sar[0:32, 0:32] = 5           # radar-dark
    optical[0:32, 0:32] = 250     # ...but bright in optical, so not water

    ev = analyse_optical_sar(
        Image.fromarray(optical),
        Image.fromarray(np.stack([sar] * 3, axis=-1)),
    )
    assert ev.water_fraction < 0.01
    assert any("rejected as water" in w for w in ev.warnings)


def test_sar_method_records_the_calibration_caveat():
    img = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
    ev = analyse_optical_sar(img, img)
    assert "NOT calibrated sigma0" in ev.method["calibration"]


# ---------------------------------------------------------------------------
# composites
# ---------------------------------------------------------------------------


def test_pair_composite_is_square_and_preserves_aspect():
    a = Image.new("RGB", (100, 100), (255, 0, 0))
    b = Image.new("RGB", (100, 100), (0, 0, 255))
    comp = make_pair_composite(a, b, size=224)

    assert comp.size == (224, 224)
    px = np.asarray(comp)
    # Left cell red, right cell blue, at the vertical centre.
    assert px[112, 56, 0] > 200 and px[112, 56, 2] < 60
    assert px[112, 168, 2] > 200 and px[112, 168, 0] < 60


def test_pair_composite_does_not_stretch_a_wide_image():
    """A 2:1 input must stay 2:1 inside its cell, letterboxed, not squashed."""
    wide = Image.new("RGB", (200, 100), (0, 255, 0))
    other = Image.new("RGB", (100, 100), (0, 0, 0))
    comp = make_pair_composite(wide, other, size=224)
    px = np.asarray(comp)

    green_cols = np.where((px[:, :112, 1] > 200).any(axis=0))[0]
    green_rows = np.where((px[:, :112, 1] > 200).any(axis=1))[0]
    width = green_cols.max() - green_cols.min() + 1
    height = green_rows.max() - green_rows.min() + 1
    assert width / height == pytest.approx(2.0, rel=0.08)


def test_single_image_composite_is_square():
    img = Image.new("RGB", (300, 150), (128, 128, 128))
    out = make_single_image(img, size=224)
    assert out.size == (224, 224)


# ---------------------------------------------------------------------------
# offline router
# ---------------------------------------------------------------------------


@pytest.fixture
def router():
    return LocalIntentRouter()


@pytest.mark.parametrize(
    "query,mode,count,expected",
    [
        ("Describe the land-cover and major objects visible in this image.", "single", 1, "caption"),
        ("Highlight the water body referred to in the query.", "single", 1, "grounding"),
        ("What changed between these two dates, and where did the change occur?", "bi_temporal", 2, "change_analysis"),
        ("Use the optical and SAR images together to identify built-up and water-covered regions.", "cross_modal", 2, "optical_sar_fusion"),
        ("Has the built-up area increased, decreased, or remained unchanged?", "bi_temporal", 2, "change_analysis"),
        ("How many aircraft are visible?", "single", 1, "vqa"),
        ("Is there a bridge in this image?", "single", 1, "vqa"),
    ],
)
def test_router_handles_every_representative_query(router, query, mode, count, expected):
    """These five queries are the problem statement's own examples."""
    cls = router.classify(query, {"image_count": count, "input_mode": mode})
    assert cls.task == expected


def test_router_never_selects_a_pair_task_with_one_image(router):
    cls = router.classify(
        "What changed between these two dates?", {"image_count": 1, "input_mode": "single"}
    )
    assert cls.task not in ("change_analysis", "optical_sar_fusion")


def test_grounding_always_carries_search_nouns(router):
    cls = router.classify(
        "Highlight the river and the road.", {"image_count": 1, "input_mode": "single"}
    )
    assert cls.task == "grounding"
    assert cls.search_nouns
    assert all(n.endswith(".") for n in cls.search_nouns)
    assert "river." in cls.search_nouns


def test_grounding_without_a_locatable_entity_falls_back(router):
    cls = router.classify("Highlight it.", {"image_count": 1, "input_mode": "single"})
    assert cls.task != "grounding"


def test_router_reports_a_backend_and_rationale(router):
    cls = router.classify("How many ships?", {"image_count": 1, "input_mode": "single"})
    assert cls.backend == "local-rules-v1"
    assert cls.rationale


def test_entity_extraction_prefers_longest_match():
    assert extract_entities("show the built-up area") == ["built-up area"]
    assert extract_entities("the water body next to the highway") == ["water body", "highway"]


def test_entity_extraction_requires_whole_words():
    # "tank" must not fire inside "tanker".
    assert "storage tank" not in extract_entities("a tanker truck")


# ---------------------------------------------------------------------------
# input validation / co-registration
# ---------------------------------------------------------------------------


@pytest.fixture
def validator():
    return SatQueryValidator()


def test_identical_grid_is_verified(tmp_path, validator, rng):
    data = rng.integers(0, 255, (3, 32, 32), dtype=np.uint8)
    a = write_raster(tmp_path / "a.tif", data, dtype="uint8")
    b = write_raster(tmp_path / "b.tif", data, dtype="uint8")
    res = validator.check_coregistration(a, b)
    assert res.status == CoregistrationStatus.IDENTICAL_GRID
    assert res.verified and res.ok_to_proceed


def test_sub_pixel_offset_is_still_verified(tmp_path, validator, rng):
    """
    The old check used atol=1e-4 on bounds, which in a projected CRS is 0.1 mm -
    it rejected genuinely co-registered Cartosat/RISAT pairs. Tolerance is now
    one pixel.
    """
    data = rng.integers(0, 255, (3, 32, 32), dtype=np.uint8)
    a = write_raster(tmp_path / "a.tif", data, origin=(500000.0, 2000000.0), res=10.0, dtype="uint8")
    b = write_raster(tmp_path / "b.tif", data, origin=(500003.0, 2000000.0), res=10.0, dtype="uint8")
    res = validator.check_coregistration(a, b)
    assert res.verified, res.message


def test_large_offset_requires_resampling_but_still_proceeds(tmp_path, validator, rng):
    data = rng.integers(0, 255, (3, 64, 64), dtype=np.uint8)
    a = write_raster(tmp_path / "a.tif", data, origin=(500000.0, 2000000.0), res=10.0, dtype="uint8")
    b = write_raster(tmp_path / "b.tif", data, origin=(500100.0, 2000000.0), res=10.0, dtype="uint8")
    res = validator.check_coregistration(a, b)
    assert res.status == CoregistrationStatus.RESAMPLING_REQUIRED
    assert res.ok_to_proceed and not res.verified


def test_disjoint_footprints_are_rejected(tmp_path, validator, rng):
    data = rng.integers(0, 255, (3, 32, 32), dtype=np.uint8)
    a = write_raster(tmp_path / "a.tif", data, origin=(500000.0, 2000000.0), dtype="uint8")
    b = write_raster(tmp_path / "b.tif", data, origin=(900000.0, 2500000.0), dtype="uint8")
    res = validator.check_coregistration(a, b)
    assert res.status == CoregistrationStatus.NO_OVERLAP
    assert not res.ok_to_proceed


def test_png_pair_is_not_claimed_as_verified(tmp_path, validator, rng):
    """
    The single most misleading behaviour of the old validator: any pair
    containing a non-georeferenced file returned True, so every demo displayed
    "Co-registration Verified" without a check having run.
    """
    for name in ("a.png", "b.png"):
        Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(tmp_path / name)
    res = validator.check_coregistration(str(tmp_path / "a.png"), str(tmp_path / "b.png"))
    assert res.status == CoregistrationStatus.NOT_GEOREFERENCED
    assert res.ok_to_proceed          # benchmark inputs are still usable
    assert not res.verified           # but nothing was verified
    assert "NOT verified" in res.message


def test_legacy_boolean_api_does_not_pass_unverified_pairs(tmp_path, validator, rng):
    for name in ("a.png", "b.png"):
        Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(tmp_path / name)
    assert validator.verify_coregistration(str(tmp_path / "a.png"), str(tmp_path / "b.png")) is False


def test_unreadable_file_is_reported(tmp_path, validator):
    bad = tmp_path / "broken.tif"
    bad.write_bytes(b"this is not a raster")
    meta = validator.extract_metadata(str(bad))
    assert not meta.is_valid
    assert meta.error
