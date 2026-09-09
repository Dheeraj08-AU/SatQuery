"""
Cross-modal optical + SAR evidence extraction.

The problem statement asks for a system that "must extract complementary
information from a co-registered optical/multispectral and SAR image pair",
and gives as a representative query: "Use the optical and SAR images together
to identify built-up and water-covered regions."

Handing two pictures to a VLM and asking it nicely does not extract
complementary information - it just looks at two pictures. This module adds the
part that actually uses the physics:

  * Water is a specular reflector at radar wavelengths. It scatters energy away
    from the sensor, so open water is the darkest thing in a SAR scene.
  * Buildings produce double-bounce (wall-ground corner reflection) and are
    among the brightest, with strong local texture.
  * Vegetation gives moderate volume scattering, between the two.

So a SAR scene alone separates water from built-up far more reliably than
optical does under cloud, and optical supplies the spectral context SAR lacks.
Combining the two masks is genuine cross-modal extraction.

Calibration caveat, recorded in every result
--------------------------------------------
Absolute sigma0 thresholds (water below about -18 dB for Sentinel-1 VV) require
radiometrically calibrated input. Uploaded GeoTIFFs may be uncalibrated DN, and
the display pipeline applies a per-image percentile stretch. This module
therefore classifies by RELATIVE percentile within the scene, which is robust
to calibration state but means the thresholds are scene-relative, not absolute.
The result labels this explicitly rather than implying calibrated physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from modules.raster_io import box_mean as _box_mean

# Percentile of SAR brightness below which a pixel is treated as specular
# (water-like), and above which as a strong scatterer (built-up-like).
DEFAULT_WATER_PCT = 15.0
DEFAULT_BUILTUP_PCT = 90.0

# Built-up areas are bright AND texturally rough. Smooth bright surfaces
# (bare rock, some bare soil) are rejected by this local-variation floor.
DEFAULT_TEXTURE_PCT = 55.0


@dataclass
class SarEvidence:
    water_mask: np.ndarray
    builtup_mask: np.ndarray
    water_fraction: float
    builtup_fraction: float
    overlay: Image.Image
    summary: str
    method: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def stats(self) -> Dict[str, Any]:
        return {
            "water_fraction": round(self.water_fraction, 4),
            "builtup_fraction": round(self.builtup_fraction, 4),
            "summary": self.summary,
            "method": self.method,
            "warnings": self.warnings,
        }


def _local_variation(gray: np.ndarray, radius: int = 2) -> np.ndarray:
    """Local standard deviation, as a texture proxy."""
    mean = _box_mean(gray, radius)
    sq = _box_mean(gray * gray, radius)
    return np.sqrt(np.maximum(sq - mean * mean, 0.0))


def analyse_optical_sar(
    optical: Image.Image,
    sar: Image.Image,
    water_pct: float = DEFAULT_WATER_PCT,
    builtup_pct: float = DEFAULT_BUILTUP_PCT,
    texture_pct: float = DEFAULT_TEXTURE_PCT,
    smooth_radius: int = 1,
) -> SarEvidence:
    """
    Derive water and built-up masks from the SAR channel, cross-checked
    against optical brightness.

    Both images must already be on a common grid (use raster_io.align_pair).
    """
    warnings: List[str] = []

    opt = optical.convert("RGB")
    rad = sar.convert("RGB")
    if opt.size != rad.size:
        warnings.append(
            f"Optical {opt.size} and SAR {rad.size} were not on a common grid; "
            f"SAR was resized. Misalignment will displace the masks."
        )
        rad = rad.resize(opt.size, Image.BILINEAR)

    # SAR was rendered to greyscale (or dual-pol false colour) by raster_io;
    # the luminance is monotonic in decibels, so percentiles on it are
    # equivalent to percentiles on dB.
    sar_gray = np.asarray(rad.convert("L"), dtype=np.float64)
    if smooth_radius > 0:
        sar_gray = _box_mean(sar_gray, smooth_radius)

    opt_gray = np.asarray(opt.convert("L"), dtype=np.float64)

    t_water = float(np.percentile(sar_gray, water_pct))
    t_bright = float(np.percentile(sar_gray, builtup_pct))

    texture = _local_variation(sar_gray, radius=2)
    t_texture = float(np.percentile(texture, texture_pct))

    water = sar_gray <= t_water
    builtup = (sar_gray >= t_bright) & (texture >= t_texture)

    # Optical cross-check: open water is dark in optical too. Pixels that are
    # radar-dark but optically bright are more likely radar shadow behind
    # terrain or a smooth road surface than open water.
    opt_water_ceiling = float(np.percentile(opt_gray, 60.0))
    radar_dark_optically_bright = water & (opt_gray > opt_water_ceiling)
    rejected = int(radar_dark_optically_bright.sum())
    water = water & ~radar_dark_optically_bright

    if rejected:
        warnings.append(
            f"{rejected:,} radar-dark pixels were rejected as water because they "
            f"are bright in the optical image (likely radar shadow or smooth "
            f"pavement rather than open water)."
        )

    water_fraction = float(water.mean())
    builtup_fraction = float(builtup.mean())

    overlay = _build_overlay(opt, water, builtup)
    summary = _summarise(water_fraction, builtup_fraction)

    method = {
        "algorithm": "relative SAR backscatter percentile classification with optical cross-check",
        "physics": (
            "water = specular scattering (radar-dark); built-up = double-bounce "
            "(radar-bright) with high local texture"
        ),
        "water_percentile": water_pct,
        "builtup_percentile": builtup_pct,
        "texture_percentile": texture_pct,
        "thresholds_applied": {
            "sar_dark_le": round(t_water, 2),
            "sar_bright_ge": round(t_bright, 2),
            "texture_ge": round(t_texture, 3),
            "optical_water_ceiling": round(opt_water_ceiling, 2),
        },
        "calibration": (
            "scene-relative percentiles, NOT calibrated sigma0 thresholds - "
            "absolute dB cut-offs would require radiometrically calibrated input"
        ),
        "optical_cross_check_rejected_px": rejected,
    }

    return SarEvidence(
        water_mask=water,
        builtup_mask=builtup,
        water_fraction=water_fraction,
        builtup_fraction=builtup_fraction,
        overlay=overlay,
        summary=summary,
        method=method,
        warnings=warnings,
    )


def _summarise(water_fraction: float, builtup_fraction: float) -> str:
    parts = []
    if water_fraction >= 0.005:
        parts.append(f"water-covered regions occupy {water_fraction:.1%} of the scene")
    else:
        parts.append("no significant open water detected")
    if builtup_fraction >= 0.005:
        parts.append(f"built-up regions occupy {builtup_fraction:.1%}")
    else:
        parts.append("no significant built-up signature detected")
    return (
        "SAR backscatter analysis: "
        + "; ".join(parts)
        + ". Water identified from specular (dark) returns, built-up from "
        "bright, high-texture double-bounce returns."
    )


def _build_overlay(
    base: Image.Image, water: np.ndarray, builtup: np.ndarray, alpha: float = 0.45
) -> Image.Image:
    """Blue = water, red = built-up, drawn over the optical image."""
    rgb = np.asarray(base.convert("RGB"), dtype=np.float64)

    tint = np.zeros_like(rgb)
    tint[water] = (30.0, 120.0, 255.0)
    tint[builtup] = (255.0, 60.0, 40.0)

    m = ((water | builtup)[..., None]).astype(np.float64) * alpha
    blended = rgb * (1.0 - m) + tint * m
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


__all__ = ["analyse_optical_sar", "SarEvidence"]
