"""
Bi-temporal change detection for SatQuery AI.

Replaces `draw_mock_change_map`, which was `abs(gray(t1) - gray(t2)) > 40`
rendered as red on black. That has three fatal problems: a fixed threshold in
8-bit units is meaningless across scenes; no radiometric normalisation means
seasonal or illumination differences light up the entire frame; and the output
was a floating mask with no imagery under it, so a viewer could not tell where
anything was.

Method implemented here
-----------------------
1.  Relative radiometric normalisation. T2 is linearly rescaled onto T1's
    radiometry per channel, anchored on the interquartile range (p25/p75).
    Quartile anchors are robust to up to 25% changed pixels, so the fit is
    driven by the unchanged background - which is exactly what should
    be matched.
2.  Change Vector Analysis. Per-pixel magnitude is the Euclidean norm of the
    spectral difference vector, normalised to [0, 1].
3.  Otsu thresholding, so the decision boundary comes from the data rather
    than from a hardcoded 40.
4.  Morphological opening then closing to drop speckle and fill pinholes.
5.  Connected-component analysis to report WHERE the change is, which is half
    of the problem statement's question ("what changed, and where").

Honest limitations, recorded in the result's `method` dict:
  * Otsu always returns a split, even for an unchanged pair. Separability is
    computed alongside it and a low value is reported as weak evidence.
  * Detection runs on the display-stretched RGB rendering, not raw radiance.
    Per-image percentile stretching is itself a normalisation, which is why
    step 1 is a relative fit rather than an absolute calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from modules.raster_io import box_mean as _box_mean

# A change must exceed this magnitude regardless of what Otsu returns. Without
# a floor, an unchanged pair yields a threshold near zero and reports the whole
# scene as changed.
MIN_MAGNITUDE_THRESHOLD = 0.05

# Below this Otsu separability the two modes are not meaningfully distinct.
WEAK_SEPARABILITY = 0.40

# Above this changed fraction the result is more likely a global radiometric
# or geometric mismatch than real change.
IMPLAUSIBLE_CHANGE_FRACTION = 0.60

# Connected-component labelling runs on a mask downsampled to at most this
# edge length, which bounds the cost of the pure-Python flood fill.
LABEL_MAX_EDGE = 256


@dataclass
class ChangeRegion:
    bbox: Tuple[int, int, int, int]      # x0, y0, x1, y1 in full-resolution pixels
    area_px: int
    area_fraction: float
    centroid: Tuple[int, int]
    direction: str                        # "north-east", "centre", ...


@dataclass
class ChangeDetectionResult:
    mask: np.ndarray
    magnitude: np.ndarray
    overlay: Image.Image
    changed_fraction: float
    threshold: float
    separability: float
    regions: List[ChangeRegion]
    summary: str
    method: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def stats(self) -> Dict[str, Any]:
        """Serialisable subset, for the execution trace and report export."""
        return {
            "changed_fraction": round(self.changed_fraction, 4),
            "threshold": round(self.threshold, 4),
            "otsu_separability": round(self.separability, 4),
            "region_count": len(self.regions),
            "regions": [
                {
                    "bbox": list(r.bbox),
                    "area_px": r.area_px,
                    "area_fraction": round(r.area_fraction, 4),
                    "centroid": list(r.centroid),
                    "direction": r.direction,
                }
                for r in self.regions
            ],
            "summary": self.summary,
            "method": self.method,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def otsu_threshold(values: np.ndarray, nbins: int = 256) -> Tuple[float, float]:
    """
    Otsu's threshold over [0, 1], plus the separability metric eta =
    between-class variance / total variance. eta near 1 means two well
    separated modes; near 0 means the histogram is effectively unimodal and
    the threshold is arbitrary.
    """
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0, 0.0

    hist, edges = np.histogram(v, bins=nbins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0, 0.0

    p = hist / total
    centers = (edges[:-1] + edges[1:]) / 2.0
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]

    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    sigma_b2 = np.where(np.isfinite(sigma_b2), sigma_b2, 0.0)

    idx = int(np.argmax(sigma_b2))
    total_var = float(np.sum(p * (centers - mu_t) ** 2))
    separability = float(sigma_b2[idx] / total_var) if total_var > 1e-12 else 0.0
    return float(centers[idx]), separability


def _binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1:
        return mask
    return _box_mean(mask.astype(np.float64), radius) >= 1.0 - 1e-9


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1:
        return mask
    return _box_mean(mask.astype(np.float64), radius) > 1e-9


def binary_open(mask: np.ndarray, radius: int) -> np.ndarray:
    """Erode then dilate: removes isolated speckle."""
    return _binary_dilate(_binary_erode(mask, radius), radius)


def binary_close(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate then erode: fills pinholes inside regions."""
    return _binary_erode(_binary_dilate(mask, radius), radius)


def radiometric_normalize(
    reference: np.ndarray, target: np.ndarray
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Linearly rescale `target` onto `reference`'s radiometry, per channel.

    Anchored on the interquartile range rather than fitted by least squares: a
    least-squares fit is pulled by the changed pixels, which are precisely the
    signal we are trying to preserve.

    p25/p75 rather than p2/p98. Tail percentiles look more informative but are
    not robust here - a change region covering 10% of the scene at an extreme
    brightness sits inside the 98th percentile, so the anchor is computed from
    the change itself and the fit drags the unchanged background off by tens of
    DN. The interquartile range tolerates up to 25% contamination in either
    tail, which comfortably covers realistic change extents.
    """
    out = np.empty_like(target, dtype=np.float64)
    params: List[Tuple[float, float]] = []

    for c in range(reference.shape[2]):
        ref_c = reference[..., c]
        tgt_c = target[..., c]
        r_lo, r_hi = np.percentile(ref_c, [25.0, 75.0])
        t_lo, t_hi = np.percentile(tgt_c, [25.0, 75.0])

        if (t_hi - t_lo) < 1e-6:
            a, b = 1.0, 0.0
        else:
            a = float((r_hi - r_lo) / (t_hi - t_lo))
            b = float(r_lo - a * t_lo)

        out[..., c] = a * tgt_c + b
        params.append((round(a, 5), round(b, 3)))

    return np.clip(out, 0.0, 255.0), params


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------


def _label_regions(mask: np.ndarray, max_edge: int = LABEL_MAX_EDGE) -> List[ChangeRegion]:
    """
    4-connected components on a downsampled copy of the mask, with bounding
    boxes rescaled back to full resolution.

    Downsampling bounds the pure-Python flood fill at ~65k cells, which keeps
    this fast without pulling in scipy for one function.
    """
    h, w = mask.shape
    if h == 0 or w == 0 or not mask.any():
        return []

    scale = max(1, int(np.ceil(max(h, w) / max_edge)))
    if scale > 1:
        sh, sw = h // scale, w // scale
        if sh == 0 or sw == 0:
            small = mask
            scale = 1
        else:
            trimmed = mask[: sh * scale, : sw * scale]
            small = trimmed.reshape(sh, scale, sw, scale).any(axis=(1, 3))
    else:
        small = mask

    sh, sw = small.shape
    visited = np.zeros((sh, sw), dtype=bool)
    total_px = int(mask.sum())
    regions: List[ChangeRegion] = []

    for y0 in range(sh):
        for x0 in range(sw):
            if not small[y0, x0] or visited[y0, x0]:
                continue

            stack = [(y0, x0)]
            visited[y0, x0] = True
            cells: List[Tuple[int, int]] = []

            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < sh and 0 <= nx < sw and small[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            ys = [c[0] for c in cells]
            xs = [c[1] for c in cells]
            x0f = int(min(xs) * scale)
            y0f = int(min(ys) * scale)
            x1f = int(min((max(xs) + 1) * scale, w))
            y1f = int(min((max(ys) + 1) * scale, h))

            sub = mask[y0f:y1f, x0f:x1f]
            area_px = int(sub.sum())
            if area_px == 0:
                continue

            cy = int(np.mean(ys) * scale)
            cx = int(np.mean(xs) * scale)
            regions.append(
                ChangeRegion(
                    bbox=(x0f, y0f, x1f, y1f),
                    area_px=area_px,
                    area_fraction=area_px / float(h * w),
                    centroid=(cx, cy),
                    direction=_describe_position(cx, cy, w, h),
                )
            )

    regions.sort(key=lambda r: r.area_px, reverse=True)
    _ = total_px
    return regions


def _describe_position(cx: int, cy: int, w: int, h: int) -> str:
    """Plain-language position of a point within the frame."""
    third_w, third_h = w / 3.0, h / 3.0
    col = 0 if cx < third_w else (1 if cx < 2 * third_w else 2)
    row = 0 if cy < third_h else (1 if cy < 2 * third_h else 2)
    if row == 1 and col == 1:
        return "centre"
    vertical = ("northern", "", "southern")[row]
    horizontal = ("western", "", "eastern")[col]
    parts = [p for p in (vertical, horizontal) if p]
    return " ".join(parts) if parts else "centre"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def detect_change(
    img_t1: Image.Image,
    img_t2: Image.Image,
    open_radius: int = 1,
    close_radius: int = 2,
    min_region_fraction: float = 0.0005,
    max_regions: int = 8,
    smoothing_radius: int = 1,
) -> ChangeDetectionResult:
    """
    Detect change between two aligned RGB renderings.

    Both images must already be on a common grid - use
    `raster_io.align_pair` first. If sizes still differ, T2 is resized as a
    last resort and a warning is recorded.
    """
    warnings: List[str] = []

    a = img_t1.convert("RGB")
    b = img_t2.convert("RGB")
    if a.size != b.size:
        warnings.append(
            f"Images were not on a common grid ({a.size} vs {b.size}); "
            f"T2 was resized. Alignment errors will appear as false change."
        )
        b = b.resize(a.size, Image.BILINEAR)

    arr1 = np.asarray(a, dtype=np.float64)
    arr2 = np.asarray(b, dtype=np.float64)

    # 1. relative radiometric normalisation
    arr2n, norm_params = radiometric_normalize(arr1, arr2)

    # 2. change vector magnitude, normalised to [0, 1]
    diff = arr2n - arr1
    magnitude = np.sqrt(np.mean(diff ** 2, axis=2)) / 255.0
    magnitude = np.clip(magnitude, 0.0, 1.0)

    if smoothing_radius > 0:
        magnitude = _box_mean(magnitude, smoothing_radius)

    # 3. Otsu
    t_otsu, separability = otsu_threshold(magnitude)
    threshold = max(t_otsu, MIN_MAGNITUDE_THRESHOLD)

    mask = magnitude > threshold

    # 4. morphology
    if open_radius > 0:
        mask = binary_open(mask, open_radius)
    if close_radius > 0:
        mask = binary_close(mask, close_radius)

    changed_fraction = float(mask.mean())

    if separability < WEAK_SEPARABILITY:
        warnings.append(
            f"Weak bimodality (Otsu separability {separability:.2f} < "
            f"{WEAK_SEPARABILITY}); the change signal is not clearly separated "
            f"from background variation. Treat the map as low confidence."
        )
    if changed_fraction > IMPLAUSIBLE_CHANGE_FRACTION:
        warnings.append(
            f"{changed_fraction:.0%} of the scene is flagged as changed. That is "
            f"more consistent with a residual radiometric or geometric mismatch "
            f"between the two acquisitions than with real land-cover change."
        )

    # 5. regions
    regions = [
        r for r in _label_regions(mask) if r.area_fraction >= min_region_fraction
    ][:max_regions]

    summary = _build_summary(changed_fraction, regions, separability)
    overlay = _build_overlay(b, mask)

    method = {
        "algorithm": "relative radiometric normalisation -> change vector analysis -> Otsu -> morphology",
        "normalisation": "interquartile-anchored linear (p25/p75) per channel",
        "normalisation_params_per_channel": norm_params,
        "otsu_threshold": round(t_otsu, 4),
        "applied_threshold": round(threshold, 4),
        "min_threshold_floor": MIN_MAGNITUDE_THRESHOLD,
        "otsu_separability": round(separability, 4),
        "morphology": {"open_radius": open_radius, "close_radius": close_radius},
        "smoothing_radius": smoothing_radius,
        "min_region_fraction": min_region_fraction,
    }

    return ChangeDetectionResult(
        mask=mask,
        magnitude=magnitude,
        overlay=overlay,
        changed_fraction=changed_fraction,
        threshold=threshold,
        separability=separability,
        regions=regions,
        summary=summary,
        method=method,
        warnings=warnings,
    )


def _build_summary(
    changed_fraction: float, regions: List[ChangeRegion], separability: float
) -> str:
    if changed_fraction < 0.005:
        return (
            "No significant change detected: less than 0.5% of the scene differs "
            "after radiometric normalisation."
        )

    pct = f"{changed_fraction:.1%}"
    if not regions:
        return f"Change affects {pct} of the scene, scattered rather than concentrated."

    biggest = regions[0]
    if len(regions) == 1:
        where = f"in the {biggest.direction} of the scene"
    else:
        others = ", ".join(sorted({r.direction for r in regions[1:4]}))
        where = (
            f"largest in the {biggest.direction} of the scene, with further "
            f"activity to the {others}"
        )

    confidence = "" if separability >= WEAK_SEPARABILITY else " (low confidence)"
    return (
        f"Change affects {pct} of the scene across {len(regions)} region"
        f"{'s' if len(regions) != 1 else ''}, {where}{confidence}."
    )


def _build_overlay(base: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    Semi-transparent red change mask over the T2 image.

    Rendering over the imagery rather than on a black field is the point: a
    floating mask cannot be checked against the ground by a human reviewer.
    """
    rgb = np.asarray(base.convert("RGB"), dtype=np.float64)
    red = np.zeros_like(rgb)
    red[..., 0] = 255.0
    m = mask[..., None].astype(np.float64) * alpha
    blended = rgb * (1.0 - m) + red * m
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


__all__ = [
    "detect_change",
    "ChangeDetectionResult",
    "ChangeRegion",
    "otsu_threshold",
    "radiometric_normalize",
    "binary_open",
    "binary_close",
]
