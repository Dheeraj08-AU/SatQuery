"""
Raster I/O layer for SatQuery AI.

Every pixel read in the system goes through this module. It replaces the
previous `PIL.Image.open(path).convert("RGB")` calls, which cannot read
multispectral GeoTIFF (12-band uint16 Sentinel-2) or SAR (single-band
float32 / uint16 with very wide dynamic range) and silently produced
black or mangled frames.

Design rules
------------
1.  Nothing here guesses silently. Every decision (which bands were used,
    which stretch percentiles, whether dB conversion or speckle filtering
    was applied) is recorded in a `RasterLoadReport` that is surfaced in the
    agent's auditable execution trace.
2.  Optical and SAR take different paths. SAR is converted to decibels and
    speckle-filtered before stretching; doing a naive 2-98% stretch on raw
    SAR intensity yields an almost-black image because the intensity
    distribution is heavy-tailed.
3.  Failures raise. A caller that hands us an unreadable file gets an
    exception, not a blank image.

Public API
----------
    load_as_rgb(path, modality=...)      -> (PIL.Image RGB, RasterLoadReport)
    read_raster_meta(path)               -> dict
    resample_to_match(src, ref)          -> (np.ndarray, dict)
    align_pair(a, b, ...)                -> (PIL a, PIL b, dict)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject
    from rasterio.enums import Resampling

    _HAS_RASTERIO = True
except Exception:  # pragma: no cover - rasterio is a hard dependency in practice
    _HAS_RASTERIO = False


# ---------------------------------------------------------------------------
# Band layouts
# ---------------------------------------------------------------------------

# 1-based band indices for (Red, Green, Blue), keyed by band count.
# Sentinel-2 / BigEarthNet orderings:
#   12-band: B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
#   13-band: B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B10 B11 B12
#   10-band (BigEarthNet v2 10m+20m): B02 B03 B04 B05 B06 B07 B08 B8A B11 B12
# In all three, true colour is (B04, B03, B02).
DEFAULT_RGB_BY_COUNT: Dict[int, Tuple[int, int, int]] = {
    3: (1, 2, 3),        # already RGB
    4: (3, 2, 1),        # B,G,R,NIR  (Cartosat-2S MX, most 4-band products)
    10: (3, 2, 1),       # B02,B03,B04,...
    12: (4, 3, 2),       # B01,B02,B03,B04,...
    13: (4, 3, 2),       # B01,B02,B03,B04,... (with B10)
}

# Canonical Sentinel-2 names we look for in GDAL band descriptions.
_S2_RGB_NAMES = ("B04", "B03", "B02")

_EPS = 1e-10

# Percentile stretch is computed on at most this many pixels (subsampled)
# so that large scenes do not blow up memory or wall-clock.
_MAX_STRETCH_SAMPLES = 4_000_000


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class RasterLoadReport:
    """Provenance for one raster read. Surfaced in the agent execution trace."""

    path: str
    driver: str
    modality: str                       # "optical" | "sar" | "unknown"
    band_count: int
    bands_used: Optional[Tuple[int, ...]] = None
    band_descriptions: Optional[List[str]] = None
    dtype: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    crs: Optional[str] = None
    bounds: Optional[Tuple[float, float, float, float]] = None
    resolution: Optional[Tuple[float, float]] = None
    nodata: Optional[float] = None
    nodata_fraction: float = 0.0
    stretch_percentiles: Optional[Tuple[float, float]] = None
    stretch_values: Optional[List[Tuple[float, float]]] = None
    db_conversion: Optional[str] = None   # "10log10" | "20log10" | "already_db" | None
    db_clip: Optional[Tuple[float, float]] = None
    speckle_filter: Optional[str] = None  # "lee(radius=2)" | None
    resized_to: Optional[Tuple[int, int]] = None
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        """One-line human summary for the UI."""
        parts = [f"{self.driver} {self.width}x{self.height} {self.band_count}b {self.dtype}"]
        if self.bands_used:
            parts.append(f"bands={self.bands_used}")
        if self.db_conversion:
            parts.append(f"dB={self.db_conversion}")
        if self.speckle_filter:
            parts.append(self.speckle_filter)
        if self.stretch_percentiles:
            lo, hi = self.stretch_percentiles
            parts.append(f"stretch={lo:g}-{hi:g}%")
        return " | ".join(parts)


class RasterReadError(RuntimeError):
    """Raised when a file cannot be read as imagery."""


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _box_mean(a: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a (2*radius+1)^2 window, via integral image. Edge-reflected."""
    if radius < 1:
        return a
    k = 2 * radius + 1
    h, w = a.shape
    ap = np.pad(a, radius, mode="reflect")
    ii = ap.cumsum(0).cumsum(1)
    ii = np.pad(ii, ((1, 0), (1, 0)), mode="constant")
    s = (
        ii[k : k + h, k : k + w]
        - ii[0:h, k : k + w]
        - ii[k : k + h, 0:w]
        + ii[0:h, 0:w]
    )
    return s / float(k * k)


def lee_filter(img: np.ndarray, radius: int = 2, cu: float = 0.523) -> np.ndarray:
    """
    Lee speckle filter for SAR amplitude/intensity.

    `cu` is the noise coefficient of variation. 0.523 corresponds to
    single-look amplitude; use ~1/sqrt(L) for L-look intensity.
    Operates on the linear-power image, before dB conversion.
    """
    img = img.astype(np.float64, copy=False)
    mean = _box_mean(img, radius)
    sqr_mean = _box_mean(img * img, radius)
    var = np.maximum(sqr_mean - mean * mean, 0.0)
    ci2 = var / np.maximum(mean * mean, _EPS)
    weight = np.clip(1.0 - (cu * cu) / np.maximum(ci2, _EPS), 0.0, 1.0)
    return mean + weight * (img - mean)


def _sample_for_percentile(values: np.ndarray) -> np.ndarray:
    if values.size <= _MAX_STRETCH_SAMPLES:
        return values
    step = int(np.ceil(values.size / _MAX_STRETCH_SAMPLES))
    return values[::step]


def percentile_stretch(
    band: np.ndarray,
    valid: Optional[np.ndarray] = None,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Linear stretch between two percentiles -> uint8. Returns (uint8, (lo, hi))."""
    band = band.astype(np.float64, copy=False)
    finite = np.isfinite(band)
    mask = finite if valid is None else (finite & valid)

    if not mask.any():
        return np.zeros(band.shape, dtype=np.uint8), (0.0, 0.0)

    sample = _sample_for_percentile(band[mask].ravel())
    lo, hi = np.percentile(sample, [lo_pct, hi_pct])
    lo = float(lo)
    hi = float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # Degenerate (constant band, or percentiles collapsed).
        lo = float(np.nanmin(sample)) if sample.size else 0.0
        hi = float(np.nanmax(sample)) if sample.size else 0.0
        if hi <= lo:
            hi = lo + 1.0

    out = (band - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~finite] = 0.0
    return (out * 255.0).round().astype(np.uint8), (lo, hi)


def to_decibels(
    arr: np.ndarray,
    is_integer_dn: bool,
    clip: Tuple[float, float] = (-30.0, 5.0),
) -> Tuple[np.ndarray, str, Tuple[float, float]]:
    """
    Convert SAR data to decibels.

    Integer DN products (e.g. RISAT amplitude) use 20*log10(DN).
    Floating-point sigma0/gamma0 power products use 10*log10(power).
    Data that already contains negatives is assumed to be in dB already.

    Returns (dB array, conversion label, clip range actually applied).
    """
    arr = arr.astype(np.float64, copy=False)
    finite = np.isfinite(arr)

    if finite.any() and np.nanmin(arr[finite]) < 0:
        db = arr
        label = "already_db"
    elif is_integer_dn:
        db = 20.0 * np.log10(np.maximum(arr, _EPS))
        label = "20log10"
    else:
        db = 10.0 * np.log10(np.maximum(arr, _EPS))
        label = "10log10"

    db = np.where(finite, db, np.nan)

    # Clip to a sane backscatter window, but only if the data actually spans it.
    lo, hi = clip
    if np.isfinite(db).any():
        data_lo = float(np.nanpercentile(_sample_for_percentile(db[np.isfinite(db)].ravel()), 1))
        data_hi = float(np.nanpercentile(_sample_for_percentile(db[np.isfinite(db)].ravel()), 99))
        lo = max(lo, data_lo - 5.0)
        hi = min(hi, data_hi + 5.0)
        if hi <= lo:
            lo, hi = data_lo, max(data_hi, data_lo + 1.0)
    db = np.clip(db, lo, hi)
    return db, label, (lo, hi)


# ---------------------------------------------------------------------------
# Band selection
# ---------------------------------------------------------------------------


def _descriptions(src) -> List[str]:
    try:
        return [d if d else "" for d in (src.descriptions or [])]
    except Exception:
        return []


def _colorinterp(src) -> List[str]:
    """GDAL colour interpretation per band, lowercased ('red', 'gray', ...)."""
    try:
        return [getattr(c, "name", str(c)).lower() for c in (src.colorinterp or [])]
    except Exception:
        return []


def choose_rgb_bands(
    band_count: int,
    descriptions: Sequence[str],
    colorinterp: Sequence[str] = (),
) -> Tuple[Optional[Tuple[int, int, int]], List[str]]:
    """
    Pick 1-based (R, G, B) band indices.

    Priority order:
      1. GDAL colour interpretation. This is what makes RGB/RGBA PNG and JPEG
         inputs correct. Without it, a 4-band RGBA PNG hits the 4-band layout
         table below, which assumes B,G,R,NIR and therefore swaps red and blue.
      2. Band descriptions naming Sentinel-2 bands.
      3. Band-count layout table.

    Returns (indices, warnings); indices is None when the raster should be
    treated as single-band.
    """
    warnings: List[str] = []

    # 1. Colour interpretation (PNG/JPEG, and any GeoTIFF that sets it).
    if colorinterp:
        ci = [str(c).lower() for c in colorinterp]
        try:
            r = ci.index("red") + 1
            g = ci.index("green") + 1
            b = ci.index("blue") + 1
            return (r, g, b), warnings
        except ValueError:
            if "gray" in ci or "grey" in ci:
                return None, warnings

    # 2. Descriptions naming Sentinel-2 bands.
    if descriptions:
        upper = [d.upper() for d in descriptions]
        found: List[int] = []
        for name in _S2_RGB_NAMES:
            hit = next(
                (i + 1 for i, d in enumerate(upper) if d == name or d.endswith("_" + name)),
                None,
            )
            if hit is None:
                found = []
                break
            found.append(hit)
        if len(found) == 3:
            return (found[0], found[1], found[2]), warnings

    # 2. Layout table.
    if band_count in DEFAULT_RGB_BY_COUNT:
        return DEFAULT_RGB_BY_COUNT[band_count], warnings

    if band_count <= 2:
        return None, warnings

    warnings.append(
        f"Unrecognised {band_count}-band layout; falling back to bands (1,2,3). "
        "Pass bands=(r,g,b) explicitly if this is wrong."
    )
    return (1, 2, 3), warnings


# ---------------------------------------------------------------------------
# Modality detection
# ---------------------------------------------------------------------------


def infer_modality(band_count: int, dtype: str, path: str) -> str:
    """
    Best-effort modality guess. Callers that know the modality (the UI does:
    the user picks the Optical and SAR upload slots) should pass it explicitly
    rather than relying on this.
    """
    name = os.path.basename(path).lower()
    sar_hints = ("sar", "risat", "s1a", "s1b", "sentinel1", "sentinel-1", "grd", "slc", "_vv", "_vh")
    if any(h in name for h in sar_hints):
        return "sar"
    opt_hints = ("optical", "cartosat", "s2a", "s2b", "sentinel2", "sentinel-2", "msi", "resourcesat")
    if any(h in name for h in opt_hints):
        return "optical"
    if band_count == 1 and dtype in ("float32", "float64", "complex64", "complex128"):
        return "sar"
    if band_count >= 3:
        return "optical"
    return "unknown"


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------


def read_raster_meta(path: str) -> Dict[str, Any]:
    """Metadata only. Never raises; returns {'readable': False, 'error': ...} on failure."""
    if not _HAS_RASTERIO:
        return {"readable": False, "error": "rasterio is not installed"}
    try:
        with rasterio.open(path) as src:
            return {
                "readable": True,
                "driver": src.driver,
                "width": src.width,
                "height": src.height,
                "band_count": src.count,
                "dtype": str(src.dtypes[0]) if src.dtypes else None,
                "descriptions": _descriptions(src),
                "colorinterp": _colorinterp(src),
                "crs": src.crs.to_string() if src.crs else None,
                "bounds": tuple(src.bounds) if src.crs else None,
                "resolution": tuple(src.res) if src.crs else None,
                "nodata": src.nodata,
                "georeferenced": bool(src.crs),
            }
    except Exception as exc:
        return {"readable": False, "error": str(exc)}


def _read_bands(src, indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Read given 1-based band indices as float64, plus a validity mask."""
    raw = src.read(list(indices))
    # Complex must be collapsed to amplitude BEFORE the float cast - casting a
    # complex array straight to float64 discards the imaginary part and only
    # emits a warning. Single-look-complex SAR would silently lose half its
    # information.
    arr = np.abs(raw).astype(np.float64) if np.iscomplexobj(raw) else raw.astype(np.float64)

    valid = np.ones(arr.shape[1:], dtype=bool)
    nodata = src.nodata
    if nodata is not None:
        valid &= ~np.all(np.isclose(arr, nodata), axis=0)
    valid &= np.all(np.isfinite(arr), axis=0)
    return arr, valid


def load_as_rgb(
    path: str,
    modality: str = "auto",
    bands: Optional[Tuple[int, int, int]] = None,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
    speckle_radius: int = 2,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Image.Image, RasterLoadReport]:
    """
    Read any supported raster as a display-ready 8-bit RGB PIL image.

    Parameters
    ----------
    path : str
        GeoTIFF/TIFF, or PNG/JPEG (benchmark inputs).
    modality : {"auto", "optical", "sar"}
        Pass explicitly whenever the caller knows. "sar" triggers dB
        conversion + Lee speckle filtering.
    bands : (r, g, b), 1-based, optional
        Overrides automatic band selection.
    target_size : (width, height), optional
        Resize after stretching.

    Returns
    -------
    (PIL.Image RGB, RasterLoadReport)

    Raises
    ------
    RasterReadError
        If the file cannot be read at all.
    """
    if not os.path.exists(path):
        raise RasterReadError(f"File does not exist: {path}")

    if not _HAS_RASTERIO:
        raise RasterReadError(
            "rasterio is required for image loading. Install it with `pip install rasterio`."
        )

    try:
        src_ctx = rasterio.open(path)
    except Exception as exc:
        raise RasterReadError(f"rasterio could not open {path}: {exc}") from exc

    with src_ctx as src:
        descriptions = _descriptions(src)
        colorinterp = _colorinterp(src)
        dtype = str(src.dtypes[0]) if src.dtypes else "unknown"
        is_integer_dn = np.issubdtype(np.dtype(dtype), np.integer) if dtype != "unknown" else False

        report = RasterLoadReport(
            path=path,
            driver=src.driver,
            modality=modality,
            band_count=src.count,
            band_descriptions=descriptions or None,
            dtype=dtype,
            width=src.width,
            height=src.height,
            crs=src.crs.to_string() if src.crs else None,
            bounds=tuple(src.bounds) if src.crs else None,
            resolution=tuple(src.res) if src.crs else None,
            nodata=src.nodata,
            stretch_percentiles=(lo_pct, hi_pct),
        )

        if modality == "auto":
            report.modality = infer_modality(src.count, dtype, path)
        resolved_modality = report.modality

        # -- band selection -------------------------------------------------
        if bands is not None:
            if any(b < 1 or b > src.count for b in bands):
                raise RasterReadError(
                    f"Requested bands {bands} out of range for a {src.count}-band raster."
                )
            chosen: Optional[Tuple[int, ...]] = tuple(bands)
        elif resolved_modality == "sar":
            # SAR: 1 band -> greyscale; 2 bands -> dual-pol false colour.
            chosen = (1,) if src.count == 1 else (1, 2)
            if src.count > 2:
                chosen = (1, 2)
                report.warnings.append(
                    f"SAR raster has {src.count} bands; using the first two as (co-pol, cross-pol)."
                )
        else:
            picked, warns = choose_rgb_bands(src.count, descriptions, colorinterp)
            report.warnings.extend(warns)
            chosen = picked if picked is not None else (1,)

        report.bands_used = tuple(chosen)

        arr, valid = _read_bands(src, chosen)

    total = valid.size
    report.nodata_fraction = float(1.0 - (valid.sum() / total)) if total else 1.0
    if report.nodata_fraction > 0.9:
        report.warnings.append(
            f"{report.nodata_fraction:.0%} of pixels are nodata/non-finite."
        )

    stretch_values: List[Tuple[float, float]] = []

    # -- SAR path -----------------------------------------------------------
    if resolved_modality == "sar":
        planes: List[np.ndarray] = []
        for i in range(arr.shape[0]):
            band = arr[i]
            if speckle_radius and speckle_radius > 0:
                filled = np.where(np.isfinite(band), band, 0.0)
                band = lee_filter(filled, radius=speckle_radius)
                band = np.where(valid, band, np.nan)
                report.speckle_filter = f"lee(radius={speckle_radius})"
            db, label, clip = to_decibels(band, is_integer_dn=is_integer_dn)
            report.db_conversion = label
            report.db_clip = clip
            u8, sv = percentile_stretch(db, valid, lo_pct, hi_pct)
            stretch_values.append(sv)
            planes.append(u8)

        if len(planes) == 1:
            rgb = np.stack([planes[0]] * 3, axis=-1)
        else:
            # Dual-pol false colour: R = co-pol, G = cross-pol, B = co/cross ratio.
            co = planes[0].astype(np.float64)
            cross = planes[1].astype(np.float64)
            ratio = np.clip(co - cross + 128.0, 0, 255).astype(np.uint8)
            rgb = np.stack([planes[0], planes[1], ratio], axis=-1)
            report.warnings.append(
                "Dual-pol false colour: R=co-pol dB, G=cross-pol dB, B=co-cross difference."
            )

    # -- optical path -------------------------------------------------------
    else:
        planes = []
        for i in range(arr.shape[0]):
            u8, sv = percentile_stretch(arr[i], valid, lo_pct, hi_pct)
            stretch_values.append(sv)
            planes.append(u8)
        if len(planes) == 1:
            rgb = np.stack([planes[0]] * 3, axis=-1)
        elif len(planes) == 2:
            rgb = np.stack([planes[0], planes[1], planes[0]], axis=-1)
        else:
            rgb = np.stack(planes[:3], axis=-1)

    report.stretch_values = stretch_values

    img = Image.fromarray(rgb, mode="RGB")

    if target_size is not None:
        img = img.resize(target_size, Image.BILINEAR)
        report.resized_to = tuple(target_size)

    return img, report


# ---------------------------------------------------------------------------
# Pair alignment
# ---------------------------------------------------------------------------


def resample_to_match(src_path: str, ref_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reproject/resample `src_path` onto the exact grid (CRS, transform, shape)
    of `ref_path`. Both must be georeferenced.

    Returns (array of shape (bands, ref_h, ref_w), info dict).
    """
    if not _HAS_RASTERIO:
        raise RasterReadError("rasterio is required for resampling.")

    with rasterio.open(ref_path) as ref, rasterio.open(src_path) as src:
        if not ref.crs or not src.crs:
            raise RasterReadError(
                "resample_to_match requires both rasters to be georeferenced."
            )

        dst = np.zeros((src.count, ref.height, ref.width), dtype=np.float64)
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=dst[i - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )

        info = {
            "resampled_from_crs": src.crs.to_string(),
            "resampled_to_crs": ref.crs.to_string(),
            "resampled_from_shape": (src.height, src.width),
            "resampled_to_shape": (ref.height, ref.width),
            "resampling": "bilinear",
        }
    return dst, info


def align_pair(
    path_a: str,
    path_b: str,
    modality_a: str = "auto",
    modality_b: str = "auto",
    **load_kwargs: Any,
) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
    """
    Load two rasters as RGB on a common grid.

    If both are georeferenced and their grids differ, `b` is reprojected onto
    `a`'s grid via `resample_to_match` before stretching. Otherwise `b` is
    simply resized to `a`'s pixel dimensions (the benchmark PNG/JPEG case).

    Returns (img_a, img_b, alignment info dict).
    """
    meta_a = read_raster_meta(path_a)
    meta_b = read_raster_meta(path_b)

    img_a, rep_a = load_as_rgb(path_a, modality=modality_a, **load_kwargs)

    info: Dict[str, Any] = {
        "method": None,
        "report_a": rep_a.as_dict(),
        "report_b": None,
    }

    both_geo = bool(meta_a.get("georeferenced")) and bool(meta_b.get("georeferenced"))
    same_grid = (
        both_geo
        and meta_a.get("crs") == meta_b.get("crs")
        and meta_a.get("bounds") == meta_b.get("bounds")
        and (meta_a.get("width"), meta_a.get("height"))
        == (meta_b.get("width"), meta_b.get("height"))
    )

    if both_geo and not same_grid:
        # Reproject b onto a's grid, then stretch the resampled array.
        try:
            arr, rinfo = resample_to_match(path_b, path_a)
            rep_b = RasterLoadReport(
                path=path_b,
                driver=meta_b.get("driver", "?"),
                modality=modality_b,
                band_count=meta_b.get("band_count", arr.shape[0]),
                dtype=meta_b.get("dtype"),
                width=arr.shape[2],
                height=arr.shape[1],
                crs=meta_a.get("crs"),
                stretch_percentiles=(
                    load_kwargs.get("lo_pct", 2.0),
                    load_kwargs.get("hi_pct", 98.0),
                ),
            )
            rep_b.warnings.append("Reprojected onto the reference image grid.")

            resolved_b = (
                infer_modality(arr.shape[0], str(meta_b.get("dtype")), path_b)
                if modality_b == "auto"
                else modality_b
            )
            rep_b.modality = resolved_b

            valid = np.all(np.isfinite(arr), axis=0)
            picked, warns = choose_rgb_bands(
                arr.shape[0],
                meta_b.get("descriptions") or [],
                meta_b.get("colorinterp") or [],
            )
            rep_b.warnings.extend(warns)

            if resolved_b == "sar":
                idx = [0] if arr.shape[0] == 1 else [0, 1]
            else:
                idx = [p - 1 for p in (picked or (1, 1, 1))]
                idx = [min(max(i, 0), arr.shape[0] - 1) for i in idx]
            rep_b.bands_used = tuple(i + 1 for i in idx)

            planes = []
            svals = []
            for i in idx:
                band = arr[i]
                if resolved_b == "sar":
                    filled = np.where(np.isfinite(band), band, 0.0)
                    band = lee_filter(filled, radius=load_kwargs.get("speckle_radius", 2))
                    band = np.where(valid, band, np.nan)
                    rep_b.speckle_filter = f"lee(radius={load_kwargs.get('speckle_radius', 2)})"
                    band, label, clip = to_decibels(
                        band,
                        is_integer_dn=np.issubdtype(
                            np.dtype(str(meta_b.get("dtype", "float32"))), np.integer
                        ),
                    )
                    rep_b.db_conversion = label
                    rep_b.db_clip = clip
                u8, sv = percentile_stretch(
                    band,
                    valid,
                    load_kwargs.get("lo_pct", 2.0),
                    load_kwargs.get("hi_pct", 98.0),
                )
                planes.append(u8)
                svals.append(sv)
            rep_b.stretch_values = svals

            if len(planes) == 1:
                rgb_b = np.stack([planes[0]] * 3, axis=-1)
            elif len(planes) == 2:
                ratio = np.clip(
                    planes[0].astype(np.float64) - planes[1].astype(np.float64) + 128.0, 0, 255
                ).astype(np.uint8)
                rgb_b = np.stack([planes[0], planes[1], ratio], axis=-1)
            else:
                rgb_b = np.stack(planes[:3], axis=-1)

            img_b = Image.fromarray(rgb_b, mode="RGB")
            info["method"] = "reproject_to_reference_grid"
            info.update(rinfo)
            info["report_b"] = rep_b.as_dict()
            return img_a, img_b, info
        except Exception as exc:
            info.setdefault("warnings", []).append(
                f"Reprojection failed ({exc}); falling back to pixel resize."
            )

    img_b, rep_b = load_as_rgb(path_b, modality=modality_b, **load_kwargs)
    if img_b.size != img_a.size:
        img_b = img_b.resize(img_a.size, Image.BILINEAR)
        rep_b.resized_to = img_a.size
        info["method"] = info["method"] or "pixel_resize"
    else:
        info["method"] = info["method"] or "identical_grid"
    info["report_b"] = rep_b.as_dict()
    return img_a, img_b, info


__all__ = [
    "RasterLoadReport",
    "RasterReadError",
    "load_as_rgb",
    "read_raster_meta",
    "resample_to_match",
    "align_pair",
    "percentile_stretch",
    "to_decibels",
    "lee_filter",
    "choose_rgb_bands",
    "infer_modality",
]
