"""
Input validation and co-registration checking for SatQuery AI.

What was wrong with the previous version
----------------------------------------
1.  Any pair where either image lacked a CRS returned True unconditionally.
    Every PNG/JPEG demo therefore displayed "Co-registration Verified" without
    a single check having run.
2.  For real data it required `np.allclose(bounds1, bounds2, atol=1e-4)`. In a
    projected CRS that is a tenth of a millimetre. A genuinely co-registered
    Cartosat-2S / RISAT pair - the ISRO evaluation format - would be rejected.
3.  It required exact CRS string equality and had no reprojection path, so two
    images of the same footprint in different projections were "not aligned".

What this version does
----------------------
Tolerance is expressed in PIXELS, not in coordinate units, because that is the
only scale-independent way to say "aligned". Bounds are compared after
transforming both footprints into a common CRS. Differing grids are reported
as RESAMPLING_REQUIRED rather than as failure, because `raster_io.align_pair`
can resolve them. Non-georeferenced benchmark imagery is reported honestly as
"cannot be verified" instead of being passed off as verified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from modules.raster_io import read_raster_meta

try:
    from rasterio.warp import transform_bounds

    _HAS_RASTERIO = True
except Exception:  # pragma: no cover
    _HAS_RASTERIO = False


# Default: two images count as aligned if their footprints agree to within one
# pixel. Sub-pixel co-registration is not achievable from header metadata
# alone, so one pixel is the honest limit of what this check can assert.
DEFAULT_TOLERANCE_PX = 1.0

# Minimum share of the smaller footprint that must be covered by the other for
# the pair to be considered the same area.
DEFAULT_MIN_OVERLAP = 0.60

# Resolutions are considered equal within this relative difference.
RESOLUTION_RTOL = 0.02


class CoregistrationStatus(str, Enum):
    IDENTICAL_GRID = "identical_grid"
    ALIGNED_WITHIN_TOLERANCE = "aligned_within_tolerance"
    RESAMPLING_REQUIRED = "resampling_required"
    NO_OVERLAP = "no_overlap"
    NOT_GEOREFERENCED = "not_georeferenced"
    DIMENSION_MISMATCH = "dimension_mismatch"
    UNREADABLE = "unreadable"


@dataclass
class GeoMetadata:
    is_valid: bool
    path: str
    format: str
    band_count: int
    dtype: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    crs: Optional[str] = None
    bounds: Optional[Tuple[float, float, float, float]] = None
    resolution: Optional[Tuple[float, float]] = None
    georeferenced: bool = False
    nodata: Optional[float] = None
    band_descriptions: Optional[List[str]] = None
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CoregistrationResult:
    status: CoregistrationStatus
    ok_to_proceed: bool
    message: str
    # Populated when both inputs are georeferenced.
    pixel_offset: Optional[Tuple[float, float]] = None
    overlap_fraction: Optional[float] = None
    same_crs: Optional[bool] = None
    same_resolution: Optional[bool] = None
    requires_resampling: bool = False
    tolerance_px: float = DEFAULT_TOLERANCE_PX
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        """True only when alignment was actually confirmed against coordinates."""
        return self.status in (
            CoregistrationStatus.IDENTICAL_GRID,
            CoregistrationStatus.ALIGNED_WITHIN_TOLERANCE,
        )

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["verified"] = self.verified
        return d


def _rect_intersection(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> Tuple[float, float]:
    """(intersection area, intersection / smaller footprint) for L,B,R,T rects."""
    left = max(a[0], b[0])
    bottom = max(a[1], b[1])
    right = min(a[2], b[2])
    top = min(a[3], b[3])
    if right <= left or top <= bottom:
        return 0.0, 0.0
    inter = (right - left) * (top - bottom)
    area_a = abs((a[2] - a[0]) * (a[3] - a[1]))
    area_b = abs((b[2] - b[0]) * (b[3] - b[1]))
    smaller = min(area_a, area_b)
    return inter, (inter / smaller if smaller > 0 else 0.0)


class SatQueryValidator:
    def __init__(
        self,
        tolerance_px: float = DEFAULT_TOLERANCE_PX,
        min_overlap: float = DEFAULT_MIN_OVERLAP,
    ):
        self.tolerance_px = tolerance_px
        self.min_overlap = min_overlap
        # GeoTIFF/TIFF for geospatial data; PNG/JPEG accepted for the
        # prescribed public benchmarks, per the problem statement.
        self.supported_drivers = {"GTiff", "COG", "PNG", "JPEG", "BMP", "GIF"}

    # -- metadata ----------------------------------------------------------

    def extract_metadata(self, file_path: str) -> GeoMetadata:
        meta = read_raster_meta(file_path)

        if not meta.get("readable"):
            return GeoMetadata(
                is_valid=False,
                path=file_path,
                format="unknown",
                band_count=0,
                error=meta.get("error", "unreadable"),
            )

        driver = meta.get("driver", "unknown")
        if driver not in self.supported_drivers:
            return GeoMetadata(
                is_valid=False,
                path=file_path,
                format=driver,
                band_count=meta.get("band_count", 0),
                error=(
                    f"Unsupported format '{driver}'. Supported: GeoTIFF/TIFF for "
                    f"geospatial imagery, PNG/JPEG for benchmark datasets."
                ),
            )

        return GeoMetadata(
            is_valid=True,
            path=file_path,
            format=driver,
            band_count=meta.get("band_count", 0),
            dtype=meta.get("dtype"),
            width=meta.get("width"),
            height=meta.get("height"),
            crs=meta.get("crs"),
            bounds=meta.get("bounds"),
            resolution=meta.get("resolution"),
            georeferenced=bool(meta.get("georeferenced")),
            nodata=meta.get("nodata"),
            band_descriptions=meta.get("descriptions") or None,
        )

    # -- co-registration ---------------------------------------------------

    def check_coregistration(self, img1_path: str, img2_path: str) -> CoregistrationResult:
        m1 = self.extract_metadata(img1_path)
        m2 = self.extract_metadata(img2_path)

        if not m1.is_valid or not m2.is_valid:
            bad = m1 if not m1.is_valid else m2
            return CoregistrationResult(
                status=CoregistrationStatus.UNREADABLE,
                ok_to_proceed=False,
                message=f"Cannot read input: {bad.error}",
                details={"image1": m1.as_dict(), "image2": m2.as_dict()},
            )

        # ---- neither / only one georeferenced ----------------------------
        if not (m1.georeferenced and m2.georeferenced):
            same_dims = (m1.width, m1.height) == (m2.width, m2.height)
            which = (
                "neither image is georeferenced"
                if not m1.georeferenced and not m2.georeferenced
                else "only one image carries a CRS"
            )
            if same_dims:
                return CoregistrationResult(
                    status=CoregistrationStatus.NOT_GEOREFERENCED,
                    ok_to_proceed=True,
                    message=(
                        f"Spatial alignment NOT verified - {which}. Pixel dimensions "
                        f"match ({m1.width}x{m1.height}), so the pair is accepted on the "
                        f"assumption that it is a co-registered benchmark sample."
                    ),
                    details={"image1": m1.as_dict(), "image2": m2.as_dict()},
                )
            return CoregistrationResult(
                status=CoregistrationStatus.DIMENSION_MISMATCH,
                ok_to_proceed=True,
                message=(
                    f"Spatial alignment NOT verified - {which}, and pixel dimensions "
                    f"differ ({m1.width}x{m1.height} vs {m2.width}x{m2.height}). "
                    f"The second image will be resampled onto the first image's grid; "
                    f"results may be unreliable if the scenes are not the same area."
                ),
                requires_resampling=True,
                details={"image1": m1.as_dict(), "image2": m2.as_dict()},
            )

        # ---- both georeferenced ------------------------------------------
        if not _HAS_RASTERIO:
            return CoregistrationResult(
                status=CoregistrationStatus.UNREADABLE,
                ok_to_proceed=False,
                message="rasterio is required to verify co-registration.",
            )

        same_crs = m1.crs == m2.crs
        b1 = m1.bounds
        b2 = m2.bounds
        assert b1 is not None and b2 is not None

        if same_crs:
            b2_in_1 = b2
        else:
            try:
                b2_in_1 = transform_bounds(m2.crs, m1.crs, *b2)
            except Exception as exc:
                return CoregistrationResult(
                    status=CoregistrationStatus.NO_OVERLAP,
                    ok_to_proceed=False,
                    message=f"Could not transform image 2 into image 1's CRS: {exc}",
                    same_crs=False,
                    details={"image1": m1.as_dict(), "image2": m2.as_dict()},
                )

        _, overlap = _rect_intersection(b1, b2_in_1)

        if overlap < self.min_overlap:
            return CoregistrationResult(
                status=CoregistrationStatus.NO_OVERLAP,
                ok_to_proceed=False,
                message=(
                    f"The two images cover different areas - footprints overlap by only "
                    f"{overlap:.1%} (minimum {self.min_overlap:.0%}). Paired analysis "
                    f"requires imagery of the same location."
                ),
                overlap_fraction=round(overlap, 4),
                same_crs=same_crs,
                details={"image1": m1.as_dict(), "image2": m2.as_dict()},
            )

        res1 = m1.resolution or (1.0, 1.0)
        res2 = m2.resolution or (1.0, 1.0)
        px = abs(res1[0]) or 1.0
        py = abs(res1[1]) or 1.0

        same_res = (
            abs(abs(res1[0]) - abs(res2[0])) <= RESOLUTION_RTOL * px
            and abs(abs(res1[1]) - abs(res2[1])) <= RESOLUTION_RTOL * py
        )

        # Offset of the two footprints, expressed in image-1 pixels.
        dx_px = max(abs(b1[0] - b2_in_1[0]), abs(b1[2] - b2_in_1[2])) / px
        dy_px = max(abs(b1[1] - b2_in_1[1]), abs(b1[3] - b2_in_1[3])) / py

        same_dims = (m1.width, m1.height) == (m2.width, m2.height)
        details = {
            "image1": m1.as_dict(),
            "image2": m2.as_dict(),
            "image2_bounds_in_image1_crs": tuple(b2_in_1),
        }

        if same_crs and same_dims and same_res and dx_px < 1e-6 and dy_px < 1e-6:
            return CoregistrationResult(
                status=CoregistrationStatus.IDENTICAL_GRID,
                ok_to_proceed=True,
                message=(
                    f"Co-registration verified: identical grid "
                    f"({m1.width}x{m1.height}, {m1.crs}, {px:g} units/px)."
                ),
                pixel_offset=(0.0, 0.0),
                overlap_fraction=round(overlap, 4),
                same_crs=True,
                same_resolution=True,
                tolerance_px=self.tolerance_px,
                details=details,
            )

        if same_crs and same_res and dx_px <= self.tolerance_px and dy_px <= self.tolerance_px:
            return CoregistrationResult(
                status=CoregistrationStatus.ALIGNED_WITHIN_TOLERANCE,
                ok_to_proceed=True,
                message=(
                    f"Co-registration verified: footprints agree to within "
                    f"{max(dx_px, dy_px):.2f} px (tolerance {self.tolerance_px:g} px), "
                    f"overlap {overlap:.1%}."
                ),
                pixel_offset=(round(dx_px, 4), round(dy_px, 4)),
                overlap_fraction=round(overlap, 4),
                same_crs=True,
                same_resolution=True,
                requires_resampling=not same_dims,
                tolerance_px=self.tolerance_px,
                details=details,
            )

        reasons = []
        if not same_crs:
            reasons.append(f"different CRS ({m1.crs} vs {m2.crs})")
        if not same_res:
            reasons.append(f"different resolution ({res1} vs {res2})")
        if max(dx_px, dy_px) > self.tolerance_px:
            reasons.append(f"footprint offset {max(dx_px, dy_px):.2f} px")
        if not same_dims:
            reasons.append(f"different raster size ({m1.width}x{m1.height} vs {m2.width}x{m2.height})")

        return CoregistrationResult(
            status=CoregistrationStatus.RESAMPLING_REQUIRED,
            ok_to_proceed=True,
            message=(
                f"Images cover the same area (overlap {overlap:.1%}) but are not on a "
                f"common grid: {'; '.join(reasons)}. Image 2 will be reprojected onto "
                f"image 1's grid before analysis."
            ),
            pixel_offset=(round(dx_px, 4), round(dy_px, 4)),
            overlap_fraction=round(overlap, 4),
            same_crs=same_crs,
            same_resolution=same_res,
            requires_resampling=True,
            tolerance_px=self.tolerance_px,
            details=details,
        )

    # -- backwards-compatible shim ----------------------------------------

    def verify_coregistration(self, img1_path: str, img2_path: str) -> bool:
        """
        Legacy boolean API. Returns True only when alignment was genuinely
        confirmed against coordinates - it does NOT return True merely because
        the inputs are PNGs. Prefer `check_coregistration`, which distinguishes
        "verified", "cannot be verified" and "different area".
        """
        return self.check_coregistration(img1_path, img2_path).verified


__all__ = [
    "SatQueryValidator",
    "GeoMetadata",
    "CoregistrationResult",
    "CoregistrationStatus",
]
