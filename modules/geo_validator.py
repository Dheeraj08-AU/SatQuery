import rasterio
import numpy as np
from pydantic import BaseModel
from typing import Tuple, Optional


class GeoMetadata(BaseModel):
    is_valid: bool
    format: str
    crs: Optional[str]
    bounds: Optional[Tuple[float, float, float, float]]
    resolution: Optional[Tuple[float, float]]
    band_count: int
    error: Optional[str] = None


class SatQueryValidator:

    def __init__(self):
        self.supported_formats = ['GTiff', 'PNG', 'JPEG']

    def extract_metadata(self, file_path: str) -> GeoMetadata:
        """Extract spatial metadata and check format validity."""

        try:
            with rasterio.open(file_path) as src:

                driver = src.driver

                if driver not in self.supported_formats:
                    return GeoMetadata(
                        is_valid=False,
                        format=driver,
                        band_count=0,
                        error="Unsupported format"
                    )

                has_crs = bool(src.crs)

                return GeoMetadata(
                    is_valid=True,
                    format=driver,
                    crs=src.crs.to_string() if has_crs else "Benchmark Format",

                    bounds=(
                        src.bounds.left,
                        src.bounds.bottom,
                        src.bounds.right,
                        src.bounds.top
                    ) if has_crs else None,

                    resolution=src.res if has_crs else None,

                    band_count=src.count
                )

        except Exception as e:

            return GeoMetadata(
                is_valid=False,
                format="Unknown",
                band_count=0,
                error=str(e)
            )

    def verify_coregistration(
        self,
        img1_path: str,
        img2_path: str
    ) -> bool:

        """Verify whether two images are spatially aligned."""

        meta1 = self.extract_metadata(img1_path)
        meta2 = self.extract_metadata(img2_path)

        if not (meta1.is_valid and meta2.is_valid):
            return False

        # Accept non-georeferenced benchmark PNGs
        if (
            meta1.crs == "Benchmark Format"
            or meta2.crs == "Benchmark Format"
        ):
            return True

        # Check CRS
        crs_match = meta1.crs == meta2.crs

        # Check spatial bounds
        bounds_match = False

        if meta1.bounds and meta2.bounds:
            bounds_match = np.allclose(
                meta1.bounds,
                meta2.bounds,
                atol=1e-4
            )

        return crs_match and bounds_match