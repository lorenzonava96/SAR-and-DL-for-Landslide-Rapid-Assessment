from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.config import EXPECTED_BAND_ORDER

@dataclass(slots=True)
class RasterInput:
    data: np.ndarray
    profile: dict[str, Any]
    transform: Any
    crs: Any


def read_sentinel1_stack(path: str | Path) -> RasterInput:
    """Read a four-band raster as HWC float32 without changing its values."""
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Local raster input requires the 'geo' extra") from exc

    path = Path(path)
    with rasterio.open(path) as src:
        if src.count != len(EXPECTED_BAND_ORDER):
            raise ValueError(
                f"Expected four bands {EXPECTED_BAND_ORDER}; found {src.count} in {path}"
            )
        data = np.moveaxis(src.read(), 0, 2).astype(np.float32, copy=False)
        if not np.isfinite(data).all():
            raise ValueError(f"Input contains NaN or infinite values: {path}")
        return RasterInput(data, src.profile.copy(), src.transform, src.crs)
