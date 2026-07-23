from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.config import EXPECTED_BAND_ORDER, MODEL_NAME, MODEL_VERSION


def write_detection_mask(
    path: str | Path,
    mask: np.ndarray,
    profile: dict[str, Any],
    *,
    orbit: str,
    relative_orbit: int | None,
    weights_sha256: str,
    effective_config: dict[str, Any],
) -> Path:
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Raster output requires the 'geo' extra") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output_profile = profile.copy()
    output_profile.update(dtype="uint8", count=1, compress="lzw", nodata=0)
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(mask.astype(np.uint8), 1)
        dst.update_tags(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            configured_model_version=str(effective_config["model"]["version"]),
            orbit=orbit,
            relative_orbit="" if relative_orbit is None else str(relative_orbit),
            weights_sha256=weights_sha256,
            input_band_order=",".join(EXPECTED_BAND_ORDER),
            output_type="binary_detection_mask",
            effective_configuration=json.dumps(effective_config, separators=(",", ":")),
        )
    return path
