from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.config import MODEL_VERSION


def write_detection_vectors(
    path: str | Path,
    mask: np.ndarray,
    *,
    transform,
    crs,
    orbit: str,
    relative_orbit: int | None,
    weights_sha256: str,
    effective_config: dict[str, Any],
) -> Path | None:
    try:
        import geopandas as gpd
        from rasterio.features import shapes
        from shapely.geometry import shape
    except ImportError as exc:
        raise RuntimeError("Vector output requires the 'geo' extra") from exc
    config_json = json.dumps(effective_config, separators=(",", ":"))
    records = []
    for geometry, value in shapes(mask.astype(np.uint8), mask=mask == 1, transform=transform):
        if value == 1:
            records.append(
                {
                    "geometry": shape(geometry),
                    "model_ver": MODEL_VERSION,
                    "cfg_version": str(effective_config["model"]["version"]),
                    "orbit": orbit,
                    "rel_orbit": relative_orbit,
                    "weights_sha256": weights_sha256,
                    "effective_config": config_json,
                }
            )
    if not records:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(records, crs=crs).to_file(path)
    return path
