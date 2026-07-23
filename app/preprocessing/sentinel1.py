from __future__ import annotations
import numpy as np
from app.config import EXPECTED_BAND_ORDER


def validate_stack(data: np.ndarray) -> np.ndarray:
    """Validate an HWC Sentinel-1 stack and return float32 data unchanged."""
    if data.ndim != 3:
        raise ValueError(f"Expected HWC array; received shape {data.shape}")
    if data.shape[-1] != len(EXPECTED_BAND_ORDER):
        raise ValueError(
            f"Expected bands {EXPECTED_BAND_ORDER}; received {data.shape[-1]} channels"
        )
    if not np.isfinite(data).all():
        raise ValueError("Sentinel-1 stack contains NaN or infinite values")
    return data.astype(np.float32, copy=False)


def compose_stack(pre_vv: np.ndarray, pre_vh: np.ndarray,
                  post_vv: np.ndarray, post_vh: np.ndarray) -> np.ndarray:
    """Compose [postVV, postVH, postVV-preVV, postVH-preVH]."""
    shapes = {array.shape for array in (pre_vv, pre_vh, post_vv, post_vh)}
    if len(shapes) != 1:
        raise ValueError(f"All bands must have the same shape; found {sorted(shapes)}")
    return validate_stack(np.stack(
        [post_vv, post_vh, post_vv - pre_vv, post_vh - pre_vh], axis=-1
    ))
