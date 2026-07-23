"""Deprecated compatibility imports.

Reusable implementation moved to :mod:`app`. New code must import from the
package directly. This file contains no production implementation.
"""
from warnings import warn
warn("notebooks/v2/utils.py is deprecated; import from app", DeprecationWarning, stacklevel=2)
from app.acquisition.earth_engine import build_composite_image, initialize
from app.inference.model import build_model, focal_loss, load_model
from app.inference.predict import predict_stack
from app.inference.windows import non_max_suppression, sliding_windows
from app.pipeline import run_local

__all__ = ["build_composite_image", "initialize", "build_model", "focal_loss",
           "load_model", "predict_stack", "non_max_suppression",
           "sliding_windows", "run_local"]
