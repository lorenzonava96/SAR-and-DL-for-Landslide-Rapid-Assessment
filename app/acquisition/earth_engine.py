from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

from app.config import EXPECTED_BAND_ORDER, ImageryConfig, Orbit

@dataclass(frozen=True, slots=True)
class CompositeArtifact:
    orbit: Orbit
    relative_orbit: int
    path: Path


def initialize(project: str | None = None, *, authenticate: bool = False) -> Any:
    """Explicitly initialize Earth Engine.

    Authentication is never triggered during package import. Callers must opt in
    with ``authenticate=True`` when interactive authentication is required.
    """
    try:
        import ee
    except ImportError as exc:
        raise RuntimeError("Earth Engine acquisition requires the 'earth-engine' extra") from exc
    if authenticate:
        ee.Authenticate()
    kwargs = {"project": project} if project else {}
    ee.Initialize(**kwargs)
    return ee


def build_composite_image(
    ee: Any,
    geometry: Any,
    *,
    orbit: Orbit,
    relative_orbit: int,
    pre_end: date,
    post_start: date,
    config: ImageryConfig,
) -> Any:
    """Build the documented four-band Sentinel-1 median/change image."""
    pre_start = pre_end - timedelta(days=config.pre_days)
    post_end = post_start + timedelta(days=config.post_days)
    base = (ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit))
        .filter(ee.Filter.eq("relativeOrbitNumber_start", relative_orbit)))
    pre = base.filterDate(str(pre_start), str(pre_end))
    post = base.filterDate(str(post_start), str(post_end))
    pre_vv = pre.select("VV").median()
    pre_vh = pre.select("VH").median()
    post_vv = post.select("VV").median().rename(EXPECTED_BAND_ORDER[0])
    post_vh = post.select("VH").median().rename(EXPECTED_BAND_ORDER[1])
    diff_vv = post_vv.subtract(pre_vv).rename(EXPECTED_BAND_ORDER[2])
    diff_vh = post_vh.subtract(pre_vh).rename(EXPECTED_BAND_ORDER[3])
    return post_vv.addBands(post_vh).addBands(diff_vv).addBands(diff_vh).clip(geometry)


def available_relative_orbits(
    ee: Any, geometry: Any, *, orbit: Orbit, pre_end: date, post_start: date,
    config: ImageryConfig,
) -> list[int]:
    pre_start = pre_end - timedelta(days=config.pre_days)
    post_end = post_start + timedelta(days=config.post_days)
    collection = (ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(str(pre_start), str(post_end))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit)))
    values = collection.aggregate_array("relativeOrbitNumber_start").distinct().getInfo()
    return sorted(int(value) for value in (values or []))
