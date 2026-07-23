from __future__ import annotations
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from .config import MODEL_NAME, MODEL_VERSION, Orbit

@dataclass(frozen=True, slots=True)
class PipelineRequest:
    request_id: str
    orbit: Orbit
    pre_end: date
    post_start: date
    weights_path: Path
    roi_geojson: dict[str, Any] | None = None
    input_raster: Path | None = None

    def __post_init__(self) -> None:
        if bool(self.roi_geojson) == bool(self.input_raster):
            raise ValueError("Provide exactly one of roi_geojson or input_raster")
        if self.post_start < self.pre_end:
            raise ValueError("post_start must not be earlier than pre_end")

@dataclass(slots=True)
class OutputArtifact:
    kind: str
    path: Path

@dataclass(slots=True)
class PipelineResult:
    request_id: str
    orbit: Orbit
    status: str
    weights_sha256: str
    artifacts: list[OutputArtifact] = field(default_factory=list)
    model_name: str = MODEL_NAME
    model_version: str = MODEL_VERSION
    effective_configuration: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["artifacts"] = [
            {"kind": item.kind, "path": str(item.path)} for item in self.artifacts
        ]
        return data
