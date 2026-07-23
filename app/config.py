from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping

MODEL_NAME = "SAR-LRA"
MODEL_VERSION = "sar-lra-v2.0.0-beta.1"
EXPECTED_BAND_ORDER = ("postVV", "postVH", "diffVV", "diffVH")
Orbit = Literal["ASCENDING", "DESCENDING"]
VALID_ORBITS = {"ASCENDING", "DESCENDING"}


@dataclass(frozen=True, slots=True)
class ModelConfig:
    version: str = "v2"
    orbit: Orbit = "ASCENDING"
    probability_threshold: float = 0.6
    nms_overlap: float = 0.1
    batch_size: int = 512

    def __post_init__(self) -> None:
        if self.orbit not in VALID_ORBITS:
            raise ValueError(f"orbit must be one of {sorted(VALID_ORBITS)}")
        if not self.version.strip():
            raise ValueError("model.version must not be empty")
        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ValueError("model.probability_threshold must be between 0 and 1")
        if not 0.0 <= self.nms_overlap <= 1.0:
            raise ValueError("model.nms_overlap must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("model.batch_size must be positive")


@dataclass(frozen=True, slots=True)
class ImageryConfig:
    pre_days: int = 60
    post_days: int = 12
    scale_m: int = 10

    def __post_init__(self) -> None:
        if self.pre_days <= 0:
            raise ValueError("imagery.pre_days must be positive")
        if self.post_days <= 0:
            raise ValueError("imagery.post_days must be positive")
        if self.scale_m <= 0:
            raise ValueError("imagery.scale_m must be positive")


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    tile_size: int = 64
    overlap: float = 0.5
    max_roi_km2: float = 10_000.0

    def __post_init__(self) -> None:
        if self.tile_size <= 0:
            raise ValueError("processing.tile_size must be positive")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("processing.overlap must be at least 0 and less than 1")
        if self.max_roi_km2 <= 0:
            raise ValueError("processing.max_roi_km2 must be positive")
        if self.window_step <= 0:
            raise ValueError("processing.overlap leaves no positive window step")

    @property
    def window_step(self) -> int:
        return max(1, round(self.tile_size * (1.0 - self.overlap)))


@dataclass(frozen=True, slots=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    imagery: ImageryConfig = field(default_factory=ImageryConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output_dir: Path = Path("outputs")

    def __post_init__(self) -> None:
        if not str(self.output_dir):
            raise ValueError("output_dir must not be empty")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        data["processing"]["window_step"] = self.processing.window_step
        return data

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "AppConfig":
        raw = raw or {}
        _reject_unknown(raw, {"model", "imagery", "processing", "output_dir"}, "configuration")
        model_raw = _mapping(raw.get("model"), "model")
        imagery_raw = _mapping(raw.get("imagery"), "imagery")
        processing_raw = _mapping(raw.get("processing"), "processing")
        _reject_unknown(model_raw, {"version", "orbit", "probability_threshold", "nms_overlap", "batch_size"}, "model")
        _reject_unknown(imagery_raw, {"pre_days", "post_days", "scale_m"}, "imagery")
        _reject_unknown(processing_raw, {"tile_size", "overlap", "max_roi_km2"}, "processing")
        return cls(
            model=ModelConfig(**model_raw),
            imagery=ImageryConfig(**imagery_raw),
            processing=ProcessingConfig(**processing_raw),
            output_dir=Path(raw.get("output_dir", "outputs")),
        )

    def with_overrides(self, **overrides: Any) -> "AppConfig":
        model = self.model
        imagery = self.imagery
        processing = self.processing
        output_dir = self.output_dir
        model_fields = {"version", "orbit", "probability_threshold", "nms_overlap", "batch_size"}
        imagery_fields = {"pre_days", "post_days", "scale_m"}
        processing_fields = {"tile_size", "overlap", "max_roi_km2"}
        for key, value in overrides.items():
            if value is None:
                continue
            if key in model_fields:
                model = replace(model, **{key: value})
            elif key in imagery_fields:
                imagery = replace(imagery, **{key: value})
            elif key in processing_fields:
                processing = replace(processing, **{key: value})
            elif key == "output_dir":
                output_dir = Path(value)
            else:
                raise ValueError(f"Unknown configuration override: {key}")
        return AppConfig(model=model, imagery=imagery, processing=processing, output_dir=output_dir)


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML configuration requires PyYAML") from exc
    config_path = Path(path)
    if not config_path.is_file():
        raise ValueError(f"Configuration file does not exist: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration: {exc}") from exc
    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("Configuration root must be a mapping")
    return AppConfig.from_mapping(raw)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} field(s): {', '.join(unknown)}")


@dataclass(frozen=True, slots=True)
class ModelLoadConfig:
    orbit: Orbit
    weights_path: Path
    filters_first_layer: int
    patch_size: int
    channels: int = 4
    learning_rate: float = 0.001
    dropout: float = 0.7
