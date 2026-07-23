from __future__ import annotations

from pathlib import Path

import pytest

from app.config import AppConfig, load_config


def test_default_configuration_matches_reference_values() -> None:
    config = AppConfig()
    assert config.model.probability_threshold == 0.6
    assert config.model.nms_overlap == 0.1
    assert config.model.batch_size == 512
    assert config.imagery.pre_days == 60
    assert config.imagery.post_days == 12
    assert config.imagery.scale_m == 10
    assert config.processing.tile_size == 64
    assert config.processing.overlap == 0.5
    assert config.processing.window_step == 32
    assert config.processing.max_roi_km2 == 10_000


def test_yaml_and_overrides(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "model:\n  orbit: DESCENDING\n  probability_threshold: 0.7\n"
        "processing:\n  overlap: 0.25\n",
        encoding="utf-8",
    )
    config = load_config(path).with_overrides(probability_threshold=0.65, batch_size=64)
    assert config.model.orbit == "DESCENDING"
    assert config.model.probability_threshold == 0.65
    assert config.model.batch_size == 64
    assert config.processing.window_step == 48


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"model": {"orbit": "SIDEWAYS"}}, "orbit"),
        ({"model": {"probability_threshold": 1.1}}, "probability_threshold"),
        ({"model": {"nms_overlap": -0.1}}, "nms_overlap"),
        ({"imagery": {"pre_days": 0}}, "pre_days"),
        ({"processing": {"overlap": 1.0}}, "overlap"),
        ({"processing": {"tile_size": 0}}, "tile_size"),
    ],
)
def test_invalid_configuration_fails_early(raw: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AppConfig.from_mapping(raw)


def test_unknown_configuration_field_fails() -> None:
    with pytest.raises(ValueError, match="Unknown model field"):
        AppConfig.from_mapping({"model": {"mystery": 1}})


def test_effective_configuration_is_serialisable() -> None:
    effective = AppConfig(output_dir=Path("custom")).to_dict()
    assert effective["output_dir"] == "custom"
    assert effective["processing"]["window_step"] == 32
