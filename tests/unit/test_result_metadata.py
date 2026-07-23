from pathlib import Path

from app.config import AppConfig
from app.schemas import OutputArtifact, PipelineResult


def test_result_contains_effective_configuration() -> None:
    config = AppConfig().with_overrides(orbit="DESCENDING", probability_threshold=0.7)
    result = PipelineResult(
        request_id="request-1",
        orbit="DESCENDING",
        status="succeeded_empty",
        weights_sha256="abc",
        artifacts=[OutputArtifact("metadata", Path("result.json"))],
        effective_configuration=config.to_dict(),
    )
    payload = result.to_dict()
    assert payload["effective_configuration"]["model"]["orbit"] == "DESCENDING"
    assert payload["effective_configuration"]["model"]["probability_threshold"] == 0.7
    assert payload["effective_configuration"]["processing"]["window_step"] == 32
