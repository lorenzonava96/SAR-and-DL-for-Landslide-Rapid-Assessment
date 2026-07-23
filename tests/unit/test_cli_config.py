from __future__ import annotations

import pytest

from app.cli import build_parser


def test_invalid_iso_date_fails_during_argument_parsing() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--request-id", "x",
            "--input-raster", "input.tif",
            "--weights", "weights.h5",
            "--pre-end", "not-a-date",
            "--post-start", "2026-01-02",
        ])
