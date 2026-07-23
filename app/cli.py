from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from app.config import load_config
from app.pipeline import run_local
from app.schemas import PipelineRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sar-lra")
    parser.add_argument("--config", type=Path, help="YAML configuration file")
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--input-raster", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--pre-end", type=date.fromisoformat, required=True)
    parser.add_argument("--post-start", type=date.fromisoformat, required=True)

    parser.add_argument("--version")
    parser.add_argument("--orbit", choices=("ASCENDING", "DESCENDING"))
    parser.add_argument("--probability-threshold", type=float)
    parser.add_argument("--nms-overlap", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--pre-days", type=int)
    parser.add_argument("--post-days", type=int)
    parser.add_argument("--scale-m", type=int)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=float)
    parser.add_argument("--max-roi-km2", type=float)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config).with_overrides(
            version=args.version,
            orbit=args.orbit,
            probability_threshold=args.probability_threshold,
            nms_overlap=args.nms_overlap,
            batch_size=args.batch_size,
            pre_days=args.pre_days,
            post_days=args.post_days,
            scale_m=args.scale_m,
            tile_size=args.tile_size,
            overlap=args.overlap,
            max_roi_km2=args.max_roi_km2,
            output_dir=args.output_dir,
        )
        request = PipelineRequest(
            request_id=args.request_id,
            orbit=config.model.orbit,
            pre_end=args.pre_end,
            post_start=args.post_start,
            weights_path=args.weights,
            input_raster=args.input_raster,
        )
        result = run_local(request, config)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
