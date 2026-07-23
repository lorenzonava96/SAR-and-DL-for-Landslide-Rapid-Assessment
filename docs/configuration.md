# Configuration

SAR-LRA loads validated configuration from YAML. Command-line values override only the corresponding YAML values.

```bash
sar-lra \
  --config config/default.yaml \
  --request-id example \
  --input-raster input.tif \
  --weights model/weights/<orbit-model>.hdf5 \
  --pre-end 2026-01-01 \
  --post-start 2026-01-02 \
  --orbit DESCENDING \
  --probability-threshold 0.65
```

The effective configuration is recorded in `result.json`, the output GeoTIFF tags, and each vector output record.

## Defaults and rationale

| Setting | Default | Rationale |
|---|---:|---|
| `model.version` | `v2` | Identifies the documented V2 rapid-assessment workflow and orbit-specific released weights. |
| `model.orbit` | `ASCENDING` | A valid explicit default for configuration templates; users must select the orbit matching the input imagery and weights. Ascending and descending weights are not interchangeable. |
| `model.probability_threshold` | `0.6` | Retains the decision threshold used by the V2 reference inference workflow. It balances screening sensitivity against false detections and is not a calibrated probability guarantee. |
| `model.nms_overlap` | `0.1` | Retains the V2 suppression setting to reduce duplicated overlapping detections while preserving spatially distinct candidates. |
| `model.batch_size` | `512` | Retains the reference inference batch size. It affects memory and throughput, not model semantics, and may be reduced on constrained hardware. |
| `imagery.pre_days` | `60` | Matches the model's documented pre-event median-composite window. Changing it creates a distribution shift from training. |
| `imagery.post_days` | `12` | Matches the rapid post-event composite used by the released V2 models. |
| `imagery.scale_m` | `10` | Matches Sentinel-1 GRD processing at nominal 10 m pixel spacing. |
| `processing.tile_size` | `64` | Matches the trained model input of 64 × 64 pixels. Other values require compatible model weights. |
| `processing.overlap` | `0.5` | Produces a 32-pixel step for 64-pixel tiles, matching the reference sliding-window workflow and reducing boundary misses. |
| `processing.max_roi_km2` | `10000` | Operational guardrail against unexpectedly large, slow, or costly requests. It is not a scientific model limit. |

Invalid or unknown fields, unsupported orbit values, non-positive durations and sizes, thresholds outside `[0, 1]`, and overlap outside `[0, 1)` fail during configuration loading, before raster or model processing starts. Request dates are validated before pipeline execution.
