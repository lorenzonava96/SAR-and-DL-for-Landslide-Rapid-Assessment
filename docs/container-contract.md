# SAR-LRA container contract v1

## Request transport

The container accepts one UTF-8 JSON request file:

```bash
sar-lra run --request /input/request.json
```

Container deployments should mount input data read-only under `/input` and an
output directory under `/output`.

## ROI

Supported ROI inputs:

1. An inline GeoJSON `Polygon` or `MultiPolygon`.
2. A file path to GeoJSON or GeoPackage. GeoPackage inputs may specify `layer`.

Input geometries may use any CRS that can be resolved from the source file.
Inline GeoJSON is always interpreted as EPSG:4326.

The application validates and, where safe, repairs polygon geometry. Empty,
non-polygonal, or unresolved-CRS inputs fail before imagery acquisition.

## Temporal semantics

Two modes are supported:

- `event_date`: derives pre/post windows from one date.
- `explicit_ranges`: uses four inclusive dates supplied by the caller.

For `event_date`, the earthquake date belongs to the post-event window:

- pre window: `[event_date - pre_days, event_date)`
- post window: `[event_date, event_date + post_days]`

This avoids placing earthquake-day imagery in the baseline period.

## Orbit

`orbit` is one of `ascending`, `descending`, or `both`.

For `both`, each orbit is processed independently. Results are then merged into
the requested vector output, while probability and mask rasters remain
orbit-specific if they cannot be safely placed on one common grid.

## Coordinate reference systems

- Inline GeoJSON input: EPSG:4326.
- File input: CRS from source metadata.
- Processing CRS: selected automatically for metric operations and recorded.
- Vector output: EPSG:4326.
- Raster output: the processing grid CRS, embedded in each GeoTIFF.

No output may omit CRS metadata.

## ROI limits

Version 1 limits a request to:

- maximum geodesic area: 10,000 km²;
- maximum 50 polygon parts;
- maximum 100,000 vertices after parsing.

Requests exceeding a limit fail with `ROI_TOO_LARGE` or
`ROI_TOO_COMPLEX`. These defaults should later become configurable environment
variables without changing the request schema.

## Outputs and naming

All output names derive from the stable, caller-supplied `request_id`; they do
not depend on notebook variables such as `place`.

Examples:

- `<request_id>_detections.geojson`
- `<request_id>_detections.gpkg`
- `<request_id>_detection_mask_ascending.tif`
- `<request_id>_probability_ascending.tif`
- `<request_id>_result.json`

A probability surface stores floating-point model scores in `[0, 1]`.
A detection mask stores binary values `0` and `1`, with NoData distinct from
both. Vector detections are polygons derived from the binary mask and are not a
substitute for probability values.

## Empty results

A valid run with no detected landslides is successful:

- process exit code: `0`;
- result status: `succeeded_empty`;
- an empty vector dataset is written with its schema and CRS;
- a zero-valued detection mask is written when requested;
- the probability surface is written when available;
- detected feature count and area are zero.

## Errors

Invalid requests and processing failures use a non-zero exit code and emit a
machine-readable error object in `<request_id>_result.json` when a request ID
can be resolved.

Error shape:

```json
{
  "code": "ROI_TOO_LARGE",
  "message": "ROI area exceeds the 10000 km² limit.",
  "details": {
    "area_km2": 12345.6,
    "maximum_area_km2": 10000
  }
}
```

Initial error codes:

- `INVALID_REQUEST`
- `INVALID_ROI`
- `ROI_TOO_LARGE`
- `ROI_TOO_COMPLEX`
- `INVALID_DATE_RANGE`
- `NO_SAR_SCENES`
- `MODEL_UNAVAILABLE`
- `PROCESSING_FAILED`
- `OUTPUT_WRITE_FAILED`
