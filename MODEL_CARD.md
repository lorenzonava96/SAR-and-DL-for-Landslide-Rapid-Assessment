# SAR-LRA Model Card

## Model overview

**Model name:** SAR-based Landslide Rapid Assessment (SAR-LRA)  
**Model family:** Orbit-specific convolutional neural-network patch classifiers  
**Model version:** `sar-lra-v2.0.0-beta.1`  
**Status:** Beta / research software  
**Primary reference:** Nava et al. (2026), *Sentinel-1 SAR-based globally distributed co-seismic landslide detection by deep neural networks*, Geoscientific Model Development, 19, 167–185. DOI: 10.5194/gmd-19-167-2026.

SAR-LRA identifies image patches likely to contain earthquake-triggered landslides from pre-event and post-event Sentinel-1 SAR composites. The current released models are separate ascending-orbit and descending-orbit classifiers.

This model is not a landslide-warning system, a deterministic inventory, or a substitute for field validation. Every result requires review by a suitably qualified remote-sensing or landslide specialist before operational use.

## Intended use

Suitable uses include:

- rapid first-pass screening after an earthquake-triggered multiple-landslide event;
- prioritising areas for expert image interpretation;
- supporting situational awareness where cloud cover limits optical imagery;
- research into transferable SAR-based landslide detection.

The model is not validated for:

- rainfall-triggered events;
- slow-moving landslides;
- single isolated failures;
- legal, regulatory, insurance, evacuation, or life-safety decisions without independent expert assessment;
- direct estimation of landslide area, volume, runout, impact, or probability of future failure.

## Training and evaluation data

The published study used 11 earthquake-triggered multiple-landslide events comprising approximately 73,000 mapped landslides across varied climatic, geologic, and tectonic settings.

For the released dual-polarisation `VV_VH` rapid-assessment models, training and testing data included:

- Papua New Guinea
- Lombok, Indonesia
- Hokkaido, Japan
- Mesetas, Colombia
- Milin, China
- Luding, China

The wider single-polarisation `VV` experiments also included Capellades, Kaikōura, and Gorkha. Sumatra (2022) and Haiti (2021) were held out from model development and used as unseen-event evaluations.

Landslide inventories were digitised primarily from optical imagery. Orbit-specific shadow and layover masks were used when preparing landslide samples. Patches retained as landslide examples contained more than 5% mapped landslide pixels, approximately 2,000 m² at the working resolution.

The geographic distribution is broad but not exhaustive. Performance should not be assumed equivalent in terrain, land cover, climate, acquisition geometry, or event types absent from the training data.

## Model architecture

Both released models use a convolutional neural network accepting a `64 × 64 × 4` tensor and returning one sigmoid score per patch.

Common settings:

- patch size: 64 × 64 pixels;
- nominal pixel spacing: 10 m;
- output: patch-level score in `[0, 1]`;
- default operational threshold in the notebook: `0.6`;
- sliding-window overlap: 50%;
- non-maximum-suppression overlap threshold: `0.1`;
- dropout: `0.7`;
- learning rate: `0.001`.

Orbit-specific differences:

| Property | Descending model | Ascending model |
|---|---:|---:|
| Acquisition geometry | Descending Sentinel-1 passes only | Ascending Sentinel-1 passes only |
| First-layer filters | 64 | 32 |
| Weight set | Separate descending weights | Separate ascending weights |
| Relative-orbit processing | Each relative orbit processed separately | Each relative orbit processed separately |

The models are not interchangeable. A descending composite must use descending weights, and an ascending composite must use ascending weights. Do not merge different relative orbits before inference because near-range and far-range geometry can introduce inconsistencies.

## Required input

### Sensor and acquisition

- Sentinel-1 Ground Range Detected (`COPERNICUS/S1_GRD`);
- Interferometric Wide (`IW`) mode;
- dual polarisation with both `VV` and `VH`;
- one orbit direction per inference run;
- one relative orbit per composite;
- nominal 10 m sampling.

### Required band order

The input raster must contain exactly four float bands in this order:

1. `postVV` — median VV backscatter over the post-event stack;
2. `postVH` — median VH backscatter over the post-event stack;
3. `diffVV` — `postVV - preVV`;
4. `diffVH` — `postVH - preVH`.

Changing the band order invalidates the model input.

### Value units and ranges

`postVV` and `postVH` must be terrain-corrected Sentinel-1 GRD backscatter in decibels (`10 × log10(x)`), matching the Google Earth Engine Sentinel-1 GRD collection.

`diffVV` and `diffVH` must be arithmetic differences in decibels between post-event and pre-event median composites.

The current inference code applies no clipping, rescaling, standardisation, or per-image normalisation before prediction. Inputs must therefore remain in the same physical representation used by the published workflow.

The publication and repository do not define strict admissible minimum and maximum values. Operational validation should reject non-finite values and record observed per-band minima and maxima. Typical-value limits must not be invented or silently clipped without retraining or formal validation.

### Temporal preprocessing

Released rapid-assessment weights correspond to:

- pre-event stack duration: 60 days;
- post-event stack duration: 12 days;
- median composite for each polarisation and period;
- change bands calculated as post-event median minus pre-event median.

The event must fall between the pre-event and post-event periods. Pre- and post-event collections must both contain images for the same orbit direction and relative orbit.

### Provider preprocessing

The study used Sentinel-1 GRD scenes preprocessed with:

- thermal-noise removal;
- radiometric calibration;
- terrain correction using SRTM 30 m, or ASTER DEM above 60° latitude;
- logarithmic dB values.

Training also considered shadow and layover masking. The operational repository composite builder does not currently export or apply those masks at inference, so predictions in geometrically distorted terrain require particular scrutiny.

## Outputs and model version

Every machine-readable output must include:

```json
{
  "model_name": "SAR-LRA",
  "model_version": "sar-lra-v2.0.0-beta.1",
  "orbit": "ASCENDING or DESCENDING",
  "weights_sha256": "<verified checksum>"
}
```

GeoTIFF outputs should store these values as dataset tags. Vector outputs should include `model_ver`, `orbit`, and `weights_sha256` fields or equivalent sidecar metadata.

The sigmoid output is a model score used for thresholding. It is not a calibrated landslide probability unless calibration is separately demonstrated.

## Known limitations and failure modes

Known or plausible failure modes supported by the repository and publication include:

- false positives along sediment-filled riverbeds and other earthquake-related surface changes;
- reduced recall in unseen regions, observed particularly in Haiti;
- SAR shadow, layover, foreshortening, and incidence-angle effects in steep terrain;
- inconsistent geometry when different relative orbits or near/far-range acquisitions are mixed;
- missing detections for landslides smaller than the training sampling threshold;
- bounding boxes and masks coarser than individual landslide outlines because classification uses 64 × 64 patches;
- sensitivity to unequal or insufficient pre/post image stacks;
- seasonal surface changes, vegetation changes, flooding, snowmelt, agriculture, construction, and other non-landslide changes;
- invalid results when orbit direction, band order, units, preprocessing, temporal windows, or model weights do not match this card;
- uncertain transferability to geographic, climatic, geologic, or land-cover conditions absent from training;
- no demonstrated validation for rainfall-triggered landslides.

A low detection count does not demonstrate absence of landslides. A high model score does not prove a landslide is present.

## Human oversight

All outputs require expert review. Review should compare detections with:

- the underlying post-event and change-detection bands;
- terrain and acquisition geometry;
- optical or higher-resolution imagery when available;
- existing landslide inventories and event reports;
- known river, road, quarry, snow, flood, and construction features.

Operational decisions should record the reviewer, review date, model version, weight checksum, thresholds, source imagery, orbit, relative orbit, and any manual edits.

## Model-weight provenance and licence

The repository root contains an MIT licence and the weight files are stored in the repository. However, neither the licence file nor the current README explicitly states that the copyright holders license the trained model weights under MIT.

Therefore:

**Model-weight licence status: NOT YET EXPLICITLY CONFIRMED.**

Until the maintainer or relevant rights holder adds an explicit statement, downstream redistribution of the weights should not be represented as confirmed MIT-licensed solely because the repository code is MIT-licensed.

Recommended maintainer statement:

> The trained model weight files listed in `model/weights-manifest.json` are copyright the repository contributors and are distributed under the MIT License included in this repository.

Issue acceptance should remain incomplete until that statement is approved by the rights holder and committed.

## Weight integrity

Canonical source URLs, orbit compatibility, architecture parameters, Git blob identifiers, and SHA-256 fields are recorded in `model/weights-manifest.json`.

Run:

```bash
python scripts/verify_model_weights.py
```

to download each file and verify or populate its SHA-256 checksum before release.

## Citation

Nava, L., Mondini, A., Bhuyan, K., Fang, C., Monserrat, O., Novellino, A., and Catani, F. (2026). Sentinel-1 SAR-based globally distributed co-seismic landslide detection by deep neural networks. *Geoscientific Model Development*, 19, 167–185. https://doi.org/10.5194/gmd-19-167-2026.
