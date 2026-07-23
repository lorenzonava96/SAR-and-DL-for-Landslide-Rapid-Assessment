# Issue 4: validated configuration

Issue 4 replaces inference literals and path conventions with a validated configuration object.

Implemented:

- YAML loading through `app.config.load_config`;
- selective CLI overrides with CLI precedence over YAML;
- early validation of orbit, thresholds, temporal-window lengths, tile settings and request dates;
- derivation of sliding-window step from tile size and overlap;
- effective configuration in JSON, GeoTIFF and GeoPackage metadata;
- documented reference defaults in `docs/configuration.md`;
- no `deploy/VV_VH/60_12` production path convention.

The model weight path remains an explicit request input because weights are local artifacts with independently verified checksums. The output directory is configurable and defaults to `outputs/`.
