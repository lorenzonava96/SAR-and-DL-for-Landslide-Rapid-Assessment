# Repository layout

The repository is organised so that scientific reference material, stable contracts, model artefacts, and future operational code are separate.

```text
.
├── assets/                  Documentation images
├── docs/                    Contracts and architecture notes
├── examples/requests/       Valid request examples
├── model/
│   ├── weights/             Canonical released model weights
│   └── weights-manifest.json
├── notebooks/
│   ├── legacy/              Historical V1 workflows retained for provenance
│   └── v2/                  Current reference notebook and utility module
├── schemas/                 Machine-readable API/container contracts
├── scripts/                 Maintenance and verification utilities
├── src/sar_lra/             Installable operational package scaffold
└── tests/                   Unit and integration test locations
```

## Rules for subsequent issues

1. New operational logic belongs under `src/sar_lra`, not inside notebooks.
2. Notebooks should call package functions and remain examples/reference workflows.
3. Runtime downloads and predictions must not be committed.
4. Model weights have one canonical location: `model/weights`.
5. Every model output must contain the model version and weight checksum.
6. Request/result changes must update schemas, examples, and tests together.
7. Historical V1 material is retained under `notebooks/legacy` and should not be modified except for security or reproducibility fixes.
