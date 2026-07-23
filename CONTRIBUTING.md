# Contributing

Use one branch and pull request per GitHub issue.

```bash
git switch main
git fetch upstream
git merge --ff-only upstream/main
git switch -c issue-<number>-<short-name>
```

Before opening a pull request:

```bash
python scripts/verify_model_weights.py
python -m compileall src
python -m pytest
```

Do not commit generated rasters, shapefiles, prediction archives, Earth Engine credentials, or local runtime directories.
