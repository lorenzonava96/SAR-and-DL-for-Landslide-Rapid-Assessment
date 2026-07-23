# Issue 3 refactor

Production modules now live in `app/`. The complete former V2 notebook is retained under `notebooks/legacy/v2-reference/`; the active V2 notebook is a thin package example.

Import guarantees:

- `import app` does not import `ee` or TensorFlow;
- Earth Engine authentication is explicit and opt-in;
- model loading accepts local paths and never downloads weights;
- no `app/` module imports notebook utilities.
