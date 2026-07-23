import importlib, sys

def test_import_does_not_load_earth_engine_or_tensorflow():
    for name in ("ee", "tensorflow"):
        sys.modules.pop(name, None)
    importlib.import_module("app")
    assert "ee" not in sys.modules
    assert "tensorflow" not in sys.modules
