from __future__ import annotations

import importlib.util
from pathlib import Path


def test_cfd_runner_import_does_not_parse_pytest_arguments() -> None:
    """The CLI runner must be importable during pytest collection."""
    path = Path(__file__).resolve().parents[2] / "test_cfd_methods.py"
    spec = importlib.util.spec_from_file_location("test_cfd_methods_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
