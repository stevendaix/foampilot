from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_SCRIPT = (
    Path(__file__).parents[4]
    / "examples"
    / "medical_build"
    / "medical_build_end_to_end.py"
)
_SPEC = importlib.util.spec_from_file_location("medical_build_end_to_end", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_export_branch_npz_preserves_variable_section_lengths(tmp_path: Path) -> None:
    branch = {
        "branch_id": 3,
        "points": [[0, 0, 0], [1, 0, 0]],
        "abscissas": [0.0, 1.0],
        "tangents": [[1, 0, 0], [1, 0, 0]],
        "sections": [
            {
                "center": [0, 0, 0],
                "points": [[0, 0, 0], [0, 1, 0]],
                "abscissa": [0.0, 1.0],
            },
            {
                "center": [1, 0, 0],
                "points": [[1, 0, 0], [1, 0.5, 0], [1, 1, 0]],
                "abscissa": [0.0, 0.5, 1.0],
            },
        ],
    }

    _MODULE.export_branch_npz(branch, tmp_path)
    exported = np.load(tmp_path / "branch_03.npz")

    np.testing.assert_array_equal(exported["section_lengths"], [2, 3])
    np.testing.assert_allclose(exported["section_points"][0, :2], branch["sections"][0]["points"])
    np.testing.assert_allclose(exported["section_points"][1, :3], branch["sections"][1]["points"])
    assert np.isnan(exported["section_points"][0, 2]).all()
    assert np.isnan(exported["section_abscissas"][0, 2])
