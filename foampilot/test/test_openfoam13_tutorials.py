from pathlib import Path

import pytest

from foampilot.tutorials import (
    OpenFOAM13Environment,
    OpenFOAMTutorialManifest,
    validate_generated_case,
)
from foampilot.solver.base_solver import BaseSolver
from foampilot.utilities import OpenFOAMDictAddFile


def _write_minimal_case(root: Path, *, include_nu: bool = True) -> None:
    (root / "system").mkdir(parents=True)
    (root / "constant").mkdir()
    (root / "0").mkdir()
    (root / "system" / "controlDict").write_text(
        "application foamRun;\n", encoding="utf-8"
    )
    (root / "system" / "fvSchemes").write_text("{}\n", encoding="utf-8")
    (root / "system" / "fvSolution").write_text("{}\n", encoding="utf-8")
    (root / "constant" / "transportProperties").write_text(
        "nu 1e-05;\n" if include_nu else "transportModel Newtonian;\n",
        encoding="utf-8",
    )


def test_validate_generated_case_requires_nu(tmp_path):
    _write_minimal_case(tmp_path, include_nu=False)
    result = validate_generated_case(tmp_path)
    assert not result.valid
    assert any("nu" in warning for warning in result.warnings)


def test_validate_generated_case_accepts_complete_case(tmp_path):
    _write_minimal_case(tmp_path)
    result = validate_generated_case(tmp_path)
    assert result.valid
    assert result.missing_files == ()
    assert result.warnings == ()


def test_manifest_discovers_families_and_external_geometry(tmp_path):
    case = tmp_path / "cases" / "openfoam.org" / "demo"
    case.mkdir(parents=True)
    (case / "run").write_text("#!/bin/sh\n", encoding="utf-8")
    (case / "system").mkdir()
    (case / "system" / "snappyHexMeshDict").write_text(
        "geometry { body.stl { type triSurfaceMesh; } }\n", encoding="utf-8"
    )
    specs = OpenFOAMTutorialManifest(tmp_path).discover()
    assert len(specs) == 1
    assert specs[0].family == "openfoam.org"
    assert specs[0].has_run_script
    assert specs[0].requires_external_geometry


def test_environment_fails_with_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="bashrc"):
        OpenFOAM13Environment(tmp_path / "missing-bashrc").environment()


def test_run_command_passes_prepared_environment(tmp_path, monkeypatch):
    solver = BaseSolver(
        tmp_path, solver_name="incompressibleFluid", turbulence_model="kEpsilon"
    )
    prepared = {"PATH": str(tmp_path / "openfoam13" / "bin"), "FOAM_VERSION": "13"}
    captured = {}

    monkeypatch.setattr(solver, "_command_environment", lambda: prepared)

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)

    monkeypatch.setattr("foampilot.solver.base_solver.subprocess.run", fake_run)
    solver.run_command(["blockMesh"], "log.blockMesh")

    assert captured["command"] == ["blockMesh"]
    assert captured["env"] == prepared
    assert (tmp_path / "log.blockMesh").exists()


def test_write_raw_preserves_existing_foamfile_header(tmp_path):
    content = "// source comment\\nFoamFile\\n{\\n    object controlDict;\\n}\\n\napplication foamRun;"
    path = OpenFOAMDictAddFile("generated").write_raw(
        "controlDict", tmp_path, content
    )
    rendered = path.read_text(encoding="utf-8")
    assert rendered.count("FoamFile") == 1
    assert "application foamRun;" in rendered
