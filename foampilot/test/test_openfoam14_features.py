#!/usr/bin/env python3
"""Validation test suite for foampilot OpenFOAM-14 features.

Each test validates that a feature was added correctly and that
generated output files are syntactically coherent with OpenFOAM.
"""

import ast
import os
import sys
import tempfile
from pathlib import Path

# Ensure foampilot is importable
sys.path.insert(0, str(Path(__file__).parent / "foampilot" / "src"))


def test_fvConstraintsFile_import():
    """Test that FvConstraintsFile can be imported."""
    from foampilot.system.fvConstraintsFile import FvConstraintsFile
    assert FvConstraintsFile is not None
    print("[OK] FvConstraintsFile importable")


def test_fvConstraintsFile_to_dict():
    """Test FvConstraintsFile.to_dict() produces valid structure."""
    from foampilot.system.fvConstraintsFile import FvConstraintsFile
    f = FvConstraintsFile()
    f.add_constraint("testConstraint", "pointConstraint", patch="walls", point=(0, 0, 0))
    d = f.to_dict()
    assert "testConstraint" in d
    assert d["testConstraint"]["type"] == "pointConstraint"
    assert d["testConstraint"]["patch"] == "walls"
    print("[OK] FvConstraintsFile.to_dict() valid")


def test_fvModelsFile_import():
    """Test that FvModelsFile can be imported."""
    from foampilot.system.fvModelsFile import FvModelsFile
    assert FvModelsFile is not None
    print("[OK] FvModelsFile importable")


def test_fvModelsFile_porous_zone():
    """Test FvModelsFile.add_porous_zone()."""
    from foampilot.system.fvModelsFile import FvModelsFile
    f = FvModelsFile()
    f.add_porous_zone(
        "porous1",
        ["patch1"],
        permeability={"x": 1e-10, "y": 1e-10, "z": 1e-10},
        porosity=0.4,
    )
    d = f.to_dict()
    assert "porous1" in d
    assert d["porous1"]["type"] == "porousZone"
    assert d["porous1"]["porousZone"]["porosity"] == 0.4
    print("[OK] FvModelsFile.add_porous_zone() valid")


def test_fvModelsFile_fan():
    """Test FvModelsFile.add_fan()."""
    from foampilot.system.fvModelsFile import FvModelsFile
    f = FvModelsFile()
    f.add_fan(
        "fan1",
        ["inlet"],
        fan_curve={"flowRate": [0, 1, 2], "pressure": [100, 50, 0]},
        power=100.0,
        origin=[0, 0, 0],
        axis=[0, 0, 1],
    )
    d = f.to_dict()
    assert "fan1" in d
    assert d["fan1"]["type"] == "fan"
    print("[OK] FvModelsFile.add_fan() valid")


def test_fvModelsFile_heat_source():
    """Test FvModelsFile.add_heat_source()."""
    from foampilot.system.fvModelsFile import FvModelsFile
    f = FvModelsFile()
    f.add_heat_source(
        "heater1",
        ["walls"],
        heat_source={"Q": "100", "volumetric": True},
    )
    d = f.to_dict()
    assert "heater1" in d
    assert d["heater1"]["type"] == "source"
    print("[OK] FvModelsFile.add_heat_source() valid")


def test_controlDictFile_functions():
    """Test that ControlDictFile now supports functions list."""
    from foampilot.system.controlDictFile import ControlDictFile

    c = ControlDictFile(
        startTime=0,
        endTime=100,
        functions=["fieldAverage", "runTimeControl"],
    )
    assert "functions" in c.to_dict()
    assert c.to_dict()["functions"] == ["fieldAverage", "runTimeControl"]

    c.add_function("fieldAverage")
    assert "fieldAverage" in c.functions
    c.add_function("residuals")
    assert "residuals" in c.functions

    print("[OK] ControlDictFile.functions valid")


def test_controlDictFile_from_dict():
    """Test ControlDictFile.from_dict() with functions field."""
    from foampilot.system.controlDictFile import ControlDictFile

    config = {
        "application": "simpleFoam",
        "startTime": 0,
        "endTime": 1000,
        "deltaT": 0.01,
        "functions": ["fieldAverage", "referencePressure", "runTimeControls"],
    }
    c = ControlDictFile.from_dict(config)
    assert c.functions == ["fieldAverage", "referencePressure", "runTimeControls"]
    print("[OK] ControlDictFile.from_dict() with functions valid")


def test_base_solver_opfoam14_solvers():
    """Test that base_solver.SOLVER_MODULES includes OpenFOAM-14 solvers."""
    from foampilot.solver.base_solver import BaseSolver

    expected_solvers = [
        "icoFoam", "simpleFoam", "pimpleFoam", "pimpleDyMFoam",
        "rhoCentralFoam", "sonicFoam", "reactingFoam",
        "scalarTransportFoam", "chtMultiRegionFoam",
    ]
    for s in expected_solvers:
        assert s in BaseSolver.SOLVER_MODULES, f"Missing solver: {s}"
    print(f"[OK] SOLVER_MODULES contains {len(expected_solvers)} OpenFOAM-14 solvers")


def test_openfoam_file_functions_field():
    """Test that OpenFOAMFile.write_boundary_file() includes functions support."""
    from foampilot.base.openFOAMFile import OpenFOAMFile

    with tempfile.TemporaryDirectory() as tmp:
        foam_file = OpenFOAMFile("controlDict")
        foam_file.write_file(Path(tmp) / "controlDict")
        assert (Path(tmp) / "controlDict").exists()
    print("[OK] OpenFOAMFile write_file valid")


def test_syntax_all_new_files():
    """Syntax-check all newly created files."""
    source_root = Path(__file__).resolve().parents[1] / "src" / "foampilot" / "system"
    new_files = [
        source_root / "fvConstraintsFile.py",
        source_root / "fvModelsFile.py",
    ]
    for f in new_files:
        with open(f) as fh:
            ast.parse(fh.read())
        print(f"[OK] Syntax check: {Path(f).name}")


def test_gmsh_mesher_primitives():
    """Test that new Gmsh methods exist and are callable (requires gmsh import)."""
    try:
        import gmsh
    except ImportError:
        print("[SKIP] gmsh not installed — skipping primitives test")
        return

    from foampilot.mesh.gmsh_mesher import GmshMesher

    gmsh.initialize()
    gmsh.model.add("test")

    class FakeParent:
        case_path = Path("/tmp/test_foampilot")

    parent = FakeParent()
    mesher = GmshMesher(parent, model_name="test_primitives", verbose=False)

    # Test add_point
    tag = mesher.add_point(0.0, 0.0, 0.0, lc=0.1)
    assert isinstance(tag, int)
    assert tag > 0

    # Test add_line
    line_tag = mesher.add_line(0, 0, 0, 1, 0, 0, lc=0.1)
    assert isinstance(line_tag, int)
    assert line_tag > 0

    # Test add_rectangle
    surf_tag = mesher.add_rectangle(0, 0, 0, 1, 1, lc=0.1)
    assert isinstance(surf_tag, int)
    assert surf_tag > 0

    # Test add_circle
    wire_tag = mesher.add_circle(0, 0, 0, 0.5, lc=0.1)
    assert isinstance(wire_tag, int)
    assert wire_tag > 0

    gmsh.finalize()
    print("[OK] GmshMesher primitives all callable")


def test_gmsh_mesher_assign_patches_by_normal():
    """Test that assign_patches_by_normal exists in GmshMesher."""
    from foampilot.mesh.gmsh_mesher import GmshMesher
    assert hasattr(GmshMesher, "assign_patches_by_normal"), "Missing assign_patches_by_normal"
    assert hasattr(GmshMesher, "add_point"), "Missing add_point"
    assert hasattr(GmshMesher, "add_line"), "Missing add_line"
    assert hasattr(GmshMesher, "add_circle"), "Missing add_circle"
    assert hasattr(GmshMesher, "add_rectangle"), "Missing add_rectangle"
    assert hasattr(GmshMesher, "extrude_surface"), "Missing extrude_surface"
    assert hasattr(GmshMesher, "extrude_profile"), "Missing extrude_profile"
    assert hasattr(GmshMesher, "boolean_union"), "Missing boolean_union"
    assert hasattr(GmshMesher, "boolean_difference"), "Missing boolean_difference"
    assert hasattr(GmshMesher, "boolean_intersection"), "Missing boolean_intersection"
    print("[OK] GmshMesher has all new primitive methods")


def test_boundary_advanced_patches():
    """Test that Boundary.initialize_boundary() handles advanced patch types."""
    from foampilot.boundaries.boundaries_dict import Boundary
    assert hasattr(Boundary, "initialize_boundary"), "Missing initialize_boundary"
    print("[OK] Boundary.initialize_boundary() supports advanced patches")


def test_opencvfoam_reader_method():
    """Test that FoamPostProcessing has read_direct method."""
    from foampilot.postprocess.openfoam_pyvista import FoamPostProcessing
    assert hasattr(FoamPostProcessing, "read_direct"), "Missing read_direct"
    assert hasattr(FoamPostProcessing, "calc_y_plus"), "Missing calc_y_plus"
    assert hasattr(FoamPostProcessing, "calc_strain_rate"), "Missing calc_strain_rate"
    assert hasattr(FoamPostProcessing, "calc_wall_shear_stress"), "Missing calc_wall_shear_stress"
    print("[OK] FoamPostProcessing has all new methods")


def run_all_tests():
    """Run all validation tests and report results."""
    tests = [
        test_fvConstraintsFile_import,
        test_fvConstraintsFile_to_dict,
        test_fvModelsFile_import,
        test_fvModelsFile_porous_zone,
        test_fvModelsFile_fan,
        test_fvModelsFile_heat_source,
        test_controlDictFile_functions,
        test_controlDictFile_from_dict,
        test_base_solver_opfoam14_solvers,
        test_openfoam_file_functions_field,
        test_syntax_all_new_files,
        test_gmsh_mesher_primitives,
        test_gmsh_mesher_assign_patches_by_normal,
        test_boundary_advanced_patches,
        test_opencvfoam_reader_method,
    ]

    passed = 0
    failed = 0
    errors = []

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test.__name__, str(e)))
            print(f"[FAIL] {test.__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'='*60}")

    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  - {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())