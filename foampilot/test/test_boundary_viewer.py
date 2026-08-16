"""
Test the BoundaryViewer module.

These tests verify that OpenFOAM boundary patches can be inspected
and visualized using the BoundaryViewer class.
"""

import sys
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from foampilot.postprocess.openfoam_direct import OpenFOAMDirectReader
from foampilot.postprocess.boundary_viewer import (
    BoundaryViewer,
    PatchInfo,
    _compute_patch_area,
    _compute_bounds,
    _classify_patch_type,
    _classify_field_bc_type,
)

TEST_DIR = Path(__file__).resolve().parent
PLANAR_CASE = TEST_DIR.parent.parent / "planarPoiseuille"
CHT_CASE = TEST_DIR.parent.parent / "examples" / "cht" / "simple_heated_duct"


def test_list_patches():
    reader = OpenFOAMDirectReader(PLANAR_CASE)
    viewer = BoundaryViewer(reader)
    patches = viewer.list_patches()
    assert len(patches) > 0
    assert "left" in patches


def test_get_patch_info():
    reader = OpenFOAMDirectReader(PLANAR_CASE)
    viewer = BoundaryViewer(reader)
    info = viewer.get_patch_info("left")
    assert isinstance(info, PatchInfo)
    assert info.name == "left"
    assert info.n_faces > 0
    assert info.area > 0
    assert len(info.bounds) == 6
    assert info.n_cells > 0


def test_get_patch_faces():
    reader = OpenFOAMDirectReader(PLANAR_CASE)
    viewer = BoundaryViewer(reader)
    faces = viewer.get_patch_faces("left")
    assert isinstance(faces, np.ndarray)
    assert len(faces) > 0
    assert faces.dtype == int


def test_get_patch_mesh():
    reader = OpenFOAMDirectReader(PLANAR_CASE)
    viewer = BoundaryViewer(reader)
    mesh = viewer.get_patch_mesh("left")
    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_points > 0


def test_get_boundary_only():
    reader = OpenFOAMDirectReader(PLANAR_CASE)
    viewer = BoundaryViewer(reader)
    mesh = viewer.get_boundary_only()
    assert isinstance(mesh, pv.PolyData)
    assert "patch_id" in mesh.cell_data
    assert "patch_name" in mesh.cell_data
    assert len(mesh.cell_data["patch_id"]) == mesh.n_cells


def test_get_bc_type_mesh():
    reader = OpenFOAMDirectReader(PLANAR_CASE)
    viewer = BoundaryViewer(reader)
    mesh = viewer.get_bc_type_mesh()
    assert isinstance(mesh, pv.PolyData)
    assert "bc_type_id" in mesh.cell_data
    assert "bc_type_name" in mesh.cell_data
    assert len(mesh.cell_data["bc_type_id"]) == mesh.n_cells


def test_plotter_creation():
    reader = OpenFOAMDirectReader(PLANAR_CASE)
    viewer = BoundaryViewer(reader)
    plotter = viewer.plot(off_screen=True)
    assert plotter.shape == (2, 2)


def test_classify_patch_type():
    assert _classify_patch_type("patch") == "patch"
    assert _classify_patch_type("wall") == "wall"
    assert _classify_patch_type("symmetryPlane") == "symmetry"
    assert _classify_patch_type("cyclicAMI") == "cyclic"
    assert _classify_patch_type("processor") == "processor"
    assert _classify_patch_type("empty") == "empty"
    assert _classify_patch_type("unknownType") == "other"


def test_classify_field_bc_type():
    assert _classify_field_bc_type("fixedValue") == "fixedValue"
    assert _classify_field_bc_type("zeroGradient") == "zeroGradient"
    assert _classify_field_bc_type("kqRWallFunction") == "wallFunction"
    assert _classify_field_bc_type("symmetry") == "symmetry"
    assert _classify_field_bc_type("inletOutlet") == "inletOutlet"
    assert _classify_field_bc_type("calculated") == "calculated"
    assert _classify_field_bc_type("unknownType") == "other"


def test_compute_patch_area():
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    area = _compute_patch_area(pts)
    assert area > 0
    assert abs(area - 1.0) < 1e-6


def test_compute_bounds():
    pts = np.array([[0, 0, 0], [1, 2, 3]], dtype=float)
    bounds = _compute_bounds(pts)
    assert bounds == (0.0, 1.0, 0.0, 2.0, 0.0, 3.0)


def test_cht_boundary_viewer():
    reader = OpenFOAMDirectReader(CHT_CASE, region="fluid")
    viewer = BoundaryViewer(reader)
    patches = viewer.list_patches()
    assert len(patches) > 0
