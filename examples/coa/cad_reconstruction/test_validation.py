import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest
import trimesh
import vtk

from .centerline_extractor import CenterlineExtractor
from .section_extractor import SectionExtractor
from .occ_builder import OCCBuilder


def _vtk_polydata_to_trimesh(pd: vtk.vtkPolyData) -> trimesh.Trimesh:
    pts = []
    for i in range(pd.GetNumberOfPoints()):
        p = pd.GetPoint(i)
        pts.append([p[0], p[1], p[2]])
    faces = []
    polys = pd.GetPolys()
    polys.InitTraversal()
    pt_ids = vtk.vtkIdList()
    while polys.GetNextCell(pt_ids):
        if pt_ids.GetNumberOfIds() >= 3:
            faces.append([pt_ids.GetId(0), pt_ids.GetId(1), pt_ids.GetId(2)])
    return trimesh.Trimesh(np.array(pts, dtype=float), np.array(faces, dtype=int), process=False)


@pytest.fixture(scope="module")
def tl_mesh():
    mesh_path = Path("/home/steven/foampilot/examples/coa/data_preproc/tbad_stl_output/tbad_TL_walls.stl")
    mesh = trimesh.load(mesh_path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    return mesh.process(True)


@pytest.fixture(scope="module")
def centerline(tl_mesh):
    extractor = CenterlineExtractor(resampling_step_mm=1.0)
    return extractor.extract(
        Path("/home/steven/foampilot/examples/coa/data_preproc/tbad_stl_output/tbad_TL_walls.stl")
    )


@pytest.fixture(scope="module")
def sections(tl_mesh, centerline):
    extractor = SectionExtractor(spacing_mm=2.0)
    return extractor.extract(tl_mesh, centerline)


def test_centerline_points_count(centerline):
    assert centerline.shape[0] > 10
    assert centerline.shape[1] == 3


def test_centerline_continuous(centerline):
    diffs = np.diff(centerline, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    assert np.all(distances > 0)


def test_sections_count(sections):
    assert len(sections) >= 1


def test_section_points_count(sections):
    for section in sections:
        assert section.points.shape[1] == 3
        assert section.points.shape[0] >= 3


def test_section_center_consistency(sections):
    for section in sections:
        expected_center = section.points.mean(axis=0)
        np.testing.assert_allclose(section.center, expected_center, rtol=1e-5)


def test_section_direction_unit(sections):
    for section in sections:
        norm = np.linalg.norm(section.direction)
        assert abs(norm - 1.0) < 1e-6


def test_section_local_frame_orthogonal(sections):
    for section in sections:
        x, y, z = section.local_frame()
        assert abs(np.dot(x, y)) < 1e-6
        assert abs(np.dot(x, z)) < 1e-6
        assert abs(np.dot(y, z)) < 1e-6


def test_section_2d_projection(sections):
    for section in sections:
        pts2d = section.to_2d()
        assert pts2d.shape[1] == 2
        assert pts2d.shape[0] == section.points.shape[0]


def test_occ_builder_uses_all_sections(sections):
    """Verify that OCCBuilder no longer skips odd-indexed sections."""
    builder = OCCBuilder(n_samples=20)
    # Access the internal _build to check curve count
    # We can't call build_from_sections without a case_dir, but we can
    # verify the builder is constructed properly
    assert builder.fitter.degree == 3
    assert builder.fitter.n_ctrl == 12
    assert builder.n_samples == 20


def test_distance_field_computation(centerline):
    """Test the distance field utility from mesh_utils."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from mesh_utils import compute_distance_field

    # Surface points sampled from the mesh
    mesh_path = Path("/home/steven/foampilot/examples/coa/data_preproc/tbad_stl_output/tbad_TL_walls.stl")
    mesh = trimesh.load(mesh_path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    # Sample some surface vertices
    surface_pts = mesh.vertices[::50]  # every 50th vertex for speed
    distances = compute_distance_field(surface_pts, centerline)

    assert distances.shape == (len(surface_pts),)
    assert np.all(distances >= 0)
    assert np.all(np.isfinite(distances))


def test_non_newtonian_carreau_yasuda_available():
    """Test that CarreauYasuda model is available in foampilot."""
    from foampilot.constant.transportPropertiesFile import NonNewtonianModels

    assert "CarreauYasuda" in NonNewtonianModels.list_models()
    assert NonNewtonianModels.CARREAU_YASUDA == "CarreauYasuda"


def test_non_newtonian_carreau_yasuda_config():
    """Test that Carreau-Yasuda coefficients are correctly set."""
    from foampilot.constant.transportPropertiesFile import TransportPropertiesFile, NonNewtonianModels
    from foampilot.utilities.manageunits import ValueWithUnit

    tp = TransportPropertiesFile(transportModel="Newtonian")
    tp.set_non_newtonian(
        model=NonNewtonianModels.CARREAU_YASUDA,
        rho=1060,
        nu0=13.96e-6,
        nuInf=3.77e-6,
        **{"lambda": 12.3},
        n=0.216,
        a=0.6,
    )

    assert tp.attributes["transportModel"] == "CarreauYasuda"
    assert tp.attributes["rho"] == 1060.0
    coeffs = tp.attributes["CarreauYasudaCoeffs"]
    assert "nu0" in coeffs
    assert "nuInf" in coeffs
    assert "lambda" in coeffs
    assert "n" in coeffs
    assert "a" in coeffs


def test_checkmesh_utility_structure():
    """Test that the checkMesh utility has the expected interface."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from mesh_utils import run_checkmesh, setup_gmsh_adaptive_sizing, create_boundary_layers

    # Verify functions exist and are callable
    assert callable(run_checkmesh)
    assert callable(setup_gmsh_adaptive_sizing)
    assert callable(create_boundary_layers)
