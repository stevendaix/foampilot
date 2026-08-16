"""
Tests for FoamPostProcessing using a synthetic cube mesh.

These tests create a simple cube mesh with synthetic field data
(velocity U, pressure p) to verify that the post-processing
functions (slice plots, vector plots, contour plots, statistics,
CSV export, etc.) work correctly without needing a real OpenFOAM case.
"""

import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from foampilot.postprocess.openfoam_pyvista import (
    FoamPostProcessing,
    NumpyEncoder,
)

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def _make_cube_mesh():
    """
    Build a simple uniform-grid cube (10x10x10 cells, 0–10 m on each side)
    with synthetic 'U' (vector) and 'p' (scalar) point data.
    """
    mesh = pv.ImageData(
        dimensions=(11, 11, 11),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )

    # Synthetic velocity field: flowing in +x with some variation
    points = mesh.points
    u = 10.0 + points[:, 1] * 0.1          # x-component ~10 m/s + shear
    v = points[:, 2] * 0.05                 # y-component small upward
    w = np.zeros_like(points[:, 0])         # z-component = 0
    mesh.point_data["U"] = np.column_stack([u, v, w])

    # Synthetic pressure field: decreasing linearly in x
    mesh.point_data["p"] = 100.0 - points[:, 0] * 0.5

    # Cell data variants (filled with zeros for tests that expect cell data)
    mesh.cell_data["U"] = np.zeros((mesh.n_cells, 3))
    mesh.cell_data["p"] = np.zeros(mesh.n_cells)

    return mesh


def _make_structure(mesh):
    """
    Build a structure dict mimicking FoamPostProcessing.get_structure().
    """
    # Create boundary patches as small surface meshes
    slice_xy = mesh.slice("z")  # bottom face
    structure = {
        "cell": mesh,
        "boundaries": {
            "bottom": slice_xy,
        },
    }
    return structure


@pytest.fixture
def foam_post(tmp_path):
    """
    Create a FoamPostProcessing instance pointing at a temporary case path.
    """
    fp = FoamPostProcessing(case_path=str(tmp_path / "fake_case"))
    return fp


@pytest.fixture
def cube_structure():
    """
    Provide a cube-mesh structure dict for tests.
    """
    mesh = _make_cube_mesh()
    return _make_structure(mesh)


# ---------------------------------------------------------------------------
# export_plot tests
# ---------------------------------------------------------------------------

def test_export_plot_writes_png(foam_post, cube_structure, tmp_path):
    """export_plot should write a valid PNG file."""
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(cube_structure["cell"], scalars="p", show_scalar_bar=True)
    out = tmp_path / "contour_test.png"
    foam_post.export_plot(pl, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_export_plot_renders_nonblank(foam_post, cube_structure, tmp_path):
    """The exported image must contain more than a single colour (not all black/white)."""
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(cube_structure["cell"], scalars="p", show_scalar_bar=True)
    pl.set_background("white")
    out = tmp_path / "contour_colours.png"
    foam_post.export_plot(pl, out)

    img = Image.open(out)
    arr = np.array(img)
    unique = len(np.unique(arr.reshape(-1, arr.shape[-1]), axis=0))
    assert unique > 10, f"Expected >10 unique colours, got {unique}"


# ---------------------------------------------------------------------------
# plot_slice tests
# ---------------------------------------------------------------------------

def test_plot_slice_offscreen_writes_png(foam_post, cube_structure, tmp_path):
    """plot_slice in off-screen mode should write a PNG."""
    out = tmp_path / "slice_test.png"
    foam_post.plot_slice(
        structure=cube_structure,
        plane="z",
        scalars="p",
        opacity=0.25,
        path_filename=out,
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_slice_renders_nonblank(foam_post, cube_structure, tmp_path):
    """The slice image should have many colours (not all black)."""
    out = tmp_path / "slice_colours.png"
    foam_post.plot_slice(
        structure=cube_structure,
        plane="z",
        scalars="p",
        opacity=0.25,
        path_filename=out,
    )
    img = Image.open(out)
    arr = np.array(img)
    unique = len(np.unique(arr.reshape(-1, arr.shape[-1]), axis=0))
    assert unique > 10, f"Expected >10 unique colours, got {unique}"


def test_plot_slice_inline_mode(foam_post, cube_structure):
    """plot_slice with no path_filename should create and return a plotter."""
    from unittest.mock import patch
    with patch.object(pv.Plotter, "show", return_value=None):
        pl = foam_post.plot_slice(
            structure=cube_structure,
            plane="z",
            scalars="p",
            path_filename=None,
        )
    assert pl is not None


# ---------------------------------------------------------------------------
# Vector plot (glyph) tests
# ---------------------------------------------------------------------------

def test_vector_glyph_scales_with_domain(foam_post, cube_structure, tmp_path):
    """Glyph factor should be derived from domain size, not a fixed tiny value."""
    mesh = cube_structure["cell"]
    bounds = mesh.bounds
    domain_length = max(
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    )
    glyph_factor = domain_length * 0.001
    assert glyph_factor > 0.001  # 10 m domain → factor ≈ 0.01

    pl = pv.Plotter(off_screen=True)
    mesh.set_active_vectors("U")
    arrows = mesh.glyph(orient="U", factor=glyph_factor, clamping=True)
    pl.add_mesh(arrows, color="blue")
    pl.add_mesh(mesh, opacity=0.1, color="white")

    out = tmp_path / "vector_test.png"
    foam_post.export_plot(pl, out)
    assert out.exists()
    assert out.stat().st_size > 0

    # Verify arrows are non-trivial in size
    assert arrows.n_cells > 0


def test_vector_glyph_subsample(foam_post, tmp_path):
    """Subsampling should reduce glyph count for large meshes."""
    # Create a large mesh (50x50x50 = 125k cells)
    big_mesh = pv.ImageData(
        dimensions=(51, 51, 51),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    points = big_mesh.points
    big_mesh.point_data["U"] = np.column_stack([
        np.full(points.shape[0], 10.0),
        np.zeros(points.shape[0]),
        np.zeros(points.shape[0]),
    ])
    big_mesh.point_data["p"] = 100.0 - points[:, 0] * 0.5

    n_cells = big_mesh.n_cells
    max_glyphs = 2000
    step = max(1, n_cells // max_glyphs)
    subsample_indices = np.arange(0, n_cells, step)
    reduced = big_mesh.extract_cells(subsample_indices)

    assert reduced.n_cells < n_cells
    assert reduced.n_cells <= n_cells // step + 1


# ---------------------------------------------------------------------------
# get_mesh_statistics tests
# ---------------------------------------------------------------------------

def test_get_mesh_statistics(foam_post, cube_structure):
    """get_mesh_statistics should return expected keys and values."""
    stats = foam_post.get_mesh_statistics(cube_structure["cell"])
    assert "num_points" in stats
    assert "num_cells" in stats
    assert "bounds" in stats
    assert stats["num_points"] == 1331  # 11^3
    assert stats["num_cells"] == 1000   # 10^3


# ---------------------------------------------------------------------------
# get_region_statistics tests
# ---------------------------------------------------------------------------

def test_get_region_statistics_scalar(foam_post, cube_structure):
    """get_region_statistics should compute mean/min/max/std for a scalar field."""
    stats = foam_post.get_region_statistics(cube_structure, "cell", "p")
    assert "mean" in stats
    assert "min" in stats
    assert "max" in stats
    assert "std" in stats
    # Pressure ranges from 100 to ~95
    assert 95 <= stats["max"] <= 100
    assert 95 <= stats["mean"] <= 100


def test_get_region_statistics_vector(foam_post, cube_structure):
    """get_region_statistics should handle vector fields (U)."""
    stats = foam_post.get_region_statistics(cube_structure, "cell", "U")
    assert "mean" in stats
    assert "min" in stats
    assert "max" in stats
    assert "std" in stats


# ---------------------------------------------------------------------------
# export_region_data_to_csv tests
# ---------------------------------------------------------------------------

def test_export_region_data_to_csv(foam_post, cube_structure, tmp_path):
    """export_region_data_to_csv should write a CSV with expected columns."""
    out = tmp_path / "test_data.csv"
    foam_post.export_region_data_to_csv(
        cube_structure, "cell", ["U", "p"], out
    )
    assert out.exists()

    import pandas as pd
    df = pd.read_csv(out)
    assert "X" in df.columns
    assert "Y" in df.columns
    assert "Z" in df.columns
    assert "U_0" in df.columns
    assert "U_1" in df.columns
    assert "U_2" in df.columns
    assert "p" in df.columns
    assert len(df) == cube_structure["cell"].n_points


# ---------------------------------------------------------------------------
# export_statistics_to_json tests
# ---------------------------------------------------------------------------

def test_export_statistics_to_json(foam_post, tmp_path):
    """export_statistics_to_json should write valid JSON."""
    stats = {
        "mesh_stats": {"num_points": 1331, "num_cells": 1000},
        "cell_region_stats_U": {"mean": [10.0, 0.0, 0.0], "min": 0, "max": 20, "std": 1.0},
        "cell_region_stats_p": {"mean": 97.5, "min": 95, "max": 100, "std": 1.5},
    }
    out = tmp_path / "stats.json"
    foam_post.export_statistics_to_json(stats, out)
    assert out.exists()

    import json
    with open(out) as f:
        data = json.load(f)
    assert "mesh_stats" in data
    assert data["mesh_stats"]["num_points"] == 1331


# ---------------------------------------------------------------------------
# calculate_q_criterion tests
# ---------------------------------------------------------------------------

def test_calculate_q_criterion(foam_post, cube_structure):
    """calculate_q_criterion should add 'q_criterion' point data."""
    mesh = cube_structure["cell"].copy()
    mesh_with_q = foam_post.calculate_q_criterion(mesh=mesh, velocity_field="U")
    assert "q_criterion" in mesh_with_q.point_data
    q = mesh_with_q.point_data["q_criterion"]
    assert q.shape[0] == mesh.n_points


# ---------------------------------------------------------------------------
# calculate_vorticity tests
# ---------------------------------------------------------------------------

def test_calculate_vorticity(foam_post, cube_structure):
    """calculate_vorticity should add 'vorticity' point data."""
    mesh = cube_structure["cell"].copy()
    mesh_with_vort = foam_post.calculate_vorticity(mesh=mesh, velocity_field="U")
    assert "vorticity" in mesh_with_vort.point_data
    vort = mesh_with_vort.point_data["vorticity"]
    assert vort.shape == (mesh.n_points, 3)


# ---------------------------------------------------------------------------
# NumpyEncoder tests
# ---------------------------------------------------------------------------

def test_numpy_encoder():
    """NumpyEncoder should serialize numpy types to JSON."""
    import json
    data = {
        "int_val": np.int64(42),
        "float_val": np.float64(3.14),
        "array_val": np.array([1, 2, 3]),
    }
    serialized = json.dumps(data, cls=NumpyEncoder)
    deserialized = json.loads(serialized)
    assert deserialized["int_val"] == 42
    assert deserialized["float_val"] == 3.14
    assert deserialized["array_val"] == [1, 2, 3]
