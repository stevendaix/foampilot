"""
Test the OpenFOAMDirectReader and CHTDirectReader.

These tests verify that OpenFOAM cases can be read directly into
PyVista meshes without requiring foamToVTK.
"""

import sys
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from foampilot.postprocess.openfoam_direct import (
    OpenFOAMDirectReader,
    CHTDirectReader,
    read_openfoam,
    read_cht_openfoam,
    _detect_regions,
    _read_field,
)

TEST_DIR = Path(__file__).resolve().parent
PLANAR_CASE = TEST_DIR.parent.parent / "planarPoiseuille"
CHT_CASE = TEST_DIR.parent.parent / "examples" / "cht" / "simple_heated_duct"


def test_detect_regions_cht():
    regions = _detect_regions(CHT_CASE)
    assert "fluid" in regions
    assert "solid" in regions


def test_detect_regions_single():
    regions = _detect_regions(PLANAR_CASE)
    assert len(regions) == 0


def test_openfoam_direct_reader_single_region():
    reader = OpenFOAMDirectReader(PLANAR_CASE)
    assert reader.points.shape[1] == 3
    assert reader.mesh.n_points > 0
    assert reader.mesh.n_cells > 0
    assert "left" in reader.boundary_patches
    assert reader.get_latest_time() == "9"
    assert "0" in reader.get_time_steps()


def test_openfoam_direct_reader_read_field():
    reader = OpenFOAMDirectReader(PLANAR_CASE)
    U = reader.read_field("U", time_step="1")
    assert U.shape == (40, 3)
    assert np.max(np.linalg.norm(U, axis=1)) > 0


def test_openfoam_direct_reader_attach_field():
    reader = OpenFOAMDirectReader(PLANAR_CASE)
    mesh = reader.to_pyvista(fields=["U"], time_step="1")
    assert "U" in mesh.cell_data
    assert mesh.cell_data["U"].shape == (40, 3)


def test_cht_direct_reader_detect_regions():
    reader = CHTDirectReader(CHT_CASE)
    assert reader.region_names == ["fluid", "solid"]
    assert reader.regions["fluid"] == "fluid"
    assert reader.regions["solid"] == "solid"


def test_cht_direct_reader_get_mesh():
    reader = CHTDirectReader(CHT_CASE)
    fluid_mesh = reader.get_mesh(region="fluid", fields=["T"], time_step="0.1")
    assert fluid_mesh.n_points == 4692
    assert fluid_mesh.n_cells == 2250
    assert "T" in fluid_mesh.cell_data
    assert fluid_mesh.cell_data["T"].min() > 290


def test_cht_direct_reader_get_all_meshes():
    reader = CHTDirectReader(CHT_CASE)
    mb = reader.get_all_meshes(fields=["T"], time_step="0.1")
    assert mb.n_blocks == 2
    assert "fluid" in mb.keys()
    assert "solid" in mb.keys()


def test_read_openfoam_convenience():
    mesh = read_openfoam(PLANAR_CASE, fields=["U"], time_step="1")
    assert isinstance(mesh, pv.UnstructuredGrid)
    assert mesh.n_cells == 40


def test_read_cht_openfoam_convenience():
    mb = read_cht_openfoam(CHT_CASE, fields=["T"], time_step="0.1")
    assert isinstance(mb, pv.MultiBlock)
    assert mb.n_blocks == 2


def test_read_field_scalar():
    reader = OpenFOAMDirectReader(CHT_CASE, region="fluid")
    T = reader.read_field("T", time_step="0.1")
    assert T.ndim == 1
    assert len(T) == 2250


def test_read_field_vector():
    reader = OpenFOAMDirectReader(CHT_CASE, region="fluid")
    U = reader.read_field("U", time_step="0.1")
    assert U.ndim == 2
    assert U.shape == (2250, 3)


def test_field_header_parsing():
    U_path = PLANAR_CASE / "1" / "U"
    vals, is_point = _read_field(U_path)
    assert is_point is False
    assert vals.shape == (40, 3)


def test_attach_field_auto_location(tmp_path):
    """as_point_data=None should auto-detect field location from header."""
    case = tmp_path / "case"
    polymesh = case / "constant" / "polyMesh"
    polymesh.mkdir(parents=True)

    (polymesh / "points").write_text(
        "/*--------------------------------*- C++ -*----------------------------------*\\\\\n"
        "  =========                 |\n"
        "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n"
        "   \\\\    /   O peration     | Website:  https://openfoam.org\n"
        "    \\\\  /    A nd           | Version:  13\n"
        "     \\\\/     M anipulation  |\n"
        "\\*---------------------------------------------------------------------------*/\n"
        "FoamFile\n"
        "{\n"
        "    format      ascii;\n"
        "    class       vectorField;\n"
        '    location    "constant/polyMesh";\n'
        "    object      points;\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
        "\n"
        "8\n"
        "(\n"
        "(0 0 0)\n"
        "(1 0 0)\n"
        "(1 1 0)\n"
        "(0 1 0)\n"
        "(0 0 1)\n"
        "(1 0 1)\n"
        "(1 1 1)\n"
        "(0 1 1)\n"
        ")\n"
    )

    (polymesh / "faces").write_text(
        "/*--------------------------------*- C++ -*----------------------------------*\\\\\n"
        "  =========                 |\n"
        "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n"
        "   \\\\    /   O peration     | Website:  https://openfoam.org\n"
        "    \\\\  /    A nd           | Version:  13\n"
        "     \\\\/     M anipulation  |\n"
        "\\*---------------------------------------------------------------------------*/\n"
        "FoamFile\n"
        "{\n"
        "    format      ascii;\n"
        "    class       faceList;\n"
        '    location    "constant/polyMesh";\n'
        "    object      faces;\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
        "\n"
        "4\n"
        "(\n"
        "3(0 1 2)\n"
        "3(0 1 3)\n"
        "3(1 2 3)\n"
        "3(0 2 3)\n"
        ")\n"
    )

    (polymesh / "owner").write_text(
        "/*--------------------------------*- C++ -*----------------------------------*\\\\\n"
        "  =========                 |\n"
        "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n"
        "   \\\\    /   O peration     | Website:  https://openfoam.org\n"
        "    \\\\  /    A nd           | Version:  13\n"
        "     \\\\/     M anipulation  |\n"
        "\\*---------------------------------------------------------------------------*/\n"
        "FoamFile\n"
        "{\n"
        "    format      ascii;\n"
        "    class       labelList;\n"
        '    note        "nPoints:8  nCells:1  nFaces:4  nInternalFaces:0";\n'
        '    location    "constant/polyMesh";\n'
        "    object      owner;\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
        "\n"
        "4\n"
        "(\n"
        "0\n"
        "0\n"
        "0\n"
        "0\n"
        ")\n"
    )

    time_dir = case / "0"
    time_dir.mkdir()

    (time_dir / "U").write_text(
        "/*--------------------------------*- C++ -*----------------------------------*\\\\\n"
        "  =========                 |\n"
        "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n"
        "   \\\\    /   O peration     | Website:  https://openfoam.org\n"
        "    \\\\  /    A nd           | Version:  13\n"
        "     \\\\/     M anipulation  |\n"
        "\\*---------------------------------------------------------------------------*/\n"
        "FoamFile\n"
        "{\n"
        "    format      ascii;\n"
        "    class       volVectorField;\n"
        '    location    "0";\n'
        "    object      U;\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
        "\n"
        "dimensions      [0 1 -1 0 0 0 0];\n"
        "\n"
        "internalField   nonuniform List<vector> \n"
        "1\n"
        "(\n"
        "(1 0 0)\n"
        ")\n"
        "\n"
        "boundaryField\n"
        "{\n"
        "    patch1\n"
        "    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n"
        "    }\n"
        "}\n"
    )

    (time_dir / "pointP").write_text(
        "/*--------------------------------*- C++ -*----------------------------------*\\\\\n"
        "  =========                 |\n"
        "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n"
        "   \\\\    /   O peration     | Website:  https://openfoam.org\n"
        "    \\\\  /    A nd           | Version:  13\n"
        "     \\\\/     M anipulation  |\n"
        "\\*---------------------------------------------------------------------------*/\n"
        "FoamFile\n"
        "{\n"
        "    format      ascii;\n"
        "    class       pointScalarField;\n"
        '    location    "0";\n'
        "    object      pointP;\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
        "\n"
        "dimensions      [0 2 -2 0 0 0 0];\n"
        "\n"
        "internalField   nonuniform List<scalar> \n"
        "8\n"
        "(\n"
        "0.0\n"
        "1.0\n"
        "2.0\n"
        "3.0\n"
        "4.0\n"
        "5.0\n"
        "6.0\n"
        "7.0\n"
        ")\n"
        "\n"
        "boundaryField\n"
        "{\n"
        "    patch1\n"
        "    {\n"
        "        type            calculated;\n"
        "        value           uniform 0;\n"
        "    }\n"
        "}\n"
    )

    (time_dir / "pointQ").write_text(
        "/*--------------------------------*- C++ -*----------------------------------*\\\\\n"
        "  =========                 |\n"
        "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n"
        "   \\\\    /   O peration     | Website:  https://openfoam.org\n"
        "    \\\\  /    A nd           | Version:  13\n"
        "     \\\\/     M anipulation  |\n"
        "\\*---------------------------------------------------------------------------*/\n"
        "FoamFile\n"
        "{\n"
        "    format      ascii;\n"
        "    class       pointScalarField;\n"
        '    location    "0";\n'
        "    object      pointQ;\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
        "\n"
        "dimensions      [0 2 -2 0 0 0 0];\n"
        "\n"
        "internalField   nonuniform List<scalar> \n"
        "1\n"
        "(\n"
        "3.0\n"
        ")\n"
        "\n"
        "boundaryField\n"
        "{\n"
        "    patch1\n"
        "    {\n"
        "        type            calculated;\n"
        "        value           uniform 0;\n"
        "    }\n"
        "}\n"
    )

    # Use a fresh reader per assertion so the mesh is not mutated across calls
    reader = OpenFOAMDirectReader(case)

    mesh = reader.attach_field("U")
    assert "U" in mesh.cell_data
    assert mesh.cell_data["U"].shape == (1, 3)
    assert "U" not in mesh.point_data

    reader = OpenFOAMDirectReader(case)
    mesh = reader.attach_field("pointP")
    assert "pointP" in mesh.point_data
    assert mesh.point_data["pointP"].shape == (8,)
    assert "pointP" not in mesh.cell_data

    reader = OpenFOAMDirectReader(case)
    mesh = reader.attach_field("pointQ", as_point_data=False)
    assert "pointQ" in mesh.cell_data
    assert mesh.cell_data["pointQ"].shape == (1,)
    assert "pointQ" not in mesh.point_data

    reader = OpenFOAMDirectReader(case)
    mesh = reader.attach_field("U", as_point_data=True)
    assert "U" in mesh.point_data
    assert mesh.point_data["U"].shape == (8, 3)
    assert "U" not in mesh.cell_data
