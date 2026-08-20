from pathlib import Path

import trimesh

from foampilot.geometry.medical_build import MedicalSnappyExporter, SnappyExportConfig


def make_patch_dir(path: Path) -> Path:
    path.mkdir()
    trimesh.creation.cylinder(radius=1.0, height=0.05, sections=16).export(path / "inlet.stl")
    trimesh.creation.cylinder(radius=0.8, height=0.05, sections=16).export(path / "outlet_0.stl")
    trimesh.creation.cylinder(radius=0.7, height=0.05, sections=16).export(path / "outlet_1.stl")
    trimesh.creation.cylinder(radius=2.0, height=4.0, sections=24).export(path / "wall.stl")
    return path


def test_snappy_export_writes_case_and_wall_layers(tmp_path):
    patch_dir = make_patch_dir(tmp_path / "patches")
    case_dir = tmp_path / "case"
    exporter = MedicalSnappyExporter(
        SnappyExportConfig(location_in_mesh=(0.0, 0.0, 0.0), n_surface_layers=4)
    )
    exporter.export(patch_dir, case_dir)
    snappy = (case_dir / "system" / "snappyHexMeshDict").read_text()
    assert "addLayers true;" in snappy
    assert '"wall"' in snappy
    assert "nSurfaceLayers 4;" in snappy
    assert (case_dir / "constant" / "triSurface" / "inlet.stl").exists()
    assert (case_dir / "constant" / "triSurface" / "outlet_1.stl").exists()
    assert "nu [0 2 -1 0 0 0 0]" in (case_dir / "constant" / "transportProperties").read_text()
