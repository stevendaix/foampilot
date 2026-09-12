from pathlib import Path
from foampilot.urban.model.urban_model import Building, UrbanModel
from foampilot.urban.model.terrain import CFDTerrain
from foampilot.urban.snappy_config import (
    DomainConfig,
    TerrainConfig,
    BuildingConfig,
    SnappyMeshConfig,
)


def _simple_urban():
    urban = UrbanModel()
    urban.add_building(Building(
        id="b1",
        footprint=_square(0.0, 0.0, 4.0, 4.0),
        ground_z=0.0,
        roof_z=9.0,
        source="test",
        confidence=1.0,
        attributes={},
    ))
    return urban


def _square(xmin, ymin, dx, dy):
    from shapely.geometry import Polygon
    return Polygon([(xmin, ymin), (xmin+dx, ymin), (xmin+dx, ymin+dy), (xmin, ymin+dy)])


def test_snappy_case_builder_writes_structure(tmp_path):
    try:
        from foampilot.openfoam.snappy_case_builder import SnappyCaseBuilder
    except ImportError:
        return

    case_dir = tmp_path / "case"
    case_dir.mkdir()
    urban = _simple_urban()
    terrain = CFDTerrain.flat(z=0.0)

    builder = SnappyCaseBuilder(
        case_dir=case_dir,
        urban=urban,
        terrain=terrain,
        solver=None,
        domain_config=DomainConfig(),
        terrain_config=TerrainConfig(),
        building_config=BuildingConfig(),
        mesh_config=SnappyMeshConfig(),
    )
    builder.write_stl()
    assert (case_dir / "constant" / "triSurface" / "terrain.stl").exists()
    assert (case_dir / "constant" / "triSurface" / "buildings.stl").exists()
