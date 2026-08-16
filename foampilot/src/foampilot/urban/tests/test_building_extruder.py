from foampilot.urban.model.urban_model import Building, UrbanModel
from foampilot.urban.geometry.building_extruder import BuildingExtruder, BuildingConfig
from foampilot.urban.model.terrain import CFDTerrain


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
    urban.add_building(Building(
        id="b2",
        footprint=_square(6.0, 2.0, 4.0, 4.0),
        ground_z=0.0,
        roof_z=12.0,
        source="test",
        confidence=1.0,
        attributes={},
    ))
    return urban


def _square(xmin, ymin, dx, dy):
    from shapely.geometry import Polygon
    return Polygon([(xmin, ymin), (xmin+dx, ymin), (xmin+dx, ymin+dy), (xmin, ymin+dy)])


def test_building_extruder_builds_solids():
    urban = _simple_urban()
    extruder = BuildingExtruder(urban.buildings(), CFDTerrain.flat(), BuildingConfig())
    solids = extruder.build_solids()
    assert len(solids) == 2
    for solid in solids:
        assert solid.n_points > 0
        assert solid.n_cells > 0


def test_building_extruder_export_stl(tmp_path):
    out = tmp_path / "buildings.stl"
    urban = _simple_urban()
    extruder = BuildingExtruder(urban.buildings(), CFDTerrain.flat(), BuildingConfig())
    result = extruder.export_stl(out)
    assert result.exists()
    assert result.stat().st_size > 0


def test_building_extruder_skips_invalid_buildings():
    urban = UrbanModel()
    urban.add_building(Building(
        id="invalid",
        footprint=_square(0.0, 0.0, 0.5, 0.5),
        ground_z=0.0,
        roof_z=1.0,
        source="test",
        confidence=1.0,
        attributes={},
    ))
    extruder = BuildingExtruder(urban.buildings(), CFDTerrain.flat(), BuildingConfig(min_area=1.0))
    solids = extruder.build_solids()
    assert len(solids) == 0
