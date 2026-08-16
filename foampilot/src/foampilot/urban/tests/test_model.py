from pathlib import Path
import tempfile
import pytest
from shapely.geometry import Polygon

from foampilot.urban.model import Building, UrbanModel, CFDDomain, WindFrame
from foampilot.urban.simplification import CFDLOD, RoofType


def test_building_height_is_computed():
    b = Building(
        id="B1",
        footprint=Polygon([(0, 0), (10, 0), (10, 5), (0, 5)]),
        ground_z=0.0,
        roof_z=12.5,
    )
    assert b.height == 12.5


def test_building_invalid_footprint_raises():
    with pytest.raises(ValueError):
        Building(
            id="B1",
            footprint=Polygon([(0, 0), (10, 0), (5, 5), (10, 10), (0, 10), (5, 5)]),
            ground_z=0.0,
            roof_z=12.5,
        )


def test_building_roof_z_must_exceed_ground_z():
    with pytest.raises(ValueError):
        Building(
            id="B1",
            footprint=Polygon([(0, 0), (10, 0), (10, 5), (0, 5)]),
            ground_z=10.0,
            roof_z=5.0,
        )


def test_urban_model_add_and_count():
    urban = UrbanModel()
    b1 = Building(
        id="B1",
        footprint=Polygon([(0, 0), (10, 0), (10, 5), (0, 5)]),
        ground_z=0.0,
        roof_z=12.5,
    )
    b2 = Building(
        id="B2",
        footprint=Polygon([(20, 0), (30, 0), (30, 8), (20, 8)]),
        ground_z=0.0,
        roof_z=23.0,
    )
    urban.add_building(b1)
    urban.add_building(b2)
    assert urban.building_count() == 2
    assert urban.buildings()[0].id == "B1"


def test_urban_model_bbox():
    urban = UrbanModel()
    urban.add_building(Building(
        id="B1",
        footprint=Polygon([(0, 0), (10, 0), (10, 5), (0, 5)]),
        ground_z=0.0,
        roof_z=12.5,
    ))
    xmin, ymin, zmin, xmax, ymax, zmax = urban.bbox()
    assert xmin == 0.0
    assert ymin == 0.0
    assert zmin == 0.0
    assert xmax == 10.0
    assert ymax == 5.0
    assert zmax == 12.5


def test_wind_frame_roundtrip():
    frame = WindFrame(direction_deg=30.0, origin=(10.0, 20.0, 0.0))
    p_world = (123.4, 567.8, 12.0)
    p_local = frame.to_local(*p_world)
    p_back = frame.to_world(*p_local)
    assert abs(p_back[0] - p_world[0]) < 1e-9
    assert abs(p_back[1] - p_world[1]) < 1e-9
    assert abs(p_back[2] - p_world[2]) < 1e-9


def test_cfd_domain_compute_box():
    urban = UrbanModel()
    urban.add_building(Building(
        id="B1",
        footprint=Polygon([(0, 0), (10, 0), (10, 5), (0, 5)]),
        ground_z=0.0,
        roof_z=12.5,
    ))
    domain = CFDDomain(upstream=2.0, downstream=4.0, lateral=1.0, top=1.0, extent_units="meters")
    xmin, ymin, zmin, xmax, ymax, zmax = domain.compute_box(urban)
    assert xmin == -2.0
    assert xmax == 14.0
    assert zmax == 13.5


def test_cfd_domain_compute_box_with_wind_frame():
    urban = UrbanModel()
    urban.add_building(Building(
        id="B1",
        footprint=Polygon([(0, 0), (10, 0), (10, 5), (0, 5)]),
        ground_z=0.0,
        roof_z=12.5,
    ))
    frame = WindFrame(direction_deg=90.0, origin=(0.0, 0.0, 0.0))
    domain = CFDDomain(upstream=2.0, downstream=4.0, lateral=1.0, top=1.0, extent_units="meters")
    xmin, ymin, zmin, xmax, ymax, zmax = domain.compute_box(urban, wind_frame=frame)
    assert ymin == -11.0
    assert ymax == pytest.approx(1.0)
    assert xmin == -2.0
    assert xmax == 9.0


def test_urban_model_geojson_roundtrip():
    import json
    urban = UrbanModel()
    urban.add_building(Building(
        id="B1",
        footprint=Polygon([(0, 0), (10, 0), (10, 5), (0, 5)]),
        ground_z=0.0,
        roof_z=12.5,
        source="manual",
    ))
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as f:
        path = Path(f.name)
    try:
        urban.to_geojson(path)
        loaded = UrbanModel.from_geojson(path)
        assert loaded.building_count() == 1
        assert loaded.buildings()[0].id == "B1"
        assert loaded.buildings()[0].height == 12.5
    finally:
        path.unlink()
