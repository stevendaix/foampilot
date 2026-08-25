import pytest
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from foampilot.urban.generation import UrbGENConfig, generate_urbgen


SITE = Polygon([(0, 0), (160, 0), (160, 120), (0, 120)])


def test_all_six_tower_typologies_generate_valid_massings():
    for mode in range(6):
        result = generate_urbgen(SITE, UrbGENConfig(bcr=0.12, far=1.5, setback=8, tower_typology_mode=mode, podium_floors=0, seed=10))
        assert result.tower_footprints
        assert all(p.is_valid and SITE.buffer(-8).covers(p) for p in result.tower_footprints)
        assert all(b.attributes["typology"] == mode for b in result.model.buildings())


def test_union_bcr_does_not_double_count_podium():
    result = generate_urbgen(SITE, UrbGENConfig(bcr=0.20, far=2.0, podium_floors=2, seed=11))
    expected = unary_union([*result.tower_footprints, *result.podium_footprints]).area / SITE.area
    assert result.actual_bcr == pytest.approx(expected)
    assert result.actual_bcr <= 1.0


def test_far_is_close_to_target():
    result = generate_urbgen(SITE, UrbGENConfig(bcr=0.16, far=2.5, podium_floors=2, height_variation=0.0, seed=12))
    assert abs(result.actual_far - 2.5) < 0.15


def test_courtyard_mode_creates_perimeter_blocks():
    result = generate_urbgen(SITE, UrbGENConfig(bcr=0.20, far=2.0, setback=8, tower_typology_mode=7, courtyard_break_count=4, podium_floors=0, seed=2))
    assert len(result.tower_footprints) >= 2
    assert all(result.buildable_site.covers(p) for p in result.tower_footprints)
    assert result.diagnostics["tower_count"] == len(result.tower_footprints)


def test_seed_is_reproducible_and_changes_layout():
    a = generate_urbgen(SITE, UrbGENConfig(bcr=0.15, tower_typology_mode=6, seed=7, podium_floors=0))
    b = generate_urbgen(SITE, UrbGENConfig(bcr=0.15, tower_typology_mode=6, seed=7, podium_floors=0))
    c = generate_urbgen(SITE, UrbGENConfig(bcr=0.15, tower_typology_mode=6, seed=8, podium_floors=0))
    assert [p.wkt for p in a.tower_footprints] == [p.wkt for p in b.tower_footprints]
    assert [p.wkt for p in a.tower_footprints] != [p.wkt for p in c.tower_footprints]


def test_height_regulation_podium_and_rotation_metadata():
    result = generate_urbgen(SITE, UrbGENConfig(bcr=0.18, far=3, setback=8, tower_typology_mode=0, global_rotation_mode=1, uniform_rotation_deg=35, podium_floors=2, min_building_height=6, max_building_height=12, enforce_height_regulation=True, seed=3))
    assert result.podium_footprints
    assert result.podium_gfa > 0
    towers = [b for b in result.model.buildings() if b.source == "urbgen"]
    assert all(6 <= b.height <= 12 for b in towers)
    assert all(b.attributes["angle_deg"] == 35 for b in towers)


def test_explicit_centroids_are_used_when_valid():
    points = [Point(30, 30), Point(90, 30), Point(30, 90)]
    result = generate_urbgen(SITE, UrbGENConfig(bcr=0.08, setback=5, tower_typology_mode=0, podium_floors=0), centroids=points)
    assert result.tower_footprints
    assert any(p.centroid.distance(points[0]) < 15 for p in result.tower_footprints)
