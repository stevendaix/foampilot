import pytest
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from foampilot.urban.generation import UrbGENConfig, generate_urbgen


SITE = Polygon([(0, 0), (160, 0), (160, 120), (0, 120)])


def test_multi_site_api_returns_deterministic_keyed_results():
    from shapely.affinity import translate
    from foampilot.urban.generation import generate_urbgen_multi_site
    sites = {"west": SITE, "east": translate(SITE, xoff=200)}
    first = generate_urbgen_multi_site(sites, UrbGENConfig(bcr=0.10, far=1.5, podium_floors=0, seed=30))
    second = generate_urbgen_multi_site(sites, UrbGENConfig(bcr=0.10, far=1.5, podium_floors=0, seed=30))
    assert list(first) == ["west", "east"]
    assert [p.wkt for p in first["west"].tower_footprints] == [p.wkt for p in second["west"].tower_footprints]
    assert first["west"].diagnostics["seed"] == 30
    assert first["east"].diagnostics["seed"] == 31


def test_max_feasible_boundary_move_preserves_containment():
    from foampilot.urban.generation.urbgen import _move_to_boundary
    region = SITE.buffer(-8)
    tower = Polygon([(60, 50), (72, 50), (72, 62), (60, 62)])
    moved = _move_to_boundary(tower, region, radial=True)
    assert region.covers(moved)
    assert moved.centroid.distance(region.boundary) <= tower.centroid.distance(region.boundary) + 1e-6


def test_max_feasible_podium_move_preserves_containment():
    from foampilot.urban.generation.urbgen import _move_tower_to_podium_edge
    region = SITE.buffer(-8)
    tower = Polygon([(60, 50), (72, 50), (72, 62), (60, 62)])
    podium = tower.buffer(18).intersection(region)
    moved = _move_tower_to_podium_edge(tower, podium, region)
    assert podium.covers(moved)
    assert region.covers(moved)


def test_bcr_convergence_on_convex_and_concave_sites():
    sites = [
        SITE,
        Polygon([(0, 0), (160, 0), (160, 120), (95, 120), (95, 65), (0, 65)]),
    ]
    for site in sites:
        result = generate_urbgen(site, UrbGENConfig(bcr=0.18, upper_bcr=0.20, far=1.8, setback=5, podium_floors=1, seed=19))
        assert result.actual_bcr <= 0.20 * 1.02
        assert all(result.buildable_site.covers(p) for p in result.tower_footprints)
        assert result.diagnostics["shrink_iterations"] >= 0
        assert result.diagnostics["expand_iterations"] >= 0
        assert result.diagnostics["converged_bcr"] == pytest.approx(result.actual_bcr, abs=1e-9)


def test_ringcache_arc_mapping_and_disjoint_segments():
    from foampilot.urban.generation.urbgen import RingCache, _courtyard_ring_segments
    zone = SITE.buffer(-8)
    cache = RingCache(zone, 12.0, samples=128)
    assert cache.ok
    assert cache.total > 0
    assert cache.inner.area < zone.area
    assert cache.param_at_arc(cache.total + 10.0) == pytest.approx(10.0)
    segments = _courtyard_ring_segments(zone, 12.0, 4, 18.0, 0.7, 0.0, 42)
    assert len(segments) == 4
    assert all(s.is_valid and zone.covers(s) for s in segments)
    assert all(segments[i].intersection(segments[j]).area < 1e-6 for i in range(len(segments)) for j in range(i + 1, len(segments)))


def test_default_config_matches_original_gha_contract():
    c = UrbGENConfig()
    assert c.bcr == 0.50
    assert c.upper_bcr == pytest.approx(0.55)
    assert c.far == 3.0
    assert c.min_width == 12.0
    assert c.min_footprint_per_tower == 80.0
    assert c.max_footprint_per_tower == 350.0
    assert c.min_tower_distance == 12.0
    assert c.tower_typology_mode == 0
    assert c.arm_length_ratio == 1.3
    assert c.floor_height == 3.5
    assert c.height_variation == 0.0
    assert c.max_building_height == 100.0


def test_typology_primitives_are_explicit_and_area_consistent():
    from foampilot.urban.generation.urbgen import (
        angle_candidates, estimate_extra_area, get_typology_modules,
        max_length_for_typology, typology_arm_count,
    )
    for mode in range(6):
        modules = get_typology_modules(24.0, 8.0, mode, 8.0)
        assert modules and modules[0] == (0.0, 0.0, 24.0, 8.0)
        assert typology_arm_count(mode) == len(modules) - 1
        assert estimate_extra_area(mode, 8.0, 24.0) >= 0
        assert max_length_for_typology(mode, 8.0, 6.0) >= 8.0
    assert angle_candidates(1, 37.0) == [37.0]
    assert len(angle_candidates(3)) == 12


def test_upper_bcr_is_exposed_as_active_limit():
    result = generate_urbgen(SITE, UrbGENConfig(bcr=0.10, upper_bcr=0.14, far=1.5, podium_floors=0, seed=4))
    assert result.target_bcr == 0.10
    assert result.actual_bcr <= 0.14 * 1.02


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
