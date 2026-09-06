from shapely.geometry import Polygon
import pytest

from foampilot.urban.generation import UrbGENConfig, generate_urbgen


def test_generate_urbgen_respects_buildable_site_and_spacing():
    site = Polygon([(0, 0), (120, 0), (120, 80), (0, 80)])
    result = generate_urbgen(
        site,
        UrbGENConfig(bcr=0.20, far=2.0, setback=5.0, min_width=8.0, min_tower_distance=6.0, podium_floors=0),
    )
    assert result.model.building_count() > 0
    assert result.actual_bcr <= result.target_bcr + 1e-9
    assert all(result.buildable_site.covers(b.footprint) for b in result.model.buildings())
    for i, first in enumerate(result.model.buildings()):
        for second in result.model.buildings()[i + 1 :]:
            assert first.footprint.distance(second.footprint) >= 6.0 - 1e-9


def test_generate_urbgen_derives_height_from_far():
    site = Polygon([(0, 0), (100, 100), (0, 100)])
    result = generate_urbgen(site, UrbGENConfig(bcr=0.10, far=3.0, setback=3.0, tower_typology_mode=0))
    assert result.actual_far >= 0.10
    assert all(b.attributes.get("typology_name", "I") == "I" for b in result.model.buildings() if b.source == "urbgen")
    assert all(b.height >= result.model.buildings()[0].height * 0.5 for b in result.model.buildings() if b.source == "urbgen")


def test_invalid_targets_fail_fast():
    with pytest.raises(ValueError):
        UrbGENConfig(bcr=1.1)
    with pytest.raises(ValueError):
        generate_urbgen(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), UrbGENConfig(setback=1.0))
