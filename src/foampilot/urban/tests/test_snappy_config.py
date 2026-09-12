from pathlib import Path
from foampilot.urban.snappy_config import (
    TerrainConfig,
    BuildingConfig,
    DomainConfig,
    SnappyMeshConfig,
)


def test_terrain_config_defaults():
    cfg = TerrainConfig()
    assert cfg.dem_resolution == 5.0
    assert cfg.bottom_offset == 20.0
    assert cfg.simplify_tolerance == 0.5


def test_building_config_defaults():
    cfg = BuildingConfig()
    assert cfg.default_height == 9.0
    assert cfg.foundation_depth == 1.0


def test_domain_config_defaults():
    cfg = DomainConfig()
    assert cfg.margin_x == 100.0
    assert cfg.top_margin == 100.0


def test_snappy_mesh_config_defaults():
    cfg = SnappyMeshConfig()
    assert cfg.building_refinement_level == 3
    assert cfg.add_layers is False
