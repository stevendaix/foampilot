import numpy as np
from foampilot.urban.model.terrain import CFDTerrain
from foampilot.urban.terrain.processor import TerrainProcessor, TerrainConfig


def _flat_terrain(z=2.0):
    terrain = CFDTerrain.flat(z=z, extent=(0.0, 0.0, 10.0, 10.0))
    return terrain


def test_terrain_processor_builds_surface():
    processor = TerrainProcessor(_flat_terrain(2.0), TerrainConfig(dem_resolution=2.0, bottom_offset=1.0))
    surface = processor.build_closed_surface()
    assert surface is not None
    assert surface.n_points > 0
    assert surface.n_cells > 0


def test_terrain_processor_export_stl(tmp_path):
    out = tmp_path / "terrain.stl"
    processor = TerrainProcessor(_flat_terrain(1.5), TerrainConfig(dem_resolution=2.0, bottom_offset=1.0))
    result = processor.export_stl(out)
    assert result.exists()
    assert result.stat().st_size > 0


def test_terrain_processor_flat_terrain_is_closed():
    processor = TerrainProcessor(_flat_terrain(0.0), TerrainConfig(dem_resolution=2.0, bottom_offset=1.0))
    surface = processor.build_closed_surface()
    assert surface is not None
    assert surface.is_manifold
