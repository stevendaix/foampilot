#!/usr/bin/env python3
"""
Synthetic fallback for the neighborhood demo when VoxCity/EE is unavailable.
"""

import numpy as np
from shapely.geometry import Polygon

from foampilot.urban import Building, UrbanModel
from foampilot.urban.model.terrain import CFDTerrain


def build_synthetic_urban(config: dict):
    """Fallback: generate a synthetic grid neighborhood similar to VoxCity output."""
    urban = UrbanModel()
    rng = np.random.default_rng(42)

    n = 25
    street_width = 15.0
    cols = 5
    rows = 5
    cell_size = 40.0
    domain_width = cols * cell_size + street_width
    domain_depth = rows * cell_size + street_width
    origin_x = -domain_width / 2
    origin_y = -domain_depth / 2

    building_id = 0
    for i in range(rows):
        for j in range(cols):
            if building_id >= n:
                break
            width = rng.uniform(12.0, 28.0)
            depth = rng.uniform(12.0, 28.0)
            height = rng.uniform(10.0, 45.0)
            x = origin_x + street_width / 2 + j * cell_size + rng.uniform(-2, 2)
            y = origin_y + street_width / 2 + i * cell_size + rng.uniform(-2, 2)

            footprint = Polygon([
                (x - width / 2, y - depth / 2),
                (x + width / 2, y - depth / 2),
                (x + width / 2, y + depth / 2),
                (x - width / 2, y + depth / 2),
            ])
            urban.add_building(Building(
                id=f"B{building_id + 1:03d}",
                footprint=footprint,
                ground_z=0.0,
                roof_z=height,
                source="synthetic",
                confidence=1.0,
            ))
            building_id += 1

    terrain = CFDTerrain.flat(z=0.0)
    return urban, terrain
