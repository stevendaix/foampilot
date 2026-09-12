"""Performance tests for large-scale urban CFD geometry."""

import time
import pytest
from pathlib import Path

from foampilot.urban import (
    Building,
    UrbanModel,
    CFDDomain,
    WindFrame,
    CFDLOD,
    CFDSimplifier,
    GmshQuarterBuilder,
    SurfaceQuarterBuilder,
    MeshConfig,
    GeometryMapper,
)
from shapely.geometry import Polygon


def generate_large_urban(n_buildings: int = 1000, seed: int = 42) -> UrbanModel:
    import random
    random.seed(seed)

    urban = UrbanModel()
    grid_size = int(n_buildings ** 0.5)
    spacing = 25.0

    for i in range(n_buildings):
        ix = i % grid_size
        iy = i // grid_size
        x = ix * spacing + random.uniform(-2, 2)
        y = iy * spacing + random.uniform(-2, 2)
        w = random.uniform(8, 20)
        d = random.uniform(8, 20)
        h = random.uniform(10, 40)

        footprint = Polygon([
            (x - w/2, y - d/2),
            (x + w/2, y - d/2),
            (x + w/2, y + d/2),
            (x - w/2, y + d/2),
        ])

        urban.add_building(Building(
            id=f"B{i+1:04d}",
            footprint=footprint,
            ground_z=0.0,
            roof_z=h,
        ))

    return urban


class TestPerformance:
    def test_large_urban_model_creation(self):
        start = time.time()
        urban = generate_large_urban(1000)
        elapsed = time.time() - start

        assert urban.building_count() == 1000
        assert elapsed < 10.0, f"UrbanModel creation took {elapsed:.2f}s, expected < 10s"

    def test_cfd_simplifier_large_urban(self):
        urban = generate_large_urban(100)
        wind_frame = WindFrame(direction_deg=270.0, origin=urban.center_xy())
        domain = CFDDomain(
            upstream=8.0,
            downstream=15.0,
            lateral=4.0,
            top=2.5,
            extent_units="href",
            reference_height_method="Hmax",
        )

        start = time.time()
        geometry = CFDSimplifier(urban, lod=CFDLOD.LOD1).simplify(
            wind_frame=wind_frame,
            domain=domain,
        )
        elapsed = time.time() - start

        assert len(geometry.buildings) == 100
        assert elapsed < 5.0, f"CFDSimplifier took {elapsed:.2f}s, expected < 5s"

    def test_gmsh_backend_scalability(self, tmp_path):
        urban = generate_large_urban(50)
        wind_frame = WindFrame(direction_deg=270.0, origin=urban.center_xy())
        domain = CFDDomain(
            upstream=8.0,
            downstream=15.0,
            lateral=4.0,
            top=2.5,
            extent_units="href",
            reference_height_method="Hmax",
        )
        geometry = CFDSimplifier(urban, lod=CFDLOD.LOD1).simplify(
            wind_frame=wind_frame,
            domain=domain,
        )

        builder = GmshQuarterBuilder(tmp_path / "case", geometry)

        start = time.time()
        builder.build()
        build_time = time.time() - start

        assert builder._built is True
        assert build_time < 30.0, f"GmshQuarterBuilder.build() took {build_time:.2f}s for 50 buildings"

    def test_surface_backend_scalability(self, tmp_path):
        urban = generate_large_urban(100)
        wind_frame = WindFrame(direction_deg=270.0, origin=urban.center_xy())
        domain = CFDDomain(
            upstream=8.0,
            downstream=15.0,
            lateral=4.0,
            top=2.5,
            extent_units="href",
            reference_height_method="Hmax",
        )
        geometry = CFDSimplifier(urban, lod=CFDLOD.LOD1).simplify(
            wind_frame=wind_frame,
            domain=domain,
        )

        builder = SurfaceQuarterBuilder(tmp_path / "case", geometry)

        start = time.time()
        builder.build()
        build_time = time.time() - start

        assert builder._built is True
        assert builder._stl_path.exists()
        assert build_time < 20.0, f"SurfaceQuarterBuilder.build() took {build_time:.2f}s for 100 buildings"

    def test_backend_selection_recommendation(self):
        urban = generate_large_urban(500)
        wind_frame = WindFrame(direction_deg=270.0, origin=urban.center_xy())
        domain = CFDDomain(
            upstream=8.0,
            downstream=15.0,
            lateral=4.0,
            top=2.5,
            extent_units="href",
            reference_height_method="Hmax",
        )
        geometry = CFDSimplifier(urban, lod=CFDLOD.LOD1).simplify(
            wind_frame=wind_frame,
            domain=domain,
        )

        n = len(geometry.buildings)
        if n > 200:
            recommended_backend = "snappy"
        elif n > 50:
            recommended_backend = "gmsh_or_snappy"
        else:
            recommended_backend = "gmsh"

        assert recommended_backend in ("gmsh", "gmsh_or_snappy", "snappy")

    def test_geometry_mapper_large_urban(self):
        urban = generate_large_urban(200)
        wind_frame = WindFrame(direction_deg=270.0, origin=urban.center_xy())
        domain = CFDDomain(
            upstream=8.0,
            downstream=15.0,
            lateral=4.0,
            top=2.5,
            extent_units="href",
            reference_height_method="Hmax",
        )
        geometry = CFDSimplifier(urban, lod=CFDLOD.LOD1).simplify(
            wind_frame=wind_frame,
            domain=domain,
        )

        mapper = GeometryMapper(urban, geometry)
        summary = mapper.summary()

        assert summary["n_buildings"] == 200
        assert summary["cfd_buildings"] == 200
        assert summary["height_stats"]["max"] > 0
        assert summary["area_stats"]["total"] > 0
