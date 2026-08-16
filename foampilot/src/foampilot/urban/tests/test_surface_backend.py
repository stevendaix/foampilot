"""Tests for SurfaceQuarterBuilder and snappyHexMesh backend."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from foampilot.urban import (
    Building,
    UrbanModel,
    CFDDomain,
    WindFrame,
    CFDLOD,
    CFDSimplifier,
    SurfaceQuarterBuilder,
    MeshConfig,
    WakeRefinement,
    RefinementRegion,
    BoundaryLayerConfig,
)
from shapely.geometry import Polygon


@pytest.fixture
def simple_urban():
    urban = UrbanModel()
    urban.add_building(Building(
        id="B1",
        footprint=Polygon([(0, 0), (20, 0), (20, 10), (0, 10)]),
        ground_z=0.0,
        roof_z=15.0,
    ))
    urban.add_building(Building(
        id="B2",
        footprint=Polygon([(30, 0), (50, 0), (50, 10), (30, 10)]),
        ground_z=0.0,
        roof_z=20.0,
    ))
    return urban


@pytest.fixture
def domain():
    return CFDDomain(
        upstream=5.0,
        downstream=10.0,
        lateral=3.0,
        top=2.0,
        extent_units="href",
        reference_height_method="Hmax",
    )


class TestSurfaceQuarterBuilder:
    def test_build_creates_stl(self, simple_urban, domain, tmp_path):
        wind_frame = WindFrame(direction_deg=0.0, origin=(25.0, 5.0, 0.0))
        geometry = CFDSimplifier(simple_urban, lod=CFDLOD.LOD1).simplify(
            wind_frame=wind_frame,
            domain=domain,
        )

        builder = SurfaceQuarterBuilder(tmp_path / "case", geometry)
        builder.build()

        assert builder._built is True
        assert builder._stl_path is not None
        assert builder._stl_path.exists()
        assert builder._stl_path.suffix == ".stl"

    def test_build_mesh_requires_build_first(self, simple_urban, domain, tmp_path):
        wind_frame = WindFrame(direction_deg=0.0, origin=(25.0, 5.0, 0.0))
        geometry = CFDSimplifier(simple_urban, lod=CFDLOD.LOD1).simplify(
            wind_frame=wind_frame,
            domain=domain,
        )

        builder = SurfaceQuarterBuilder(tmp_path / "case", geometry)
        with pytest.raises(RuntimeError, match="build\\(\\) must be called"):
            builder.build_mesh(MeshConfig())

    def test_build_mesh_writes_dicts(self, simple_urban, domain, tmp_path):
        wind_frame = WindFrame(direction_deg=0.0, origin=(25.0, 5.0, 0.0))
        geometry = CFDSimplifier(simple_urban, lod=CFDLOD.LOD1).simplify(
            wind_frame=wind_frame,
            domain=domain,
        )

        builder = SurfaceQuarterBuilder(tmp_path / "case", geometry)
        builder.build()
        builder.build_mesh(MeshConfig(
            global_size=10.0,
            building_size=1.0,
            max_size=30.0,
            algorithm_3d=1,
        ))

        assert (tmp_path / "case" / "system" / "snappyHexMeshDict").exists()
        assert (tmp_path / "case" / "system" / "blockMeshDict").exists()

    def test_build_mesh_with_wake_refinement(self, simple_urban, domain, tmp_path):
        wind_frame = WindFrame(direction_deg=0.0, origin=(25.0, 5.0, 0.0))
        geometry = CFDSimplifier(simple_urban, lod=CFDLOD.LOD1).simplify(
            wind_frame=wind_frame,
            domain=domain,
        )

        builder = SurfaceQuarterBuilder(tmp_path / "case", geometry)
        builder.build()
        builder.build_mesh(MeshConfig(
            global_size=10.0,
            building_size=1.0,
            max_size=30.0,
            wake_refinement=WakeRefinement(
                length=5.0,
                width=2.0,
                height=2.0,
                target_size=1.0,
            ),
        ))

        content = (tmp_path / "case" / "system" / "snappyHexMeshDict").read_text()
        assert "refinementRegions" in content

    def test_build_mesh_with_boundary_layers(self, simple_urban, domain, tmp_path):
        wind_frame = WindFrame(direction_deg=0.0, origin=(25.0, 5.0, 0.0))
        geometry = CFDSimplifier(simple_urban, lod=CFDLOD.LOD1).simplify(
            wind_frame=wind_frame,
            domain=domain,
        )

        builder = SurfaceQuarterBuilder(tmp_path / "case", geometry)
        builder.build()
        builder.build_mesh(MeshConfig(
            global_size=10.0,
            building_size=1.0,
            max_size=30.0,
            boundary_layers=BoundaryLayerConfig(
                first_layer_height=0.05,
                growth_rate=1.2,
                num_layers=5,
                patches=["ground", "buildings"],
            ),
        ))

        content = (tmp_path / "case" / "system" / "snappyHexMeshDict").read_text()
        assert "addLayers true;" in content
        assert "layers" in content

    def test_run_requires_build_first(self, simple_urban, domain, tmp_path):
        wind_frame = WindFrame(direction_deg=0.0, origin=(25.0, 5.0, 0.0))
        geometry = CFDSimplifier(simple_urban, lod=CFDLOD.LOD1).simplify(
            wind_frame=wind_frame,
            domain=domain,
        )

        builder = SurfaceQuarterBuilder(tmp_path / "case", geometry)
        with pytest.raises(RuntimeError, match="build\\(\\) must be called"):
            builder.run()

    def test_export_openfoam_returns_path(self, simple_urban, domain, tmp_path):
        wind_frame = WindFrame(direction_deg=0.0, origin=(25.0, 5.0, 0.0))
        geometry = CFDSimplifier(simple_urban, lod=CFDLOD.LOD1).simplify(
            wind_frame=wind_frame,
            domain=domain,
        )

        builder = SurfaceQuarterBuilder(tmp_path / "case", geometry)
        path = builder.export_openfoam()
        assert path == tmp_path / "case" / "constant" / "polyMesh"
