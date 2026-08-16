"""Tests for urban geometry mapping, mesh sizing, and boundary layers."""

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
    GmshQuarterBuilder,
    MeshConfig,
    WakeRefinement,
    RefinementRegion,
    BoundaryLayerConfig,
    GeometryMapper,
)
from foampilot.urban.model.terrain import CFDTerrain
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


class TestGeometryMapper:
    def test_summary_basic(self, simple_urban):
        mapper = GeometryMapper(simple_urban)
        summary = mapper.summary()

        assert summary["n_buildings"] == 2
        assert summary["height_stats"]["max"] == 20.0
        assert summary["height_stats"]["min"] == 15.0
        assert summary["area_stats"]["total"] > 0

    def test_summary_with_geometry(self, simple_urban, domain):
        wind_frame = WindFrame(direction_deg=0.0, origin=(25.0, 5.0, 0.0))
        geometry = CFDSimplifier(simple_urban, lod=CFDLOD.LOD1).simplify(
            wind_frame=wind_frame,
            domain=domain,
        )

        mapper = GeometryMapper(simple_urban, geometry)
        summary = mapper.summary()

        assert summary["cfd_buildings"] == 2
        assert "domain_box" in summary

    def test_save_summary(self, simple_urban, tmp_path):
        mapper = GeometryMapper(simple_urban)
        output = tmp_path / "geometry_summary.json"
        mapper.save_summary(output)

        assert output.exists()
        import json
        with open(output) as f:
            data = json.load(f)
        assert data["n_buildings"] == 2

    def test_plot_footprints(self, simple_urban):
        mapper = GeometryMapper(simple_urban)
        ax = mapper.plot_footprints()
        assert ax is not None

    def test_plot_domain(self, simple_urban, domain):
        wind_frame = WindFrame(direction_deg=0.0, origin=(25.0, 5.0, 0.0))
        geometry = CFDSimplifier(simple_urban, lod=CFDLOD.LOD1).simplify(
            wind_frame=wind_frame,
            domain=domain,
        )

        mapper = GeometryMapper(simple_urban, geometry)
        ax = mapper.plot_domain()
        assert ax is not None

    def test_plot_3d(self, simple_urban):
        mapper = GeometryMapper(simple_urban)
        ax = mapper.plot_3d()
        assert ax is not None

    def test_save_plots(self, simple_urban, tmp_path):
        mapper = GeometryMapper(simple_urban)
        saved = mapper.save_plots(tmp_path, prefix="test")

        assert len(saved) == 3
        for p in saved:
            assert p.exists()


class TestMeshConfig:
    def test_default_values(self):
        config = MeshConfig()
        assert config.global_size.get_in("m") == 15.0
        assert config.building_size.get_in("m") == 2.0
        assert config.wake_size.get_in("m") == 4.0
        assert config.ground_size.get_in("m") == 2.0
        assert config.min_size.get_in("m") == 0.1
        assert config.max_size.get_in("m") == 50.0
        assert config.algorithm_3d == 1

    def test_wake_refinement(self):
        wr = WakeRefinement(length=8.0, width=3.0, height=2.0, target_size=1.5)
        config = MeshConfig(wake_refinement=wr)
        assert config.wake_refinement.length == 8.0
        assert config.wake_refinement.target_size.get_in("m") == 1.5

    def test_boundary_layer_config(self):
        bl = BoundaryLayerConfig(
            first_layer_height=0.03,
            growth_rate=1.15,
            num_layers=8,
            patches=["ground", "buildings"],
        )
        config = MeshConfig(boundary_layers=bl)
        assert config.boundary_layers.first_layer_height.get_in("m") == 0.03
        assert config.boundary_layers.num_layers == 8

    def test_refinement_regions(self):
        regions = [
            RefinementRegion(center=(0, 0, 0), size=2.0, radius=5.0),
            RefinementRegion(center=(10, 10, 5), size=1.0),
        ]
        config = MeshConfig(refinement_regions=regions)
        assert len(config.refinement_regions) == 2
        assert config.refinement_regions[0].radius is not None


class TestGmshQuarterBuilderMeshSizing:
    def test_build_mesh_basic(self, simple_urban, domain, tmp_path):
        wind_frame = WindFrame(direction_deg=0.0, origin=(25.0, 5.0, 0.0))
        geometry = CFDSimplifier(simple_urban, lod=CFDLOD.LOD1).simplify(
            wind_frame=wind_frame,
            domain=domain,
        )

        builder = GmshQuarterBuilder(tmp_path / "case", geometry)
        builder.build()
        builder.assign_patches()
        builder.build_mesh(MeshConfig(
            global_size=10.0,
            building_size=1.0,
            max_size=30.0,
            algorithm_3d=4,
        ))
        assert builder._meshed is True

    def test_build_mesh_with_wake_refinement(self, simple_urban, domain, tmp_path):
        wind_frame = WindFrame(direction_deg=0.0, origin=(25.0, 5.0, 0.0))
        geometry = CFDSimplifier(simple_urban, lod=CFDLOD.LOD1).simplify(
            wind_frame=wind_frame,
            domain=domain,
        )

        builder = GmshQuarterBuilder(tmp_path / "case", geometry)
        builder.build()
        builder.assign_patches()
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
        assert builder._meshed is True

    def test_build_mesh_with_refinement_regions(self, simple_urban, domain, tmp_path):
        wind_frame = WindFrame(direction_deg=0.0, origin=(25.0, 5.0, 0.0))
        geometry = CFDSimplifier(simple_urban, lod=CFDLOD.LOD1).simplify(
            wind_frame=wind_frame,
            domain=domain,
        )

        builder = GmshQuarterBuilder(tmp_path / "case", geometry)
        builder.build()
        builder.assign_patches()
        builder.build_mesh(MeshConfig(
            global_size=10.0,
            building_size=1.0,
            max_size=30.0,
            refinement_regions=[
                RefinementRegion(center=(10, 5, 7.5), size=5.0, radius=None),
            ],
        ))
        assert builder._meshed is True

    def test_build_mesh_with_boundary_layers(self, simple_urban, domain, tmp_path):
        wind_frame = WindFrame(direction_deg=0.0, origin=(25.0, 5.0, 0.0))
        geometry = CFDSimplifier(simple_urban, lod=CFDLOD.LOD1).simplify(
            wind_frame=wind_frame,
            domain=domain,
        )

        builder = GmshQuarterBuilder(tmp_path / "case", geometry)
        builder.build()
        builder.assign_patches()
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
        assert builder._meshed is True


class TestTerrainIntegration:
    def test_terrain_flat(self, simple_urban):
        terrain = CFDTerrain.flat(z=0.5)
        assert terrain.get_elevation(0, 0) == 0.5

    def test_terrain_slope(self):
        terrain = CFDTerrain.slope(slope_x=0.1, slope_y=0.2)
        z = terrain.get_elevation(10, 20)
        assert z == 0.1 * 10 + 0.2 * 20

    def test_terrain_in_domain(self, simple_urban):
        terrain = CFDTerrain.flat(z=0.5)
        domain = CFDDomain(
            upstream=5.0,
            downstream=10.0,
            lateral=3.0,
            top=2.0,
            extent_units="href",
            reference_height_method="Hmax",
        )

        xmin, ymin, zmin, xmax, ymax, zmax = domain.compute_box(
            simple_urban,
            wind_frame=WindFrame(direction_deg=0.0, origin=(25.0, 5.0, 0.0)),
            terrain=terrain,
        )

        assert zmin == 0.0
        assert zmax > 15.0


class TestOSMReaderValidation:
    def test_osm_reader_import(self):
        from foampilot.urban.readers import OSMReader
        reader = OSMReader(distance=100)
        assert reader is not None

    def test_osm_reader_small_area(self):
        from foampilot.urban.readers import OSMReader
        reader = OSMReader(distance=50)
        urban = reader.read("Lyon, France")
        assert urban.building_count() > 0
        assert urban.building_count() < 5000
