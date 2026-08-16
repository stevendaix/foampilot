from pathlib import Path
from typing import Optional
import logging

import pyvista as pv

from foampilot.urban.model.urban_model import UrbanModel
from foampilot.urban.model.terrain import CFDTerrain
from foampilot.urban.terrain.processor import TerrainProcessor, TerrainConfig
from foampilot.urban.geometry.building_extruder import BuildingExtruder, BuildingConfig
from foampilot.urban.snappy_config import DomainConfig, SnappyMeshConfig

logger = logging.getLogger(__name__)


class SnappyCaseBuilder:
    """Build an OpenFOAM case for snappyHexMesh from UrbanModel + terrain."""

    def __init__(
        self,
        case_dir: Path,
        urban: UrbanModel,
        terrain: CFDTerrain,
        solver,
        domain_config: DomainConfig,
        terrain_config: TerrainConfig,
        building_config: BuildingConfig,
        mesh_config: SnappyMeshConfig,
    ):
        self.case_dir = Path(case_dir)
        self.urban = urban
        self.terrain = terrain
        self.solver = solver
        self.domain_config = domain_config
        self.terrain_config = terrain_config
        self.building_config = building_config
        self.mesh_config = mesh_config

        self.tri_surface_dir = self.case_dir / "constant" / "triSurface"
        self.tri_surface_dir.mkdir(parents=True, exist_ok=True)

    def write_stl(self) -> tuple[Path, Path]:
        """Generate terrain.stl and buildings.stl."""
        terrain_path = self.tri_surface_dir / "terrain.stl"
        buildings_path = self.tri_surface_dir / "buildings.stl"

        terrain_processor = TerrainProcessor(self.terrain, self.terrain_config)
        terrain_processor.export_stl(terrain_path)

        extruder = BuildingExtruder(self.urban.buildings(), self.terrain, self.building_config)
        extruder.export_stl(buildings_path)

        return terrain_path, buildings_path

    def _compute_domain_from_stl(self):
        terrain_path = self.tri_surface_dir / "terrain.stl"
        buildings_path = self.tri_surface_dir / "buildings.stl"

        bounds = None
        for path in [terrain_path, buildings_path]:
            if not path.exists():
                continue
            mesh = pv.read(str(path))
            if bounds is None:
                bounds = list(mesh.bounds)
            else:
                bounds[0] = min(bounds[0], mesh.bounds[0])
                bounds[1] = min(bounds[1], mesh.bounds[1])
                bounds[2] = min(bounds[2], mesh.bounds[2])
                bounds[3] = max(bounds[3], mesh.bounds[3])
                bounds[4] = max(bounds[4], mesh.bounds[4])
                bounds[5] = max(bounds[5], mesh.bounds[5])

        if bounds is None:
            raise RuntimeError("No STL files found to compute domain bounds")

        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        dc = self.domain_config
        xmin -= dc.margin_x
        xmax += dc.margin_x
        ymin -= dc.margin_y
        ymax += dc.margin_y
        zmin -= dc.bottom_margin
        zmax += dc.top_margin
        return xmin, ymin, zmin, xmax, ymax, zmax

    def configure_snappy(self):
        """Configure SnappyMesher from urban model and STL bounds."""
        try:
            from foampilot.mesh.snappymesh import SnappyMesher
        except ImportError as exc:
            raise RuntimeError("pyvista/snappy mesh module is required") from exc

        mesher = SnappyMesher(case_path=self.case_dir)
        mesher.add_geometry("terrain", self.tri_surface_dir / "terrain.stl")
        mesher.add_geometry("buildings", self.tri_surface_dir / "buildings.stl")

        mesher.castellatedMeshControls["refinementSurfaces"] = {
            "terrain": {"level": (self.mesh_config.terrain_refinement_level, self.mesh_config.terrain_refinement_level)},
            "buildings": {"level": (self.mesh_config.building_refinement_level, self.mesh_config.building_refinement_level)},
        }
        mesher.castellatedMeshControls["maxGlobalCells"] = self.mesh_config.max_global_cells
        mesher.castellatedMeshControls["nCellsBetweenLevels"] = self.mesh_config.n_cells_between_walls
        mesher.addLayers = self.mesh_config.add_layers

        xmin, ymin, zmin, xmax, ymax, zmax = self._compute_domain_from_stl()
        mesher.locationInMesh = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, zmax - 1.0)

        return mesher

    def write(self, run_mesh: bool = False) -> Path:
        """Generate all OpenFOAM files and optionally run mesh pipeline."""
        self.write_stl()
        mesher = self.configure_snappy()
        mesher.write_block_mesh_dict(padding=0.0, base_cell_size=self.mesh_config.base_cell_size)
        mesher.write_snappyHexMeshDict()
        mesher.write_surface_features_dict(stl_list_for_emesh=["terrain.stl", "buildings.stl"])

        if self.solver is not None:
            self.solver.setup_case()
            self.solver.constant.write()
            self.solver.system.write()

        if run_mesh:
            self.build_mesh(mesher)

        return self.case_dir

    def build_mesh(self, mesher=None):
        """Run full mesh pipeline: blockMesh -> surfaceFeatures -> snappyHexMesh."""
        if mesher is None:
            mesher = self.configure_snappy()
        mesher.run_block_mesh()
        mesher.run_surface_feature_extract()
        mesher.run()
