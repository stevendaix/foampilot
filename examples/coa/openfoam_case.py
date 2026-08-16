#!/usr/bin/env python3
"""
OpenFOAM case builder from TBAD geometry.
Complete pipeline: STL/NIfTI → CAD → Mesh → OpenFOAM case.
"""
import shutil
import logging
from pathlib import Path
from typing import Optional, Union, List

from foampilot import Meshing, ValueWithUnit, Solver, Boundary

logger = logging.getLogger(__name__)


class OpenFOAMCaseBuilder:
    def __init__(self, case_dir: Path):
        self.case_dir = Path(case_dir)
        self.case_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_solver(self, steady: bool = True, turbulence: str = "laminar"):
        from foampilot import ValueWithUnit
        
        rho = ValueWithUnit(1060, "kg/m^3")
        nu = ValueWithUnit(3.77e-6, "m^2/s")
        
        self.solver = Solver(self.case_dir)
        self.solver.compressible = False
        self.solver.with_gravity = False
        self.solver.transient = not steady
        self.solver.turbulence_model = turbulence
        self.solver.constant.transportProperties.nu = nu
        self.solver.constant.transportProperties.rho = rho
        
        self.solver.system.controlDict.application = "foamRun"
        if steady:
            self.solver.system.controlDict.startTime = 0
            self.solver.system.controlDict.endTime = 1
            self.solver.system.controlDict.deltaT = 1
            self.solver.system.controlDict.writeInterval = 1
            self.solver.system.controlDict.adjustTimeStep = False
        
        self.solver.system.fvSolution.solvers["p"]["solver"] = "smoothSolver"
        self.solver.system.fvSolution.solvers["p"]["smoother"] = "GAMG"
        self.solver.system.fvSolution.solvers["U"]["solver"] = "smoothSolver"
        
        self.solver.system.write()
        self.solver.constant.write()
        
    def setup_mesh_snappy(self, stl_file: Path, location_in_mesh: tuple = (-16, -21, -12),
                          refinement: int = 1, layers: int = 3, layer_thickness: float = 0.2):
        stl_dest = self.case_dir / "constant" / "triSurface"
        stl_dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(stl_file, stl_dest / stl_file.name)
        
        self.mesh = Meshing(self.case_dir, mesher="snappy")
        snappy = self.mesh.mesher
        snappy.stl_file = stl_file.name
        snappy.locationInMesh = location_in_mesh
        snappy.geometry = {
            "wall_aorta": {
                "type": "triSurfaceMesh",
                "file": stl_file.name,
                "name": "wall_aorta",
            }
        }
        snappy.castellatedMeshControls["refinementSurfaces"] = {
            "wall_aorta": {"level": (refinement - 1, refinement)}
        }
        snappy.addLayers = True
        snappy.add_layer("wall_aorta", layers)
        snappy.addLayersControls["finalLayerThickness"] = layer_thickness
        self.mesh.write()
        snappy.run()
        
    def setup_boundary_conditions(self, inlet_patches: List[str] = None,
                                   outlet_patches: List[str] = None,
                                   wall_patches: List[str] = None):
        self.solver.boundary.initialize_boundary()
        
        wall_patches = wall_patches or ["wall_aorta"]
        for patch in wall_patches:
            self.solver.boundary.fields["U"][patch] = {"type": "noSlip"}
            self.solver.boundary.fields["p"][patch] = {"type": "zeroGradient"}
            
        if inlet_patches:
            for patch in inlet_patches:
                self.solver.boundary.fields["U"][patch] = {
                    "type": "timeVaryingMappedFixedValue",
                    "offset": "(0 0 0)",
                    "setAverage": "false"
                }
                self.solver.boundary.fields["p"][patch] = {"type": "zeroGradient"}
                
        if outlet_patches:
            for patch in outlet_patches:
                self.solver.boundary.fields["U"][patch] = {"type": "zeroGradient"}
                self.solver.boundary.fields["p"][patch] = {"type": "fixedValue", "value": "uniform 0"}
                
        self.solver.boundary.write_boundary_conditions()
        
    def build(self, stl_file: Path, **kwargs):
        self.setup_solver()
        self.setup_mesh_snappy(stl_file)
        self.setup_boundary_conditions()
        logger.info(f"Case built at {self.case_dir}")
        return self.case_dir


def build_case_from_cad(stl_path: Path, case_dir: Path, **kwargs):
    """High-level function to build complete OpenFOAM case from STL."""
    builder = OpenFOAMCaseBuilder(case_dir)
    return builder.build(stl_path, **kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    base = Path("/home/steven/foampilot/examples/coa")
    stl = base / "data_preproc" / "tbad_stl_output" / "tbad_TL_walls.stl"
    case = base / "openfoam_cases" / "tbad_simple"
    build_case_from_cad(stl, case)
