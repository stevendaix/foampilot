from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from foampilot.mesh.snappymesh import SnappyMesher


@dataclass
class SnappyExportConfig:
    location_in_mesh: tuple[float, float, float]
    surface_refinement: tuple[int, int] = (2, 3)
    padding: float = 0.20
    base_cell_size: float = 0.75
    n_surface_layers: int = 5
    first_layer_thickness: float = 0.12
    expansion_ratio: float = 1.2


class MedicalSnappyExporter:
    """Generate an OpenFOAM snappyHexMesh case from Python-only CFD patches."""

    def __init__(self, config: SnappyExportConfig):
        self.config = config

    @staticmethod
    def _header(object_name: str, location: str) -> str:
        return f'''FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    location "{location}";
    object {object_name};
}}
'''

    def _write_fields(self, case: Path, outlets: list[str]) -> None:
        (case / "system" / "controlDict").write_text(
            self._header("controlDict", "system")
            + "application foamRun;\nsolver incompressibleFluid;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\n"
        )
        (case / "constant" / "transportProperties").write_text(
            self._header("transportProperties", "constant")
            + "transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] 3.77e-06;\n"
        )
        (case / "system" / "fvSchemes").write_text(
            self._header("fvSchemes", "system")
            + "ddtSchemes { default steadyState; }\ngraduSchemes { default Gauss linear; }\ndivSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U); }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\nwallDist { method meshWave; }\n"
        )
        (case / "system" / "fvSolution").write_text(
            self._header("fvSolution", "system")
            + "solvers { p { solver GAMG; tolerance 1e-7; relTol 0; } U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0; } }\nSIMPLE { nNonOrthogonalCorrectors 1; }\n"
        )
        outlet_u = "\n".join(f"    {name} {{ type zeroGradient; }}" for name in outlets)
        outlet_p = "\n".join(f"    {name} {{ type fixedValue; value uniform 0; }}" for name in outlets)
        (case / "0" / "U").write_text(
            self._header("U", "0")
            + f"dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\nboundaryField\n{{\n    inlet {{ type fixedValue; value uniform (0 0 0); }}\n{outlet_u}\n    wall {{ type noSlip; }}\n}}\n"
        )
        (case / "0" / "p").write_text(
            self._header("p", "0")
            + f"dimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{{\n    inlet {{ type zeroGradient; }}\n{outlet_p}\n    wall {{ type zeroGradient; }}\n}}\n"
        )

    def export(self, patch_dir: str | Path, case_dir: str | Path) -> Path:
        patch_dir = Path(patch_dir)
        case = Path(case_dir)
        tri = case / "constant" / "triSurface"
        tri.mkdir(parents=True, exist_ok=True)
        (case / "system").mkdir(parents=True, exist_ok=True)
        (case / "0").mkdir(parents=True, exist_ok=True)

        outlet_files = sorted(patch_dir.glob("outlet_*.stl"))
        required = [patch_dir / "inlet.stl", patch_dir / "wall.stl"]
        missing = [str(path) for path in required if not path.exists()]
        if missing or not outlet_files:
            raise FileNotFoundError(f"Missing CFD patches: {missing}; outlets={len(outlet_files)}")
        patch_names = ["inlet"] + [path.stem for path in outlet_files] + ["wall"]
        for name in patch_names:
            shutil.copy2(patch_dir / f"{name}.stl", tri / f"{name}.stl")

        mesher = SnappyMesher(case_path=case, stl_file=None, castellatedMesh=True, snap=True, addLayers=True)
        for name in patch_names:
            mesher.add_geometry(name, tri / f"{name}.stl")
        mesher.locationInMesh = self.config.location_in_mesh
        mesher.castellatedMeshControls["locationInMesh"] = self.config.location_in_mesh
        mesher.castellatedMeshControls["refinementSurfaces"] = {
            name: {"level": self.config.surface_refinement} for name in patch_names
        }
        final_layer = self.config.first_layer_thickness * self.config.expansion_ratio ** max(self.config.n_surface_layers - 1, 0)
        mesher.addLayersControls.update({
            "relativeSizes": False,
            "expansionRatio": self.config.expansion_ratio,
            "firstLayerThickness": self.config.first_layer_thickness,
            "finalLayerThickness": final_layer,
            "minThickness": self.config.first_layer_thickness * 0.35,
        })
        mesher.add_layer("wall", self.config.n_surface_layers)
        mesher.write_block_mesh_dict(padding=self.config.padding, base_cell_size=self.config.base_cell_size)
        mesher.write_surface_features_dict(patch_names, included_angle=30)
        mesher.write()
        self._write_fields(case, [path.stem for path in outlet_files])
        return case
