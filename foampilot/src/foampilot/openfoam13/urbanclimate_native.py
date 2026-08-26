"""Native OpenFOAM-13 dictionary generation for urban climate cases.

This module deliberately generates the case dictionaries from typed profile
configuration. Geometry is an optional external asset; no complete case tree
is copied by the native writer.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from foampilot.base import Meshing
from foampilot.base.cases_variables import CaseFieldsManager
from foampilot.cht.regions import FluidRegion, SolidRegion
from foampilot.openfoam13.physics import PhysicsConfig


@dataclass(frozen=True)
class RegionSpec:
    name: str
    kind: str = "fluid"
    temperature: float = 300.0
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)


class UrbanClimateNativeCaseBuilder:
    """Generate the OpenFOAM case dictionaries for one urban profile."""

    def __init__(self, case_path: str | Path, regions: tuple[RegionSpec, ...], *, profile: str, ham: bool = False, vegetation: bool = False, radiation: bool = False, physics: PhysicsConfig | None = None):
        self.case_path = Path(case_path)
        self.regions = regions
        self.profile = profile
        self.ham = ham
        self.vegetation = vegetation
        self.radiation = radiation
        self.physics = physics

    def write_case(self, *, overwrite: bool = False) -> Path:
        if self.case_path.exists():
            if not overwrite:
                raise FileExistsError(f"Refusing to overwrite existing case: {self.case_path}")
            import shutil
            shutil.rmtree(self.case_path)
        for folder in ("0", "constant", "system"):
            (self.case_path / folder).mkdir(parents=True, exist_ok=True)
        self._write_control()
        self._write_region_properties()
        self._write_system_dictionaries()
        self._write_constant_dictionaries()
        self._write_initial_fields()
        self._write_allrun()
        physics = self.physics or PhysicsConfig(urban={"profile": self.profile, "ham": self.ham, "vegetation": self.vegetation, "radiation": self.radiation})
        physics.write_support_files(self.case_path)
        self._write("foampilotUrbanClimate.json", json.dumps({
            "profile": self.profile,
            "regions": [r.name for r in self.regions],
            "ham": self.ham,
            "vegetation": self.vegetation,
            "radiation": self.radiation,
            "openfoam": {"vendor": physics.openfoam_vendor, "version": physics.openfoam_version},
        }, indent=2))
        return self.case_path

    def _foam_header(self, cls: str, obj: str, location: str | None = None) -> str:
        loc = f'    location    "{location}";\n' if location else ""
        return f"FoamFile\n{{\n    version     2.0;\n    format      ascii;\n    class       {cls};\n{loc}    object      {obj};\n}}\n\n"

    def _write(self, relative: str, content: str) -> None:
        path = self.case_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")

    def _write_control(self) -> None:
        fluids = [r.name for r in self.regions if r.kind == "fluid"]
        solids = [r.name for r in self.regions if r.kind == "solid"]
        vegetation = [r.name for r in self.regions if r.kind == "vegetation"]
        def names(values: list[str]) -> str:
            return "(" + " ".join(values) + ")"
        regions = (
            "regions\n{\n"
            f"    fluid {names(fluids)};\n"
            f"    solid {names(solids)};\n"
            f"    vegetation {names(vegetation)};\n"
            "}\n"
        )
        content = self._foam_header("dictionary", "controlDict", "system") + (
            "application     urbanMicroclimateFoam;\n"
            "startFrom       startTime;\nstartTime       0;\n"
            "stopAt          endTime;\nendTime         1;\ndeltaT          1;\n"
            "writeControl    timeStep;\nwriteInterval   1;\n"
            "maxFluidIteration 10000;\nminFluidIteration 1;\n"
            "libs            ();\n" + regions
        )
        self._write("system/controlDict", content)

    def _write_allrun(self) -> None:
        self._write("Allrun", '''#!/bin/sh
set -eu
cd "${0%/*}" || exit 1
blockMesh
exec urbanMicroclimateFoam
''')
        (self.case_path / "Allrun").chmod(0o755)

    def _write_region_properties(self) -> None:
        fluids = [r.name for r in self.regions if r.kind == "fluid"]
        solids = [r.name for r in self.regions if r.kind == "solid"]
        vegetation = [r.name for r in self.regions if r.kind == "vegetation"]
        content = self._foam_header("dictionary", "regionProperties", "constant")
        content += "fluid\n(\n" + "\n".join(f"    {r};" for r in fluids) + "\n);\nsolid\n(\n" + "\n".join(f"    {r};" for r in solids) + "\n);\n"
        if vegetation:
            content += "vegetation\n(\n" + "\n".join(f"    {r};" for r in vegetation) + "\n);\n"
        self._write("constant/regionProperties", content)

    def _write_system_dictionaries(self) -> None:
        # Use FoamPilot's native mesh writer, as the other generated examples do.
        mesh = Meshing(self.case_path, mesher="blockMesh").mesher
        mesh.vertices = [[0, 0, 0], [70, 0, 0], [70, 100, 0], [0, 100, 0], [0, 0, 60], [70, 0, 60], [70, 100, 60], [0, 100, 60]]
        mesh.blocks = ["hex (0 1 2 3 4 5 6 7) (70 100 60) simpleGrading (1 1 1)"]
        mesh.edges = []
        mesh.defaultPatch = {"defaultFaces": "wall"}
        mesh.boundary = {
            "inlet": {"type": "patch", "faces": [[0, 4, 7, 3]]},
            "outlet": {"type": "patch", "faces": [[1, 2, 6, 5]]},
            "side": {"type": "symmetryPlane", "faces": [[0, 1, 5, 4], [3, 7, 6, 2]]},
            "top": {"type": "patch", "faces": [[4, 5, 6, 7]]},
            "ground": {"type": "wall", "faces": [[0, 3, 2, 1]]},
        }
        mesh.mergePatchPairs = []
        mesh.write(self.case_path / "system" / "blockMeshDict")
        self._write("system/decomposeParDict", self._foam_header("dictionary", "decomposeParDict", "system") + "numberOfSubdomains 1;\nmethod          scotch;\n")
        for region in self.regions:
            base = f"system/{region.name}"
            self._write(f"{base}/fvSchemes", self._foam_header("dictionary", "fvSchemes", base) + "ddtSchemes { default Euler; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; div(phi,U) Gauss linearUpwind grad(U); }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n")
            self._write(f"{base}/fvSolution", self._foam_header("dictionary", "fvSolution", base) + "solvers {}\nPIMPLE {}\n")

    def _write_constant_dictionaries(self) -> None:
        for region in self.regions:
            base = f"constant/{region.name}"
            self._write(f"{base}/g", self._foam_header("uniformDimensionedVectorField", "g", base) + "dimensions [0 1 -2 0 0 0 0];\nvalue (0 0 -9.81);\n")
            if region.kind == "fluid":
                self._write(f"{base}/transportProperties", self._foam_header("dictionary", "transportProperties", base) + "transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] 1e-05;\n")
                self._write(f"{base}/thermophysicalProperties", self._foam_header("dictionary", "thermophysicalProperties", base) + "thermoType { type heRhoThermo; mixture pureMixture; transport const; thermo hConst; equationOfState perfectGas; specie specie; energy sensibleEnthalpy; }\n")
                self._write(f"{base}/momentumTransport", self._foam_header("dictionary", "momentumTransport", base) + "simulationType RAS;\nRAS { RASModel realizableKE; turbulence on; printCoeffs on; }\n")
            elif region.kind == "solid":
                self._write(f"{base}/transportProperties", self._foam_header("dictionary", "transportProperties", base) + "rho [1 -3 0 0 0 0 0] 1800;\nCp [0 2 -2 -1 0 0 0] 900;\nkappa [1 1 -3 -1 0 0 0] 1.4;\n")
                self._write(f"{base}/buildingMaterials", self._foam_header("dictionary", "buildingMaterials", base) + "materials (\n    { name default; buildingMaterialModel solid; rho 1800; cap 900; lambda1 1.4; lambda2 1.4; }\n);\n")
            elif region.kind == "vegetation":
                self._write(f"{base}/vegetationProperties", self._foam_header("dictionary", "vegetationProperties", base) + "vegetationModel simple;\nleafAreaIndex 2.0;\ncanopyHeight 8.0;\n")
                self._write(f"{base}/radiationProperties", self._foam_header("dictionary", "radiationProperties", base) + "absorptionEmissionModel constant;\n")
                self._write(f"{base}/solarLoadProperties", self._foam_header("dictionary", "solarLoadProperties", base) + "solarLoadModel fvDOM;\n")
        if self.radiation:
            self._write("constant/sunPosVector", self._foam_header("uniformDimensionedVectorField", "sunPosVector", "constant") + "dimensions [0 0 0 0 0 0 0];\nvalue (1 1 1);\n")
            self._write("constant/IDN", self._foam_header("uniformDimensionedScalarField", "IDN", "constant") + "dimensions [0 0 0 0 0 0 0];\nvalue 800;\n")
            self._write("constant/Idif", self._foam_header("uniformDimensionedScalarField", "Idif", "constant") + "dimensions [0 0 0 0 0 0 0];\nvalue 100;\n")

    def _write_initial_fields(self) -> None:
        region_objects: list[Any] = []
        for r in self.regions:
            region_objects.append(FluidRegion(r.name, temperature=r.temperature, velocity=r.velocity) if r.kind == "fluid" else SolidRegion(r.name, temperature=r.temperature))
        manager = CaseFieldsManager(energy_activated=True, with_gravity=True, with_radiation=self.radiation, regions=region_objects)
        for r, obj in zip(self.regions, region_objects):
            fields = manager.get_region_field_names(r.name)
            for field in fields:
                cls = "volVectorField" if field == "U" else "volScalarField"
                dims = {"U": "[0 1 -1 0 0 0 0]", "p_rgh": "[1 -1 -2 0 0 0 0]", "T": "[0 0 0 1 0 0 0]", "k": "[0 2 -2 0 0 0 0]", "epsilon": "[0 2 -3 0 0 0 0]", "nut": "[0 2 -1 0 0 0 0]"}.get(field, "[0 0 0 0 0 0 0]")
                value = "(0 0 0)" if field == "U" else str(r.temperature if field == "T" else 0)
                if field == "U":
                    boundary = {
                        "inlet": ("fixedValue", "value uniform (1 0 0)"),
                        "outlet": ("zeroGradient", None),
                        "side": ("symmetryPlane", None),
                        "top": ("zeroGradient", None),
                        "ground": ("noSlip", None),
                    }
                elif field == "p_rgh":
                    boundary = {
                        "inlet": ("zeroGradient", None),
                        "outlet": ("fixedValue", "value uniform 0"),
                        "side": ("symmetryPlane", None),
                        "top": ("zeroGradient", None),
                        "ground": ("zeroGradient", None),
                    }
                elif field == "T":
                    boundary = {
                        "inlet": ("fixedValue", f"value uniform {r.temperature}"),
                        "outlet": ("zeroGradient", None),
                        "side": ("symmetryPlane", None),
                        "top": ("zeroGradient", None),
                        "ground": ("zeroGradient", None),
                    }
                else:
                    boundary = {p: ("zeroGradient", None) for p in ("inlet", "outlet", "side", "top", "ground")}
                bc = ["boundaryField", "{"]
                for patch, (kind, extra) in boundary.items():
                    bc.append(f"    {patch}")
                    bc.append("    {")
                    bc.append(f"        type {kind};")
                    if extra:
                        key, val = extra.split(" ", 1)
                        bc.append(f"        {key} {val};")
                    bc.append("    }")
                bc.append("}")
                content = self._foam_header(cls, field, f"0/{r.name}") + f"dimensions      {dims};\ninternalField   uniform {value};\n" + "\n".join(bc) + "\n"
                self._write(f"0/{r.name}/{field}", content)
