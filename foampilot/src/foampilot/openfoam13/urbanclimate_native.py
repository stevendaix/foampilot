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
            "stopAt          endTime;\nendTime         0.02;\ndeltaT          0.01;\n"
            "writeControl    timeStep;\nwriteInterval   1;\n"
            "maxFluidIteration 20;\nminFluidIteration 2;\n"
            "libs            ();\n" + regions
        )
        self._write("system/controlDict", content)

    def _write_allrun(self) -> None:
        regions = " ".join(r.name for r in self.regions)
        solid_regions = " ".join(r.name for r in self.regions if r.kind == "solid")
        topo = ""
        if solid_regions:
            topo = "python3 make_cell_zones.py\n"
        radiation_steps = ""
        if self.vegetation:
            radiation_steps = "faceAgglomerate -region vegetation\ncalcLAI -region air\nviewFactorsGen -region vegetation\nsolarRayTracingGen -region vegetation\n"
        self._write("Allrun", f'''#!/bin/sh
set -eu
cd "${{0%/*}}" || exit 1
for region in {regions}; do
    blockMesh -region "$region"
done
{topo}{radiation_steps}exec urbanMicroclimateFoam
''')
        if solid_regions:
            region_names = repr(tuple(r.name for r in self.regions if r.kind == "solid"))
            radiation_names = repr(tuple(r.name for r in self.regions if r.kind in ("fluid", "vegetation") and self.vegetation))
            script = '''from pathlib import Path\n\nN_CELLS = 14*20*12\nregions = __REGIONS__\nradiation_regions = __RADIATION_REGIONS__\nPATCH_FACE_COUNTS = (240, 240, 168, 168, 280, 280)\nPATCH_GROUP_SIZES = (1, 1, 1, 1, 1, 1)\nheader = chr(10).join(["FoamFile", "{", "    version 2.0;", "    format ascii;", "    class cellZoneList;", "    location \\"constant/{region}/polyMesh\\";", "    object cellZones;", "}", ""])\nlabels = chr(10).join(f"        {i}" for i in range(N_CELLS))\nfor region in regions:\n    path = Path("constant") / region / "polyMesh" / "cellZones"\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(header.replace("{region}", region) + f"1\\n(\\nallCells\\n{{\\n    cellLabels {N_CELLS}\\n    (\\n{labels}\\n    );\\n}}\\n)\\n")\nfor region in radiation_regions:\n    path = Path("constant") / region / "finalAgglom"\n    lists = []\n    for count, group_size in zip(PATCH_FACE_COUNTS, PATCH_GROUP_SIZES):\n        values = chr(10).join(f"        {i // group_size}" for i in range(count))\n        lists.append(f"{count}\\n(\\n{values}\\n)\\n")\n    final_header = chr(10).join(["FoamFile", "{", "    version 2.0;", "    format ascii;", "    class labelListList;", "    location \\"constant/{region}\\";", "    object finalAgglom;", "}", ""])
    path.write_text(final_header.replace("{region}", region) + "6\\n(\\n" + "".join(lists) + ")\\n")
    boundary = Path("constant") / region / "polyMesh" / "boundary"
    text = boundary.read_text()
    for patch, neighbour in (("inlet", "inlet"), ("outlet", "outlet"), ("air_to_vegetation", "side1"), ("side2", "side2"), ("top", "top")):
        old = f"{patch}\\n    {{\\n        type            patch;"
        new = f"{patch}\\n    {{\\n        type            mapped;\\n        neighbourRegion air;\\n        neighbourPatch {neighbour};\\n        offsetMode uniform;\\n        offset (0 0 0);"
        text = text.replace(old, new)
    old = "ground\\n    {\\n        type            wall;"
    new = "ground\\n    {\\n        type            mappedWall;\\n        neighbourRegion air;\\n        neighbourPatch ground;\\n        offsetMode uniform;\\n        offset (0 0 0);"
    text = text.replace(old, new)
    boundary.write_text(text)
'''.replace("__REGIONS__", region_names).replace("__RADIATION_REGIONS__", radiation_names)
            self._write("make_cell_zones.py", script)
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
        mesh.blocks = ["hex (0 1 2 3 4 5 6 7) (14 20 12) simpleGrading (1 1 1)"]
        mesh.edges = []
        mesh.defaultPatch = {"defaultFaces": "wall"}
        side_type = "patch" if self.vegetation else "symmetryPlane"
        mesh.boundary = {
            "inlet": {"type": "patch", "faces": [[0, 4, 7, 3]]},
            "outlet": {"type": "patch", "faces": [[1, 2, 6, 5]]},
            "side1": {"type": side_type, "faces": [[0, 1, 5, 4]]},
            "side2": {"type": side_type, "faces": [[3, 7, 6, 2]]},
            "top": {"type": "patch", "faces": [[4, 5, 6, 7]]},
            "ground": {"type": "wall", "faces": [[0, 3, 2, 1]]},
        }
        mesh.mergePatchPairs = []
        for region in self.regions:
            region_system = self.case_path / "system" / region.name
            region_system.mkdir(parents=True, exist_ok=True)
            original_boundary = mesh.boundary
            if region.kind == "vegetation":
                mesh.boundary = dict(original_boundary)
                mesh.boundary["air_to_vegetation"] = mesh.boundary.pop("side1")
            mesh.write(region_system / "blockMeshDict")
            # Keep the standard root mesh for the fluid region as a portable
            # preflight/inspection entry point; region-scoped dictionaries remain
            # authoritative for multi-region execution.
            if region.kind == "fluid":
                mesh.write(self.case_path / "system" / "blockMeshDict")
            mesh.boundary = original_boundary
        if self.vegetation:
            mapped = self._foam_header("dictionary", "changeDictionaryDict", "system/vegetation") + "boundary\n{\n    air_to_vegetation\n    {\n        type mapped;\n        neighbourRegion air;\n        neighbourPatch side1;\n        offsetMode uniform;\n        offset (0 0 0);\n    }\n}\n"
            self._write("system/vegetation/changeDictionaryDict", mapped)
        self._write("system/decomposeParDict", self._foam_header("dictionary", "decomposeParDict", "system") + "numberOfSubdomains 1;\nmethod          scotch;\n")
        for region in self.regions:
            base = f"system/{region.name}"
            schemes = self._foam_header("dictionary", "fvSchemes", base) + "ddtSchemes { default Euler; }\ngradSchemes { default Gauss linear; }\ndivSchemes\n{\n    default none;\n    div(phi,U) Gauss linearUpwind grad(U);\n    div(phi,he) Gauss linearUpwind grad(he);\n    div(phi,h) Gauss linearUpwind grad(h);\n    div(phi,w) Gauss upwind;\n    div(phi,K) Gauss linear;\n    div(phi,k) Gauss upwind;\n    div(phi,epsilon) Gauss upwind;\n    div(phi,omega) Gauss upwind;\n    div(alphaRhoPhi,k) Gauss upwind;\n    div(alphaRhoPhi,epsilon) Gauss upwind;\n    div(alphaRhoPhi,omega) Gauss upwind;\n}\nlaplacianSchemes { default Gauss linear corrected; laplacian(Krel,pc) Gauss linear corrected; laplacian(lambda_m,Ts) Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n"
            self._write(f"{base}/fvSchemes", schemes)
            solution = self._foam_header("dictionary", "fvSolution", base) + "solvers\n{\n    p_rgh { solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.05; cacheAgglomeration true; agglomerator faceAreaPair; mergeLevels 1; }\n    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.05; }\n    T { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.05; }\n    Ts { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.05; }\n    pc { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.05; }\n    h { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.05; }\n    he { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.05; }\n    k { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.05; }\n    epsilon { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.05; }\n    omega { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.05; }\n    nut { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.05; }\n    alphat { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.05; }\n    w { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.05; }\n    gcr { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.05; }\n}\nSIMPLE\n{\n    residualControl {}\n    nNonOrthogonalCorrectors 0;\n    momentumPredictor yes;\n    consistent yes;\n    pRefCell 0;\n    pRefValue 100000;\n}\nPIMPLE\n{\n    nOuterCorrectors 1;\n    nCorrectors 1;\n    nNonOrthogonalCorrectors 0;\n}\nrelaxationFactors\n{\n    fields { p_rgh 0.3; }\n    equations { U 0.3; h 0.1; T 0.1; k 0.2; epsilon 0.2; w 0.2; }\n}\n"
            self._write(f"{base}/fvSolution", solution)
            if self.vegetation and region.kind in ("fluid", "vegetation"):
                view_body = "skyPosVector (0 0 1);\nwriteFacesAgglomeration true;\ndebug 0;\ndumpRays false;\nmaxDynListLength 10000000;\n" + "\n".join(f"{patch}\n{{\n    nFacesInCoarsestLevel 20;\n    featureAngle 10;\n}}" for patch in ("inlet", "outlet", "air_to_vegetation", "side2", "top", "ground")) + "\n"
                view_dict = self._foam_header("dictionary", "viewFactorsDict", base) + view_body
                self._write(f"{base}/viewFactorsDict", view_dict)
                self._write("system/viewFactorsDict", self._foam_header("dictionary", "viewFactorsDict", "system") + view_body)
            if region.kind == "solid":
                topo = self._foam_header("dictionary", "topoSetDict", base) + "actions\n(\n    { name allCells; type cellSet; action new; source boxToCell; sourceInfo { box (-1 -1 -1) (71 101 61); } }\n    { name allCells; type cellZoneSet; action new; source setToCellZone; sourceInfo { set allCells; } }\n);\n"
                self._write(f"{base}/topoSetDict", topo)

    def _write_constant_dictionaries(self) -> None:
        for region in self.regions:
            base = f"constant/{region.name}"
            gravity = "(0 0 0)" if self.profile == "streetCanyon_CFD" else "(0 0 -9.81)"
            self._write(f"{base}/g", self._foam_header("uniformDimensionedVectorField", "g", base) + f"dimensions [0 1 -2 0 0 0 0];\nvalue {gravity};\n")
            if region.kind == "fluid":
                self._write(f"{base}/transportProperties", self._foam_header("dictionary", "transportProperties", base) + "transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] 1e-05;\n")
                if self.vegetation:
                    vegetation = self._foam_header("dictionary", "vegetationProperties", base) + "vegetationModel simplifiedVegetation;\nsimplifiedVegetationCoeffs\n{\n    a1 [1 0 -3 0 0 0 0] 169;\n    a2 [1 0 -3 0 0 0 0] 18;\n    a3 [1 -1 -2 0 0 0 0] 0.005;\n    D0 [1 -1 -2 0 0 0 0] 1.2;\n    C [0 -1 0.5 0 0 0 0] 131.035;\n    betaP [0 0 0 0 0 0 0] 1.0;\n    betaD [0 0 0 0 0 0 0] 5.1;\n    H [0 1 0 0 0 0 0] 1.5;\n    kc [0 0 0 0 0 0 0] 0.5;\n    l [0 1 0 0 0 0 0] 0.1;\n    rsMin [0 -1 1 0 0 0 0] 150;\n    Rg0 [1 0 -3 0 0 0 0] 800;\n    Rl0 [1 0 -3 0 0 0 0] 350;\n    nEvapSides [0 0 0 0 0 0 0] 1;\n}\n"
                    self._write(f"{base}/vegetationProperties", vegetation)
                thermo = self._foam_header("dictionary", "thermophysicalProperties", base) + "thermoType\n{\n    type heRhoThermo; mixture pureMixture; transport const; thermo hConst; equationOfState incompressiblePerfectGas; specie specie; energy sensibleEnthalpy;\n}\nmixture\n{\n    specie { molWeight 28.9; }\n    thermodynamics { Cp 1000; Hf 0; }\n    transport { mu 1.8e-05; Pr 0.7; }\n    equationOfState { pRef 1e5; }\n}\n"
                self._write(f"{base}/thermophysicalProperties", thermo)
                turbulence = "simulationType laminar;\n" if self.profile == "streetCanyon_CFD" else "simulationType RAS;\nRAS { RASModel realizableKE; turbulence on; printCoeffs on; }\n"
                self._write(f"{base}/momentumTransport", self._foam_header("dictionary", "momentumTransport", base) + turbulence)
                if self.vegetation:
                    air_radiation = self._foam_header("dictionary", "radiationProperties", base) + "radiationModel opaqueSolid;\nabsorptionEmissionModel constant;\nconstantCoeffs\n{\n    absorptivity 0;\n    emissivity 0.9;\n    E 0;\n}\nscatterModel none;\nsootModel none;\n"
                    self._write(f"{base}/radiationProperties", air_radiation)
            elif region.kind == "solid":
                material_model = "Soil" if region.name == "ground" else "HamstadConcrete"
                solid_transport = self._foam_header("dictionary", "transportProperties", base) + f"rho [1 -3 0 0 0 0 0] 1800;\nCp [0 2 -2 -1 0 0 0] 900;\nkappa [1 1 -3 -1 0 0 0] 1.4;\nbuildingMaterials\n(\n    {{ name allCells; buildingMaterialModel {material_model}; rho 1800; cap 900; lambda1 1.4; lambda2 0.0; }}\n);\n"
                self._write(f"{base}/transportProperties", solid_transport)
                self._write(f"{base}/buildingMaterials", self._foam_header("dictionary", "buildingMaterials", base) + f"materials (\n    {{ name allCells; buildingMaterialModel {material_model}; rho 1800; cap 900; lambda1 1.4; lambda2 0.0; }}\n);\n")
            elif region.kind == "vegetation":
                vegetation = self._foam_header("dictionary", "vegetationProperties", base) + "vegetation true;\nsolverFreq 1;\nvegetationModel simplifiedVegetation;\nsimplifiedVegetationCoeffs\n{\n    a1 [1 0 -3 0 0 0 0] 169;\n    a2 [1 0 -3 0 0 0 0] 18;\n    a3 [1 -1 -2 0 0 0 0] 0.005;\n    D0 [1 -1 -2 0 0 0 0] 1.2;\n    C [0 -1 0.5 0 0 0 0] 131.035;\n    betaP [0 0 0 0 0 0 0] 1.0;\n    betaD [0 0 0 0 0 0 0] 5.1;\n    H [0 1 0 0 0 0 0] 1.5;\n    kc [0 0 0 0 0 0 0] 0.5;\n    l [0 1 0 0 0 0 0] 0.1;\n    rsMin [0 -1 1 0 0 0 0] 150;\n    Rg0 [1 0 -3 0 0 0 0] 800;\n    Rl0 [1 0 -3 0 0 0 0] 350;\n    nEvapSides [0 0 0 0 0 0 0] 1;\n}\n"
                self._write(f"{base}/vegetationProperties", vegetation)
                self._write(f"{base}/radiationProperties", self._foam_header("dictionary", "radiationProperties", base) + "radiationModel viewFactorSky;\nviewFactorSkyCoeffs { smoothing true; constantEmissivity true; }\nsolverFreq 1;\nabsorptionEmissionModel none;\nscatterModel none;\n")
                self._write(f"{base}/solarLoadProperties", self._foam_header("dictionary", "solarLoadProperties", base) + "solarLoadModel directAndDiffuse;\ndirectAndDiffuseCoeffs { smoothing false; constantAlbedo false; }\nsolverFreq 1;\nsolarLoadAbsorptionEmissionModel none;\nsolarLoadScatterModel none;\n")
        if self.radiation:
            solar_tables = {
                "sunPosVector": "(\n(0 (1 1 1))\n(3600 (1 1 1))\n)\n",
                "IDN": "(\n(0 800)\n(3600 800)\n)\n",
                "Idif": "(\n(0 100)\n(3600 100)\n)\n",
            }
            # In a region case, mesh.time().constant() resolves to
            # constant/<region>, not the root constant directory.
            for region in self.regions:
                if region.kind in ("fluid", "vegetation"):
                    for name, content in solar_tables.items():
                        self._write(f"constant/{region.name}/{name}", content)
            # Keep root copies for global inspection and compatibility scripts.
            for name, content in solar_tables.items():
                self._write(f"constant/{name}", content)

    def _write_initial_fields(self) -> None:
        region_objects: list[Any] = []
        for r in self.regions:
            region_objects.append(FluidRegion(r.name, temperature=r.temperature, velocity=r.velocity) if r.kind == "fluid" else SolidRegion(r.name, temperature=r.temperature))
        manager = CaseFieldsManager(energy_activated=True, with_gravity=True, with_radiation=self.radiation, regions=region_objects)
        for r, obj in zip(self.regions, region_objects):
            fields = list(manager.get_region_field_names(r.name))
            if r.kind == "fluid":
                fields.extend(("p", "w", "gcr", "epsilon", "alphat", "h"))
                if self.vegetation:
                    # calcLAI reads LAD from the air region in the native
                    # urbanMicroclimate vegetation pipeline.
                    fields.append("LAD")
            elif r.kind == "solid":
                fields.extend(("Ts", "ws", "pc"))
            if self.radiation and r.kind in ("fluid", "vegetation"):
                fields.append("qr")
                if r.kind == "vegetation":
                    fields.append("qs")
            fields = list(dict.fromkeys(fields))
            for field in fields:
                cls = "volVectorField" if field == "U" else "volScalarField"
                dims = {"U": "[0 1 -1 0 0 0 0]", "p": "[1 -1 -2 0 0 0 0]", "p_rgh": "[1 -1 -2 0 0 0 0]", "T": "[0 0 0 1 0 0 0]", "Ts": "[0 0 0 1 0 0 0]", "h": "[0 2 -2 0 0 0 0]", "Ts": "[0 0 0 1 0 0 0]", "k": "[0 2 -2 0 0 0 0]", "epsilon": "[0 2 -3 0 0 0 0]", "nut": "[0 2 -1 0 0 0 0]", "alphat": "[1 -1 -1 0 0 0 0]", "qr": "[1 -1 -3 0 0 0 0]", "qs": "[1 -1 -3 0 0 0 0]", "w": "[0 0 0 0 0 0 0]", "gcr": "[0 0 0 0 0 0 0]", "pc": "[1 -1 -2 0 0 0 0]", "ws": "[1 -3 0 0 0 0 0]", "LAD": "[0 -1 0 0 0 0 0]", "pc": "[1 -1 -2 0 0 0 0]"}.get(field, "[0 0 0 0 0 0 0]")
                values = {"p": "100000", "p_rgh": "100000", "pc": "-100000", "h": str(r.temperature*1000.0), "k": "0.1", "epsilon": "0.01", "omega": "1", "nut": "1e-05", "alphat": "0.001", "LAD": "1.0"}
                value = (f"({r.velocity[0]} {r.velocity[1]} {r.velocity[2]})" if field == "U" else str(r.temperature if field in ("T", "Ts") else values.get(field, "0")))
                side_bc = ("zeroGradient", None) if self.vegetation else ("symmetryPlane", None)
                if field == "qs":
                    boundary = {
                        patch: ("solarLoadRadiationViewFactor", "qso uniform 0")
                        for patch in ("inlet", "outlet", "air_to_vegetation", "side2", "top", "ground")
                    }
                elif field == "qr":
                    boundary = {
                        patch: ("greyDiffusiveRadiationViewFactor", "value uniform 0")
                        for patch in ("inlet", "outlet", "side1", "side2", "top", "ground")
                    }
                elif field == "U":
                    boundary = {
                        "inlet": ("fixedValue", f"value uniform ({r.velocity[0]} {r.velocity[1]} {r.velocity[2]})"),
                        "outlet": ("zeroGradient", None),
                        "side1": side_bc,
                        "side2": side_bc,
                        "top": ("zeroGradient", None),
                        "ground": ("noSlip", None),
                    }
                elif field == "p":
                    boundary = {
                        "inlet": ("calculated", "value uniform 100000"),
                        "outlet": ("calculated", "value uniform 100000"),
                        "side1": side_bc,
                        "side2": side_bc,
                        "top": ("slip", None),
                        "ground": ("calculated", "value uniform 100000"),
                    }
                elif field == "p_rgh":
                    boundary = {
                        "inlet": ("fixedFluxPressure", "value uniform 100000"),
                        "outlet": ("fixedValue", "value uniform 100000"),
                        "side1": side_bc,
                        "side2": side_bc,
                        "top": ("slip", None),
                        "ground": ("fixedFluxPressure", "value uniform 100000"),
                    }
                elif field in ("T", "Ts", "h"):
                    thermal_value = str(r.temperature) if field in ("T", "Ts") else values["h"]
                    boundary = {
                        "inlet": ("fixedValue", f"value uniform {thermal_value}"),
                        "outlet": ("fixedValue", f"value uniform {thermal_value}"),
                        "side1": side_bc,
                        "side2": side_bc,
                        "top": ("fixedValue", f"value uniform {thermal_value}"),
                        "ground": ("fixedValue", f"value uniform {thermal_value}"),
                    }
                else:
                    boundary = {
                        "inlet": ("zeroGradient", None),
                        "outlet": ("zeroGradient", None),
                        "side1": side_bc,
                        "side2": side_bc,
                        "top": ("zeroGradient", None),
                        "ground": ("zeroGradient", None),
                    }
                bc = ["boundaryField", "{"]
                for patch, (kind, extra) in boundary.items():
                    bc.append(f"    {patch}")
                    bc.append("    {")
                    bc.append(f"        type {kind};")
                    if field == "qr":
                        bc.append("        emissivityMode lookup;")
                        bc.append("        emissivity uniform 0.9;")
                        bc.append("        value uniform 0;")
                        bc.append("        qro uniform 0;")
                    elif extra:
                        key, val = extra.split(" ", 1)
                        bc.append(f"        {key} {val};")
                        if field == "qs":
                            bc.append("        albedoMode lookup;")
                            bc.append("        albedo uniform 0.1;")
                            bc.append("        value uniform 0;")
                    bc.append("    }")
                bc.append("}")
                if r.kind == "vegetation":
                    for line_no, line in enumerate(bc):
                        if line == "    side1":
                            bc[line_no] = "    air_to_vegetation"
                content = self._foam_header(cls, field, f"0/{r.name}") + f"dimensions      {dims};\ninternalField   uniform {value};\n" + "\n".join(bc) + "\n"
                self._write(f"0/{r.name}/{field}", content)
        if self.ham:
            # HAM wall transfer conditions read these Function1 tables from
            # the global case path 0/air in the OpenFOAM-13 port.
            self._write("0/air/Tambient", "(\n(0 285)\n(3600 285)\n)\n")
            self._write("0/air/wambient", "(\n(0 0.0076)\n(3600 0.0076)\n)\n")
