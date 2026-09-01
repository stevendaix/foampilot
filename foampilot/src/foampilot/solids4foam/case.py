"""Preparation of a standard solids4foam fluid-solid interaction case."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class Solids4FoamConfigurationError(ValueError):
    """Raised when a solids4foam case configuration is invalid."""


def _word(value: str, name: str) -> str:
    if not value or any(ch.isspace() for ch in value):
        raise Solids4FoamConfigurationError(f"{name} must be a non-empty OpenFOAM word")
    return value


def _vec(values: tuple[float, float, float], name: str) -> str:
    if len(values) != 3:
        raise Solids4FoamConfigurationError(f"{name} must contain 3 values")
    return "(" + " ".join(f"{float(v):.16g}" for v in values) + ")"


@dataclass(frozen=True)
class SolidMaterial:
    """Mechanical material for the solids4foam ``mechanicalProperties`` file."""

    name: str = "solidMaterial"
    law: str = "linearElastic"
    density: float = 1000.0
    young_modulus: float = 1.0e6
    poisson_ratio: float = 0.4
    plane_stress: bool = False
    alternate_pressure_definition: bool = False

    def __post_init__(self) -> None:
        _word(self.name, "material name")
        _word(self.law, "mechanical law")
        if self.density <= 0 or self.young_modulus <= 0:
            raise Solids4FoamConfigurationError("density and Young modulus must be positive")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise Solids4FoamConfigurationError("Poisson ratio must be in (-1, 0.5)")


@dataclass
class Solids4FoamCase:
    """Generate the configuration layer of a solids4foam FSI case.

    The class does not create a mesh or overwrite region fields. The expected
    case layout is ``0/fluid``, ``0/solid``, ``constant/fluid`` and
    ``constant/solid``. Mesh generation is intentionally left to the caller so
    Foampilot can continue to use its existing blockMesh/classy_blocks tools.
    """

    case_path: str | Path
    fluid_patch: str = "interface"
    solid_patch: str = "interface"
    material: SolidMaterial = SolidMaterial()
    coupling: str = "IQNILS"
    relaxation_factor: float = 0.1
    outer_corr_tolerance: float = 1.0e-6
    n_outer_corr: int = 50
    predictor: bool = False
    interface_transfer_method: str = "directMap"
    fluid_solver: str = "pimpleFluid"
    solid_model: str = "linearGeometryTotalDisplacement"
    solution_algorithm: str = "implicitSegregated"
    inlet_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fluid_properties: str | None = None

    def __post_init__(self) -> None:
        self.case_path = Path(self.case_path)
        _word(self.fluid_patch, "fluid_patch")
        _word(self.solid_patch, "solid_patch")
        _word(self.coupling, "coupling")
        _word(self.fluid_solver, "fluid_solver")
        _word(self.solid_model, "solid_model")
        _word(self.solution_algorithm, "solution_algorithm")
        if self.coupling not in {"fixedRelaxation", "Aitken", "IQNILS", "weakCoupling", "oneWayCoupling"}:
            raise Solids4FoamConfigurationError("unsupported solids4foam coupling")
        if not 0 < self.relaxation_factor <= 1:
            raise Solids4FoamConfigurationError("relaxation_factor must be in (0, 1]")
        if self.outer_corr_tolerance <= 0 or self.n_outer_corr < 1:
            raise Solids4FoamConfigurationError("FSI tolerance must be positive and n_outer_corr >= 1")
        _vec(self.inlet_velocity, "inlet_velocity")

    def physics_properties(self) -> str:
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      physicsProperties;
}

type fluidSolidInteraction;
"""

    def fsi_properties(self) -> str:
        predictor = "yes" if self.predictor else "no"
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fsiProperties;
}}

fluidSolidInterface    {self.coupling};
"{self.coupling}Coeffs"
{{
    solidPatch              {self.solid_patch};
    fluidPatch              {self.fluid_patch};
    predictor               {predictor};
    predictSolid            {predictor};
    relaxationFactor        {self.relaxation_factor:.16g};
    outerCorrTolerance      {self.outer_corr_tolerance:.16g};
    nOuterCorr              {self.n_outer_corr};
    coupled                 yes;
    interfaceTransferMethod {self.interface_transfer_method};
    writeResidualsToFile    yes;
}}
"""

    def solid_properties(self) -> str:
        predictor = "yes" if self.predictor else "no"
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      solidProperties;
}}

solidModel     {self.solid_model};
"{self.solid_model}Coeffs"
{{
    solutionAlgorithm {self.solution_algorithm};
    predictor         {predictor};
}}
"""

    def decompose_par_dict(self) -> str:
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}
numberOfSubdomains 2;
method          scotch;
"""

    def control_dict(self) -> str:
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     solids4Foam;
regionSolvers
{
    fluid           solids4Foam;
    solid           solids4Foam;
}
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          0.001;
writeControl    timeStep;
writeInterval   100;
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable yes;
"""

    def foundation13_physical_properties(self) -> str:
        m = self.material
        plane_stress = "yes" if m.plane_stress else "no"
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      physicalProperties;
}}

rho
{{
    type        uniform;
    value       {m.density:.16g};
}}
nu
{{
    type        uniform;
    value       {m.poisson_ratio:.16g};
}}
E
{{
    type        uniform;
    value       {m.young_modulus:.16g};
}}
Cv
{{
    type        uniform;
    value       0;
}}
kappa
{{
    type        uniform;
    value       0;
}}
alphav
{{
    type        uniform;
    value       0;
}}
planeStress     {plane_stress};
thermalStress   no;
"""

    def gravity(self) -> str:
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       uniformDimensionedVectorField;
    object      g;
}
dimensions      [0 1 -2 0 0 0 0];
value           (0 0 0);
"""

    def fluid_momentum_transport(self) -> str:
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      momentumTransport;
}
simulationType laminar;
laminar
{
    model Stokes;
}
"""

    def fluid_fv_schemes(self) -> str:
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
d2dt2Schemes { default backward; }
ddtSchemes { default backward; }
gradSchemes { default leastSquares; }
    divSchemes { default Gauss linear; }
laplacianSchemes
{
    default Gauss linear corrected;
    laplacian(interpolate(nuEff),U) Gauss linear corrected;
}
snGradSchemes { default corrected; }
interpolationSchemes { default linear; }
fluxRequired { default no; p; }
"""
    def solid_fv_schemes(self) -> str:
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
d2dt2Schemes { default Euler; }
ddtSchemes { default backward; }
gradSchemes { default leastSquares; }
divSchemes { default Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
snGradSchemes { default corrected; }
interpolationSchemes { default linear; }
"""

    def fluid_fv_solution(self) -> str:
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
solvers
{
    p
    {
        solver GAMG;
        tolerance 1e-8;
        relTol 0;
    }
    U
    {
        solver smoothSolver;
        smoother symGaussSeidel;
        tolerance 1e-8;
        relTol 0;
    }
}
PIMPLE
{
    momentumPredictor yes;
    nOuterCorrectors 1;
    nCorrectors 2;
    nNonOrthogonalCorrectors 0;
    pRefCell 0;
    pRefValue 0;
}
"""

    def solid_fv_solution(self) -> str:
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
solvers
{
    D
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-8;
        relTol          0.01;
    }
}
"""

    def mechanical_properties(self) -> str:
        m = self.material
        plane_stress = "yes" if m.plane_stress else "no"
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      mechanicalProperties;
}}

planeStress {plane_stress};
mechanical
(
    {m.name}
    {{
        type {m.law};
        rho rho [1 -3 0 0 0 0 0] {m.density:.16g};
        E   E   [1 -1 -2 0 0 0 0] {m.young_modulus:.16g};
        nu  nu  [0 0 0 0 0 0 0] {m.poisson_ratio:.16g};
{"        alternatePressureDefinition true;\n" if m.alternate_pressure_definition and m.law == "neoHookeanElastic" else ""}    }}
);
"""

    def functions(self) -> str:
        return f"""functions
{{
    fluidForces
    {{
        type                forces;
        region              fluid;
        libs                ("libforces.so");
        writeControl        timeStep;
        writeInterval       1;
        patches             ({self.fluid_patch});
        rho                 rhoInf;
        rhoInf              1000;
        CofR                (0 0 0);
        log                 true;
    }}
    solidDisplacement
    {{
        type                solidPointDisplacement;
        point               (0 0 0);
    }}
}}
"""

    def write(self) -> dict[str, Path]:
        """Write solids4foam dictionaries; return a mapping of generated paths."""
        constant = self.case_path / "constant"
        system = self.case_path / "system"
        (constant / "solid").mkdir(parents=True, exist_ok=True)
        (constant / "fluid").mkdir(parents=True, exist_ok=True)
        system.mkdir(parents=True, exist_ok=True)
        (system / "fluid").mkdir(parents=True, exist_ok=True)
        (system / "solid").mkdir(parents=True, exist_ok=True)
        files = {
            system / "controlDict": self.control_dict(),
            system / "decomposeParDict": self.decompose_par_dict(),
            constant / "physicsProperties": self.physics_properties(),
            constant / "fsiProperties": self.fsi_properties(),
            constant / "solid" / "solidProperties": self.solid_properties(),
            constant / "solid" / "mechanicalProperties": self.mechanical_properties(),
            constant / "solid" / "physicalProperties": self.foundation13_physical_properties(),
            constant / "solid" / "g": self.gravity(),
            constant / "fluid" / "momentumTransport": self.fluid_momentum_transport(),
            system / "fluid" / "fvSchemes": self.fluid_fv_schemes(),
            system / "fluid" / "fvSolution": self.fluid_fv_solution(),
            system / "solid" / "fvSchemes": self.solid_fv_schemes(),
            system / "solid" / "fvSolution": self.solid_fv_solution(),
            system / "functions": self.functions(),
        }
        fluid_properties = self.fluid_properties
        if fluid_properties is None:
            fluid_properties = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fluidProperties;
}

rho 1000;
nu 1e-6;
fluidModel pimpleFluid;
pimpleFluidCoeffs
{
    ddtCorr true;
}
"""
        files[constant / "fluid" / "fluidProperties"] = fluid_properties.rstrip() + "\n"
        for path, content in files.items():
            path.write_text(content, encoding="utf-8")
        return {str(path.relative_to(self.case_path)): path for path in files}

    def prepare_from_gmsh_entities(
        self,
        *,
        fluid_volumes: list[int] | tuple[int, ...],
        solid_volumes: list[int] | tuple[int, ...],
        interface_surfaces: list[int] | tuple[int, ...],
        fluid_volume: str = "FLUID",
        solid_volume: str = "SOLID",
        interface_surface: str = "interface",
    ) -> dict[str, object]:
        """Create physical groups from CAD tags, then export a two-region case."""
        from .gmsh_regions import create_fsi_physical_groups

        create_fsi_physical_groups(
            fluid_volumes=fluid_volumes,
            solid_volumes=solid_volumes,
            interface_surfaces=interface_surfaces,
            fluid_name=fluid_volume,
            solid_name=solid_volume,
            interface_name=interface_surface,
        )
        return self.prepare_from_gmsh(
            fluid_volume=fluid_volume,
            solid_volume=solid_volume,
            interface_surface=interface_surface,
        )

    def prepare_from_gmsh(
        self,
        *,
        region_map: Mapping[str, str] | None = None,
        fluid_volume: str = "FLUID",
        solid_volume: str = "SOLID",
        interface_surface: str | None = "interface",
    ) -> dict[str, object]:
        """Write configuration and export the active Gmsh model as two regions.

        The active Gmsh model must contain two named 3-D physical groups and,
        preferably, one shared 2-D physical group for the FSI interface. The
        existing :class:`DirectOpenFOAMExporter` writes each region to
        ``constant/<region>/polyMesh``; no external mesh converter is used.
        """
        try:
            import gmsh
        except ImportError as error:  # pragma: no cover - depends on installation
            raise Solids4FoamConfigurationError(
                "prepare_from_gmsh requires the optional gmsh dependency"
            ) from error

        region_map = dict(region_map or {fluid_volume: "fluid", solid_volume: "solid"})
        if fluid_volume not in region_map or solid_volume not in region_map:
            raise Solids4FoamConfigurationError(
                "region_map must define both fluid and solid physical volumes"
            )
        physical_volumes = {
            gmsh.model.getPhysicalName(3, tag)
            for _, tag in gmsh.model.getPhysicalGroups(3)
        }
        missing = {fluid_volume, solid_volume} - physical_volumes
        if missing:
            raise Solids4FoamConfigurationError(
                "missing Gmsh 3-D physical volume(s): " + ", ".join(sorted(missing))
            )
        if interface_surface is not None:
            physical_surfaces = {
                gmsh.model.getPhysicalName(2, tag)
                for _, tag in gmsh.model.getPhysicalGroups(2)
            }
            if interface_surface not in physical_surfaces:
                raise Solids4FoamConfigurationError(
                    f"missing Gmsh 2-D FSI interface physical group: {interface_surface}"
                )

        from foampilot.mesh.direct_openfoam_exporter import DirectOpenFOAMExporter

        generated = self.write()
        mesh_paths = DirectOpenFOAMExporter(self.case_path).export_multi_region(
            region_map=region_map
        )
        generated["regionMeshes"] = mesh_paths
        if interface_surface is not None:
            expected_patch = interface_surface
            missing_regions = []
            for region_name in (region_map[fluid_volume], region_map[solid_volume]):
                boundary_file = (
                    self.case_path / "constant" / region_name / "polyMesh" / "boundary"
                )
                if not boundary_file.is_file() or not self._boundary_has_patch(
                    boundary_file, expected_patch
                ):
                    missing_regions.append(region_name)
            if missing_regions:
                raise Solids4FoamConfigurationError(
                    "FSI interface patch is missing from region(s): "
                    + ", ".join(missing_regions)
                    + ". Assign the shared Gmsh interface surface to both sides "
                    "of the fluid-solid interface."
                )
        return generated

    @staticmethod
    def _boundary_has_patch(path: Path, patch_name: str) -> bool:
        """Check a generated OpenFOAM boundary file without a parser dependency."""
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped == patch_name or stripped.startswith(patch_name + " "):
                return True
        return False

    def run_plan(self, parallel: bool = False) -> list[list[str]]:
        """Return commands in solids4foam's standard serial/parallel workflow."""
        commands: list[list[str]] = [
            ["solids4Foam", "-region", "solid", "blockMesh"],
            ["solids4Foam", "-region", "fluid", "blockMesh"],
        ]
        if parallel:
            commands.extend([
                ["decomposePar", "-allRegions"],
                [
                    "sh", "-c",
                    "for p in processor*; do "
                    "mkdir -p \"$p/constant/fluid\" \"$p/constant/solid\" "
                    "\"$p/system/fluid\" \"$p/system/solid\"; "
                    "cp constant/physicsProperties constant/fsiProperties \"$p/constant/\"; "
                    "cp constant/fluid/fluidProperties constant/fluid/momentumTransport \"$p/constant/fluid/\"; "
                    "cp constant/solid/solidProperties constant/solid/mechanicalProperties \"$p/constant/solid/\"; "
                    "cp system/controlDict system/functions \"$p/system/\"; "
                    "cp system/fluid/fvSchemes system/fluid/fvSolution \"$p/system/fluid/\"; "
                    "cp system/solid/fvSchemes system/solid/fvSolution \"$p/system/solid/\"; "
                    "done",
                ],
                [
                    "mpirun", "--allow-run-as-root", "--oversubscribe",
                    "-np", "2", "solids4Foam", "-parallel",
                ],
                ["reconstructPar", "-allRegions"],
            ])
        else:
            commands.append(["solids4Foam"])
        return commands


def write_solids4foam_case(case_path: str | Path, **kwargs: object) -> dict[str, Path]:
    """Convenience wrapper around :class:`Solids4FoamCase`."""
    return Solids4FoamCase(case_path=case_path, **kwargs).write()
