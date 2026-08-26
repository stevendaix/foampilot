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
    predictor: bool = True
    interface_transfer_method: str = "directMap"
    fluid_solver: str = "pimpleFluid"
    solid_model: str = "nonLinearGeometryTotalLagrangianTotalDisplacement"
    solution_algorithm: str = "PETScSNES"
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

    def mechanical_properties(self) -> str:
        m = self.material
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      mechanicalProperties;
}}

planeStress {{'yes' if m.plane_stress else 'no'}};
mechanical
(
    {m.name}
    {{
        type {m.law};
        rho rho [1 -3 0 0 0 0 0] {m.density:.16g};
        E   E   [1 -1 -2 0 0 0 0] {m.young_modulus:.16g};
        nu  nu  [0 0 0 0 0 0 0] {m.poisson_ratio:.16g};
    }}
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
        files = {
            constant / "physicsProperties": self.physics_properties(),
            constant / "fsiProperties": self.fsi_properties(),
            constant / "solid" / "solidProperties": self.solid_properties(),
            constant / "solid" / "mechanicalProperties": self.mechanical_properties(),
            system / "functions": self.functions(),
        }
        if self.fluid_properties is not None:
            files[constant / "fluid" / "fluidProperties"] = self.fluid_properties.rstrip() + "\n"
        for path, content in files.items():
            path.write_text(content, encoding="utf-8")
        return {str(path.relative_to(self.case_path)): path for path in files}

    def run_plan(self, parallel: bool = False) -> list[list[str]]:
        """Return commands in solids4foam's standard serial/parallel workflow."""
        commands: list[list[str]] = [
            ["solids4Foam", "-region", "solid", "blockMesh"],
            ["solids4Foam", "-region", "fluid", "blockMesh"],
        ]
        if parallel:
            commands.extend([
                ["decomposePar", "-region", "fluid"],
                ["decomposePar", "-region", "solid"],
                ["solids4Foam", "-parallel"],
                ["reconstructPar", "-region", "fluid"],
                ["reconstructPar", "-region", "solid"],
            ])
        else:
            commands.append(["solids4Foam"])
        return commands


def write_solids4foam_case(case_path: str | Path, **kwargs: object) -> dict[str, Path]:
    """Convenience wrapper around :class:`Solids4FoamCase`."""
    return Solids4FoamCase(case_path=case_path, **kwargs).write()
