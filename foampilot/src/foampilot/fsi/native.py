"""Native rigid-body FSI case generation for OpenFOAM.

This module intentionally generates dictionaries only. The actual coupling is
performed by OpenFOAM's sixDoFRigidBodyMotion: hydrodynamic forces are obtained
from the moving-body patch and fed back into the rigid-body equations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class FSIConfigurationError(ValueError):
    """Raised when an FSI configuration cannot produce a valid case."""


def _vector(values: Sequence[float], name: str) -> str:
    if len(values) != 3:
        raise FSIConfigurationError(f"{name} must contain exactly 3 values")
    return "(" + " ".join(f"{float(value):.16g}" for value in values) + ")"


def _tensor(values: Sequence[float], name: str) -> str:
    if len(values) != 6:
        raise FSIConfigurationError(f"{name} must contain exactly 6 values")
    return "(" + " ".join(f"{float(value):.16g}" for value in values) + ")"


def _words(values: Iterable[str]) -> str:
    result = tuple(values)
    if not result or any(not value or any(ch.isspace() for ch in value) for value in result):
        raise FSIConfigurationError("patch and constraint names must be non-empty words")
    return "(" + " ".join(result) + ")"


@dataclass(frozen=True)
class RigidBody:
    """Physical and kinematic data for one rigid body in the fluid mesh."""

    name: str = "body"
    patch: str = "body"
    cell_set: str | None = None
    mass: float = 1.0
    centre_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0)
    moment_of_inertia: tuple[float, float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not self.name or any(ch.isspace() for ch in self.name):
            raise FSIConfigurationError("body name must be a non-empty OpenFOAM word")
        if not self.patch or any(ch.isspace() for ch in self.patch):
            raise FSIConfigurationError("body patch must be a non-empty OpenFOAM word")
        if self.cell_set is not None and (not self.cell_set or any(ch.isspace() for ch in self.cell_set)):
            raise FSIConfigurationError("cell_set must be a non-empty OpenFOAM word")
        if self.mass <= 0:
            raise FSIConfigurationError("body mass must be positive")
        _vector(self.centre_of_mass, "centre_of_mass")
        _tensor(self.moment_of_inertia, "moment_of_inertia")
        _vector(self.initial_velocity, "initial_velocity")
        _vector(self.initial_angular_velocity, "initial_angular_velocity")


@dataclass
class NativeRigidFSI:
    """Generate the native OpenFOAM moving-mesh part of a rigid-body FSI case.

    ``variant='foundation13'`` writes the modern top-level ``mover`` form used
    by current OpenFOAM Foundation workflows. ``variant='legacy'`` writes the
    classic ``dynamicFvMesh``/``sixDoFRigidBodyMotionCoeffs`` form used by many
    OpenFOAM installations. The caller still supplies the fluid fields,
    boundary conditions, mesh, and solver-specific fvSchemes/fvSolution files.
    """

    case_path: str | Path
    body: RigidBody = field(default_factory=RigidBody)
    variant: str = "foundation13"
    solver: str = "incompressibleFluid"
    acceleration_relaxation: float = 0.4
    acceleration_damping: float = 0.9
    report: bool = True
    restraints: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    constraints: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.case_path = Path(self.case_path)
        if self.variant not in {"foundation13", "legacy"}:
            raise FSIConfigurationError("variant must be 'foundation13' or 'legacy'")
        if not self.solver:
            raise FSIConfigurationError("solver must be non-empty")
        if not 0 < self.acceleration_relaxation <= 1:
            raise FSIConfigurationError("acceleration_relaxation must be in (0, 1]")
        if not 0 <= self.acceleration_damping <= 1:
            raise FSIConfigurationError("acceleration_damping must be in [0, 1]")
        _words(self.constraints) if self.constraints else None

    def dynamic_mesh_dict(self) -> str:
        if self.variant == "legacy":
            return self._legacy_dynamic_mesh_dict()
        return self._foundation_dynamic_mesh_dict()

    def forces_function_object(self) -> str:
        """Return a controlDict function-object fragment for force feedback."""
        return f"""    {self.body.name}Forces
    {{
        type            forces;
        libs            ("libforces.so");
        patches         ({self.body.patch});
        rho             rhoInf;
        rhoInf          1;
        CofR            {_vector(self.body.centre_of_mass, 'centre_of_mass')};
        writeControl    timeStep;
        writeInterval   1;
    }}
"""

    def write(self) -> dict[str, Path]:
        """Write native FSI dictionaries and return their paths."""
        constant = self.case_path / "constant"
        system = self.case_path / "system"
        constant.mkdir(parents=True, exist_ok=True)
        system.mkdir(parents=True, exist_ok=True)
        dynamic_path = constant / "dynamicMeshDict"
        forces_path = system / f"{self.body.name}Forces.functionObject"
        dynamic_path.write_text(self.dynamic_mesh_dict(), encoding="utf-8")
        forces_path.write_text(self.forces_function_object(), encoding="utf-8")
        return {"dynamicMeshDict": dynamic_path, "forcesFunctionObject": forces_path}

    def _legacy_dynamic_mesh_dict(self) -> str:
        constraints = ""
        if self.constraints:
            constraints = "\n        constraints\n        {\n" + "".join(
                f"            {name}\n            {{ type {name}; }}\n" for name in self.constraints
            ) + "        }"
        restraints = self._restraints_block(indent="        ")
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      dynamicMeshDict;
}}

dynamicFvMesh       dynamicMotionSolverFvMesh;
motionSolverLibs    ("libfvMotionSolvers.so");
solver              sixDoFRigidBodyMotion;

sixDoFRigidBodyMotionCoeffs
{{
    patches             ({self.body.patch});
    innerDistance       0.3;
    outerDistance       1.0;
    centreOfMass        {_vector(self.body.centre_of_mass, 'centre_of_mass')};
    mass                {self.body.mass:.16g};
    momentOfInertia     {_tensor(self.body.moment_of_inertia, 'moment_of_inertia')};
    report              {"on" if self.report else "off"};
    solver              Newmark;
    accelerationRelaxation {self.acceleration_relaxation:.16g};
    accelerationDamping    {self.acceleration_damping:.16g};
    {constraints}
    {restraints}
}}
"""

    def _foundation_dynamic_mesh_dict(self) -> str:
        cell_set = f"\n                cellSet         {self.body.cell_set};" if self.body.cell_set else ""
        restraints = self._restraints_block(indent="                ")
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      dynamicMeshDict;
}}

mover
{{
    type            motionSolver;
    libs            ("librigidBodyMeshMotion.so");
    motionSolver    rigidBodyMotion;
    report          {"on" if self.report else "off"};
    solver
    {{
        type        Newmark;
    }}
    accelerationRelaxation {self.acceleration_relaxation:.16g};
    accelerationDamping    {self.acceleration_damping:.16g};
    nIter           3;
    bodies
    {{
        {self.body.name}
        {{
            type            rigidBody;
            parent          root;
            mass            {self.body.mass:.16g};
            centreOfMass    {_vector(self.body.centre_of_mass, 'centre_of_mass')};
            inertia         {_tensor(self.body.moment_of_inertia, 'moment_of_inertia')};
            transform       (1 0 0 0 1 0 0 0 1) {_vector(self.body.centre_of_mass, 'centre_of_mass')};
            joint           {{ type free; }}
            patches         ({self.body.patch});
            innerDistance   0.3;
            outerDistance   1.0;{cell_set}
        }}
    }}
    {restraints}
}}
"""

    def _restraints_block(self, indent: str) -> str:
        if not self.restraints:
            return ""
        lines = ["restraints", "{"]
        for name, data in self.restraints.items():
            if not name or any(ch.isspace() for ch in name):
                raise FSIConfigurationError("restraint names must be OpenFOAM words")
            lines.append(f"{indent}    {name}")
            lines.append(f"{indent}    {{")
            for key, value in data.items():
                if isinstance(value, (tuple, list)):
                    rendered = _vector(value, key)
                elif isinstance(value, bool):
                    rendered = "on" if value else "off"
                else:
                    rendered = str(value)
                lines.append(f"{indent}        {key} {rendered};")
            lines.append(f"{indent}    }}")
        lines.append(indent + "}")
        return "\n".join(lines)


def write_native_rigid_fsi(
    case_path: str | Path,
    *,
    body: RigidBody | None = None,
    variant: str = "foundation13",
    **kwargs: object,
) -> dict[str, Path]:
    """Convenience wrapper for writing a native rigid-body FSI configuration."""
    return NativeRigidFSI(
        case_path=case_path,
        body=body or RigidBody(),
        variant=variant,
        **kwargs,
    ).write()
