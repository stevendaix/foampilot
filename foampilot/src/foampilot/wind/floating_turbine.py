"""Declarative OpenFOAM 13 configuration for floating actuator-line turbines.

The physical models originate from the educational thesis-FloatingTurbine
repository.  This module deliberately generates dictionaries instead of
copying old case trees, so the configuration remains reproducible and can be
checked before a solver is launched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping


def _vec(value: Iterable[float]) -> str:
    values = tuple(float(v) for v in value)
    if len(values) != 3:
        raise ValueError("A vector must contain exactly three values")
    return "(" + " ".join(f"{v:g}" for v in values) + ")"


def _scalar(value: float) -> str:
    return f"{float(value):g}"


@dataclass(frozen=True)
class MooringLine:
    """Quasi-steady catenary line used by the source ``mooringLine`` restraint."""

    name: str
    anchor: tuple[float, float, float]
    attachment_point: tuple[float, float, float]
    mass_per_length: float
    line_length: float
    thickness: float = 0.0766

    def __post_init__(self) -> None:
        if not self.name or any(ch.isspace() for ch in self.name):
            raise ValueError("Mooring line name must be a non-empty token")
        if self.mass_per_length <= 0 or self.line_length <= 0 or self.thickness <= 0:
            raise ValueError("Mooring line physical dimensions must be positive")

    def render(self, gravity: tuple[float, float, float]) -> str:
        return f"""        {self.name}
        {{
            sixDoFRigidBodyMotionRestraint mooringLine;
            anchor              {_vec(self.anchor)};
            refAttachmentPt     {_vec(self.attachment_point)};
            massPerLength       {_scalar(self.mass_per_length)};
            lineLength          {_scalar(self.line_length)};
            gravityVector       {_vec(gravity)};
            thickness           {_scalar(self.thickness)};
        }}
"""


@dataclass(frozen=True)
class FloatingTurbine:
    """Physical inputs shared by the actuator-line and rigid-body models."""

    name: str = "floatingTurbine"
    position: tuple[float, float, float] = (0.0, 0.0, 90.0)
    rotor_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    free_stream_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    rotor_diameter: float = 178.0
    n_blades: int = 3
    blade_pitch: float = 0.0
    airfoil_data: str = "constant/airfoilData"
    turbine_library: str = "libfloatingTurbinesFoam.so"
    rigid_body_library: str = "libfloatingSixDoFRigidBodyMotion.so"
    coupling: bool = True
    mooring_lines: tuple[MooringLine, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name or any(ch.isspace() for ch in self.name):
            raise ValueError("Turbine name must be a non-empty OpenFOAM token")
        if self.rotor_diameter <= 0 or self.n_blades < 1:
            raise ValueError("Rotor diameter and blade count must be positive")
        if not self.airfoil_data:
            raise ValueError("airfoil_data must not be empty")

    def validate(self) -> list[str]:
        """Return actionable configuration errors without running OpenFOAM."""
        errors: list[str] = []
        if abs(sum(float(v) ** 2 for v in self.rotor_axis) - 1.0) > 1e-6:
            errors.append("rotor_axis must be a unit vector")
        if abs(sum(float(v) ** 2 for v in self.free_stream_direction) - 1.0) > 1e-6:
            errors.append("free_stream_direction must be a unit vector")
        if self.coupling and not self.mooring_lines:
            # A turbine can be coupled without moorings (e.g. a prescribed
            # motion test), therefore this is a warning-like validation only.
            pass
        return errors

    def runtime_libraries(self) -> tuple[str, ...]:
        libraries = [self.turbine_library]
        if self.mooring_lines:
            libraries.append(self.rigid_body_library)
        return tuple(libraries)

    def render_source_dictionary(self, *, cell_zone: str = "rotor", container: str = "fvOptions") -> str:
        """Render the actuator-line source in ``fvOptions`` or ``fvModels``.

        The thesis library is registered against the legacy ``fvOptions`` API.
        A port of that library to OpenFOAM 13 can request ``fvModels``
        explicitly; silently selecting the newer container would otherwise
        produce a case that cannot load the historical shared library.
        """
        if container not in {"fvOptions", "fvModels"}:
            raise ValueError("container must be 'fvOptions' or 'fvModels'")
        couple = "true" if self.coupling else "false"
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      {container};
}}

{self.name}
{{
    type                axialFlowTurbineALSource;
    active              true;
    selectionMode       cellZone;
    cellZone            {cell_zone};
    rotorOrigin         {_vec(self.position)};
    rotorAxis           {_vec(self.rotor_axis)};
    freeStreamDirection {_vec(self.free_stream_direction)};
    rotorDiameter       {_scalar(self.rotor_diameter)};
    nBlades             {self.n_blades};
    bladePitch          {_scalar(self.blade_pitch)};
    airfoilData         {self.airfoil_data};
    coupleLoads         {couple};
}}
"""

    def render_fv_models(self, *, cell_zone: str = "rotor") -> str:
        """Render the OpenFOAM 13 ``fvModels`` variant."""
        return self.render_source_dictionary(cell_zone=cell_zone, container="fvModels")

    def render_legacy_fv_options(self, *, cell_zone: str = "rotor") -> str:
        """Render the legacy container used by the v2012 source library."""
        return self.render_source_dictionary(cell_zone=cell_zone, container="fvOptions")

    def render_dynamic_mesh(
        self,
        *,
        body_patch: str = "floater",
        center_of_mass: tuple[float, float, float] | None = None,
        mass: float = 1.0,
        moment_of_inertia: tuple[float, float, float] = (1.0, 1.0, 1.0),
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.8065),
        constraints: Mapping[str, Mapping[str, object]] | None = None,
    ) -> str:
        """Render a sixDoF dynamicMeshDict compatible with OpenFOAM 13."""
        if mass <= 0 or any(v <= 0 for v in moment_of_inertia):
            raise ValueError("Rigid-body mass and inertia must be positive")
        com = center_of_mass or self.position
        lines = "\n".join(line.render(gravity) for line in self.mooring_lines)
        constraint_text = ""
        if constraints:
            rendered = []
            for name, data in constraints.items():
                kind = str(data["type"])
                if kind == "axis":
                    rendered.append(
                        f"        {name}\n        {{\n            sixDoFRigidBodyMotionConstraint axis;\n            axis {_vec(data['axis'])};\n        }}"
                    )
                elif kind == "line":
                    rendered.append(
                        f"        {name}\n        {{\n            sixDoFRigidBodyMotionConstraint line;\n            direction {_vec(data['direction'])};\n        }}"
                    )
                else:
                    raise ValueError(f"Unsupported sixDoF constraint type: {kind}")
            constraint_text = "\n".join(rendered)
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      dynamicMeshDict;
}}

dynamicFvMesh dynamicMotionSolverFvMesh;
motionSolverLibs
(
    \"libfvMotionSolvers.so\"
    \"{self.rigid_body_library}\"
);
solver sixDoFRigidBodyMotion;
diffusivity quadratic inverseDistance ( {body_patch} );

sixDoFRigidBodyMotionCoeffs
{{
    patches             ( {body_patch} );
    centreOfMass        {_vec(com)};
    mass                {_scalar(mass)};
    g                   {_vec(gravity)};
    momentOfInertia     {_vec(moment_of_inertia)};
    velocity            (0 0 0);
    orientation         (1 0 0 0 1 0 0 0 1);
    accelerationRelaxation 1.0;
    accelerationDamping    1.0;
    report              on;
    reportToFile        on;
    solver
    {{
        type Newmark;
    }}
    constraints
    {{
{constraint_text}
    }}
    restraints
    {{
{lines}    }}
}}
"""

    def configure_solver(self, solver: object) -> object:
        """Attach runtime libraries to an existing Foampilot solver.

        The helper does not replace the solver's own dictionary generation; it
        only composes the physics module with the existing ``controlDict``.
        Call ``solver.write_case()`` before ``write()`` when generating a full
        case.
        """
        control_dict = getattr(getattr(solver, "system", None), "controlDict", None)
        if control_dict is None or not hasattr(control_dict, "add_library"):
            raise TypeError("solver must expose system.controlDict.add_library()")
        for library in self.runtime_libraries():
            control_dict.add_library(library)
        if hasattr(solver, "transient"):
            solver.transient = True
        return solver

    def write(
        self,
        case_path: str | Path,
        *,
        cell_zone: str = "rotor",
        source_container: str = "fvOptions",
        **dynamic_mesh: object,
    ) -> dict[str, Path]:
        """Write only physics-owned dictionaries and return their paths."""
        if source_container not in {"fvOptions", "fvModels"}:
            raise ValueError("source_container must be 'fvOptions' or 'fvModels'")
        errors = self.validate()
        if errors:
            raise ValueError("Invalid floating turbine: " + "; ".join(errors))
        root = Path(case_path)
        constant = root / "constant"
        constant.mkdir(parents=True, exist_ok=True)
        source_path = constant / source_container
        paths = {source_container: source_path}
        source_path.write_text(
            self.render_source_dictionary(cell_zone=cell_zone, container=source_container),
            encoding="utf-8",
        )
        if self.mooring_lines:
            paths["dynamicMeshDict"] = constant / "dynamicMeshDict"
            paths["dynamicMeshDict"].write_text(self.render_dynamic_mesh(**dynamic_mesh), encoding="utf-8")
        return paths
