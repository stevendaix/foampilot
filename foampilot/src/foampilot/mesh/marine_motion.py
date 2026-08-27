"""Marine motion helpers for Foundation OpenFOAM 13 cases."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from foampilot.mesh.ops import write_dynamic_mesh_dict


FOUNDATION13_JOINTS = ("Px", "Py", "Pz", "Rx", "Ry", "Rz")


def write_six_dof_dynamic_mesh_dict(
    case_path: str | Path,
    *,
    body_name: str = "hull",
    patch_name: str = "hull",
    mass: float,
    centre_of_mass: Tuple[float, float, float],
    inertia: Tuple[float, float, float, float, float, float],
    transform_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    inner_distance: float,
    outer_distance: float,
    translation_damper_coeff: float = 0.0,
    rotation_damper_coeff: float = 0.0,
) -> Path:
    """Write a six-degree-of-freedom Foundation 13 rigid-body mover.

    This helper deliberately exposes all physical inputs and does not add
    artificial restraints by default.  A caller can use non-zero damping when
    reproducing a captive or numerically stabilized test case.
    """
    if mass <= 0:
        raise ValueError("mass must be strictly positive")
    if inner_distance <= 0 or outer_distance <= inner_distance:
        raise ValueError("outer_distance must be greater than inner_distance > 0")
    if len(centre_of_mass) != 3 or len(inertia) != 6:
        raise ValueError("centre_of_mass must have 3 values and inertia 6 values")

    return write_dynamic_mesh_dict(
        case_path,
        body_name=body_name,
        patch_name=patch_name,
        mass=mass,
        centre_of_mass=centre_of_mass,
        inertia=inertia,
        transform_origin=transform_origin,
        joints=FOUNDATION13_JOINTS,
        inner_distance=inner_distance,
        outer_distance=outer_distance,
        translation_damper_coeff=translation_damper_coeff,
        rotation_damper_coeff=rotation_damper_coeff,
    )
