"""General mesh operation helpers.

These functions are intentionally solver-agnostic so they can be reused
across ``blockMesh``, ``snappyHexMesh``, ``gmsh``, overset, MRF, and
Foundation 13 rigid-body mover workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from foampilot.base.openFOAMFile import OpenFOAMFile


def write_rotating_zone(
    case_path: str | Path,
    *,
    cell_zone: str,
    origin: Tuple[float, float, float],
    axis: Tuple[float, float, float],
    omega: float,
    non_rotating_patches: Tuple[str, ...],
) -> Path:
    """Write an MRF zone dictionary.

    Parameters
    ----------
    case_path:
        Root path of the OpenFOAM case.
    cell_zone:
        Name of the cell zone to rotate.
    origin:
        Rotation origin coordinates.
    axis:
        Rotation axis vector.
    omega:
        Angular speed in rad/s.
    non_rotating_patches:
        Patch names that must not rotate.
    """
    case_path = Path(case_path)
    constant_path = case_path / "constant"
    constant_path.mkdir(parents=True, exist_ok=True)

    dictionary = OpenFOAMFile(
        object_name="MRFProperties",
        MRF1={
            "cellZone": cell_zone,
            "active": "yes",
            "nonRotatingPatches": "(" + " ".join(non_rotating_patches) + ")",
            "origin": "(" + " ".join(str(value) for value in origin) + ")",
            "axis": "(" + " ".join(str(value) for value in axis) + ")",
            "omega": omega,
        },
    )
    path = constant_path / "MRFProperties"
    dictionary.write_file(path)
    return path


def write_mesh_motion(
    case_path: str | Path,
    *,
    mode: str,
    body_name: str,
    patch_name: str,
    mass: float,
    centre_of_mass: Tuple[float, float, float],
    inertia: Tuple[float, float, float, float, float, float],
    transform_origin: Tuple[float, float, float],
    joints: Tuple[str, ...],
    cell_set: Optional[str] = None,
    inner_distance: Optional[float] = None,
    outer_distance: Optional[float] = None,
    translation_damper_coeff: float,
    rotation_damper_coeff: float,
    legacy_inner_distance: float,
    legacy_outer_distance: float,
) -> Path:
    """Write a mesh-motion dictionary for overset or moving-body simulations.

    Parameters
    ----------
    mode:
        ``openfoam13`` for Foundation 13 ``mover`` blocks,
        ``legacy`` for OpenCFD ``dynamicOversetFvMesh`` solvers.
    """
    case_path = Path(case_path)
    constant_path = case_path / "constant"
    constant_path.mkdir(parents=True, exist_ok=True)

    joint_blocks = "\n".join(f"{{ type {joint}; }}" for joint in joints)

    if mode == "openfoam13":
        if inner_distance is None or outer_distance is None:
            raise ValueError("inner_distance and outer_distance are required for openfoam13 mode")
        dictionary = OpenFOAMFile(
            object_name="dynamicMeshDict",
            mover={
                "type": "motionSolver",
                "libs": '("librigidBodyMeshMotion.so")',
                "motionSolver": "rigidBodyMotion",
                "report": "on",
                "solver": {"type": "Newmark"},
                "accelerationRelaxation": 0.4,
                "bodies": {
                    body_name: {
                        "type": "rigidBody",
                        "parent": "root",
                        "centreOfMass": "(" + " ".join(str(value) for value in centre_of_mass) + ")",
                        "mass": mass,
                        "inertia": "(" + " ".join(str(value) for value in inertia) + ")",
                        "transform": "(1 0 0 0 1 0 0 0 1) (" + " ".join(str(value) for value in transform_origin) + ")",
                        "joint": {"type": "composite", "joints": "(\n" + joint_blocks + "\n);"},
                        "patches": "(" + patch_name + ")",
                        "innerDistance": inner_distance,
                        "outerDistance": outer_distance,
                    }
                },
                "restraints": {
                    "translationDamper": {"type": "linearDamper", "body": body_name, "coeff": translation_damper_coeff},
                    "rotationDamper": {"type": "sphericalAngularDamper", "body": body_name, "coeff": rotation_damper_coeff},
                },
            },
        )
    elif mode == "legacy":
        if cell_set is None:
            raise ValueError("cell_set is required for legacy mode")
        dictionary = OpenFOAMFile(
            object_name="dynamicMeshDict",
            dynamicFvMesh="dynamicOversetFvMesh",
            solvers={
                "boat": {
                    "motionSolverLibs": "(librigidBodyMeshMotion)",
                    "motionSolver": "rigidBodyMotion",
                    "report": "on",
                    "cellSet": cell_set,
                    "solver": {"type": "Newmark"},
                    "accelerationRelaxation": 0.8,
                    "accelerationDamping": 0.9,
                    "nIter": 3,
                    "bodies": {
                        body_name: {
                            "type": "rigidBody",
                            "parent": "root",
                            "mass": mass,
                            "centreOfMass": "(" + " ".join(str(value) for value in centre_of_mass) + ")",
                            "inertia": "(" + " ".join(str(value) for value in inertia) + ")",
                            "transform": "(1 0 0 0 1 0 0 0 1) (" + " ".join(str(value) for value in transform_origin) + ")",
                            "joint": {"type": "composite", "joints": "\n        (\n" + joint_blocks + "\n        );"},
                            "patches": "(" + patch_name + ")",
                            "innerDistance": legacy_inner_distance,
                            "outerDistance": legacy_outer_distance,
                        }
                    },
                    "restraints": {
                        "translationDamper": {"type": "linearDamper", "body": body_name, "coeff": translation_damper_coeff},
                        "rotationDamper": {"type": "sphericalAngularDamper", "body": body_name, "coeff": rotation_damper_coeff},
                    },
                }
            },
        )
    else:
        raise ValueError(f"Unknown mesh motion mode: {mode!r}. Expected 'openfoam13' or 'legacy'.")

    path = constant_path / "dynamicMeshDict"
    dictionary.write_file(path)
    return path


def write_dynamic_mesh_dict(
    case_path: str | Path,
    *,
    body_name: str = "hull",
    patch_name: str = "hull",
    mass: float = 412.73,
    centre_of_mass: Tuple[float, float, float] = (2.929541, 0.0, 0.2),
    inertia: Tuple[float, float, float, float, float, float] = (40.0, 0.0, 0.0, 921.0, 0.0, 921.0),
    transform_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    joints: Tuple[str, ...] = ("Pz", "Ry"),
    inner_distance: float = 0.3,
    outer_distance: float = 1.0,
    translation_damper_coeff: float = 8596,
    rotation_damper_coeff: float = 11586,
) -> Path:
    """Write an OpenFOAM 13 rigid-body ``dynamicMeshDict``.

    Parameters are fully configurable so this helper can be reused for any
    Foundation 13 overset/moving-body case.
    """
    joint_blocks = "\n".join(f"        {{ type {joint}; }}" for joint in joints)
    outer_distance_str = str(outer_distance).rstrip("0").rstrip(".") if isinstance(outer_distance, float) and outer_distance == int(outer_distance) else str(outer_distance)
    inner_distance_str = str(inner_distance).rstrip("0").rstrip(".") if isinstance(inner_distance, float) and inner_distance == int(inner_distance) else str(inner_distance)

    content = f"""FoamFile
{{
    version     2.0;
    format     ascii;
    class     dictionary;
    object     dynamicMeshDict;
}}

mover
{{
    type motionSolver;
    libs ("librigidBodyMeshMotion.so");
    motionSolver rigidBodyMotion;
    report on;
    solver
    {{
        type Newmark;
    }}
    accelerationRelaxation 0.4;
    bodies
    {{
        {body_name}
        {{
            type rigidBody;
            parent root;
            centreOfMass ({" ".join(str(value) for value in centre_of_mass)});
            mass {mass};
            inertia ({" ".join(str(value) for value in inertia)});
            transform (1 0 0 0 1 0 0 0 1) ({" ".join(str(value) for value in transform_origin)});
            joint
            {{
                type composite;
                joints (
{joint_blocks}
);
            }}
            patches ({patch_name});
            innerDistance {inner_distance_str};
            outerDistance {outer_distance_str};
        }}
    }}
    restraints
    {{
        translationDamper
        {{
            type linearDamper;
            body {body_name};
            coeff {translation_damper_coeff};
        }}
        rotationDamper
        {{
            type sphericalAngularDamper;
            body {body_name};
            coeff {rotation_damper_coeff};
        }}
    }}
}}
"""

    case_path = Path(case_path)
    constant_path = case_path / "constant"
    constant_path.mkdir(parents=True, exist_ok=True)
    path = constant_path / "dynamicMeshDict"
    path.write_text(content)
    return path


def create_case_structure(
    case_path: str | Path,
    *,
    extra_dirs: Tuple[str, ...] = ("triSurface", "geometry", "postProcessing"),
) -> Path:
    """Create the standard OpenFOAM case directory tree.

    Ensures ``0``, ``0.orig``, ``constant``, ``system`` and any additional
    directories exist.  Returns the resolved ``case_path``.
    """
    case_path = Path(case_path).expanduser().resolve()
    for sub in ("0", "0.orig", "constant", "system", *extra_dirs):
        (case_path / sub).mkdir(parents=True, exist_ok=True)
    return case_path


def restore_initial_fields(
    case_path: str | Path,
    source: str | Path = "0.orig",
    destination: str | Path = "0",
) -> None:
    """Restore initial fields from ``.orig`` backups or directories.

    Parameters
    ----------
    case_path:
        Root path of the OpenFOAM case.
    source:
        Source directory or glob pattern for ``.orig`` files.
    destination:
        Destination directory for restored fields.
    """
    import shutil

    case_path = Path(case_path)
    source_dir = case_path / source
    destination_dir = case_path / destination

    if source_dir.is_dir() and any(source_dir.iterdir()):
        destination_dir.mkdir(parents=True, exist_ok=True)
        for field in source_dir.iterdir():
            if field.is_file():
                shutil.copy2(field, destination_dir / field.name)
        return

    if destination_dir.is_dir():
        for original_field in sorted(destination_dir.glob("*.orig")):
            field_name = original_field.name.removesuffix(".orig")
            shutil.copy2(original_field, destination_dir / field_name)
