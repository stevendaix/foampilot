"""Marine OpenFOAM workflow builders.

The functions in this module describe the operational patterns shared by three
published OpenFOAM studies: free-running ship manoeuvres, a rotating-frame
propeller calculation, and an overset moving-hull calculation.  They create
new, inspectable input dictionaries and command sequences; they do not copy
third-party geometries or case files.
"""

from __future__ import annotations

from pathlib import Path

from foampilot.base.openFOAMFile import OpenFOAMFile
from foampilot.workflows.openfoam import OpenFOAMWorkflow


def write_mrf_properties(
    case_path: str | Path,
    *,
    cell_zone: str = "rotor",
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    omega: float = 314.16,
    non_rotating_patches: tuple[str, ...] = ("AMI1", "AMI2"),
) -> Path:
    """Write a legacy OpenCFD ``MRFProperties`` dictionary.

    ``rhoSimpleFoam`` studies based on the rotating-reference-frame approach
    require the cell zone to exist in the mesh.  The caller remains
    responsible for creating the zone during meshing.
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


def write_overset_dynamic_mesh(
    case_path: str | Path,
    *,
    cell_set: str = "c1",
    body_name: str = "hull",
    patch_name: str = "hull",
    mass: float = 412.73,
    centre_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0),
    inertia: tuple[float, float, float, float, float, float] = (40.0, 0.0, 0.0, 921.0, 0.0, 921.0),
    transform_origin: tuple[float, float, float] = (2.929541, 0.0, 0.2),
    joints: tuple[str, ...] = ("Pz", "Ry"),
) -> Path:
    """Write a two-degree-of-freedom rigid-body overset ``dynamicMeshDict``.

    It targets legacy OpenCFD solvers such as ``overInterDyMFoam``.  The
    generated configuration is purposefully parameterised so the same pattern
    can be used for a DTC-like model or an independent hull geometry.
    """
    case_path = Path(case_path)
    constant_path = case_path / "constant"
    constant_path.mkdir(parents=True, exist_ok=True)
    joint_blocks = "\n".join(f"{{ type {joint}; }}" for joint in joints)
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
                        "joint": {"type": "composite", "joints": "(\n" + joint_blocks + "\n)"},
                        "patches": "(" + patch_name + ")",
                        "innerDistance": 100,
                        "outerDistance": 200,
                    }
                },
                "restraints": {
                    "translationDamper": {"type": "linearDamper", "body": body_name, "coeff": 8596},
                    "rotationDamper": {"type": "sphericalAngularDamper", "body": body_name, "coeff": 11586},
                },
            }
        },
    )
    path = constant_path / "dynamicMeshDict"
    dictionary.write_file(path)
    return path


def maneuvering_turning_workflow(root: str | Path, processors: int = 8) -> OpenFOAMWorkflow:
    """Build the two-stage free-running turning workflow.

    The root must contain ``hull``, ``rudder`` and ``background`` subcases,
    with the latter providing ``0.orig``, stage-specific control dictionaries,
    and stage-specific ``dynamicMeshDict`` files.  The custom maneuvering
    library must be installed in the active OpenFOAM environment before this
    workflow is run.
    """
    if processors < 1:
        raise ValueError("processors must be at least one")
    workflow = OpenFOAMWorkflow(root, "free-running-turning")
    workflow.add_copy("hull-geometry", "hull/Geometry", "hull/constant/triSurface")
    workflow.add_command("hull-block-mesh", "blockMesh", cwd="hull")
    workflow.add_command("hull-topo-set", "topoSet", "-dict", "system/topoSetDict", cwd="hull")
    workflow.add_command("hull-refine", "refineMesh", "-dict", "system/refineMeshDict", "-overwrite", cwd="hull")
    workflow.add_command("hull-decompose", "decomposePar", cwd="hull")
    workflow.add_command("hull-snappy", "mpirun", "--oversubscribe", "-np", str(processors), "snappyHexMesh", "-parallel", "-dict", "system/snappyHexMeshDict", "-overwrite", cwd="hull")
    workflow.add_command("hull-reconstruct", "redistributePar", "-reconstruct", "-constant", "-overwrite", cwd="hull")
    workflow.add_command("hull-renumber", "renumberMesh", "-constant", "-overwrite", cwd="hull")
    workflow.add_command("hull-check-mesh", "checkMesh", cwd="hull")
    workflow.add_copy("rudder-geometry", "rudder/Geometry/rudder.stl", "rudder/constant/triSurface/rudder.stl")
    workflow.add_command("rudder-block-mesh", "blockMesh", cwd="rudder")
    workflow.add_command("rudder-features", "surfaceFeatureExtract", cwd="rudder")
    workflow.add_command("rudder-decompose", "decomposePar", cwd="rudder")
    workflow.add_command("rudder-snappy", "mpirun", "--oversubscribe", "-np", str(processors), "snappyHexMesh", "-parallel", "-dict", "system/snappyHexMeshDict", "-overwrite", cwd="rudder")
    workflow.add_command("rudder-reconstruct", "redistributePar", "-reconstruct", "-constant", "-overwrite", cwd="rudder")
    workflow.add_command("rudder-renumber", "renumberMesh", "-constant", "-overwrite", cwd="rudder")
    workflow.add_command("rudder-check-mesh", "checkMesh", cwd="rudder")
    workflow.add_command("background-block-mesh", "blockMesh", "-dict", "system/blockMeshDict", cwd="background")
    for iteration in range(1, 6):
        workflow.add_command(f"background-topo-set-{iteration}", "topoSet", "-dict", f"system/topoSetDict.{iteration}", cwd="background")
        workflow.add_command(f"background-refine-{iteration}", "refineMesh", "-dict", "system/refineMeshDict", "-overwrite", cwd="background")
    workflow.add_copy("restore-initial-fields", "background/0.orig", "background/0")
    workflow.add_copy("select-propulsion-control", "background/system/controlDict.propulsion", "background/system/controlDict")
    workflow.add_copy("select-propulsion-motion", "background/constant/dynamicMeshDict.propulsion", "background/constant/dynamicMeshDict")
    workflow.add_command("merge-hull", "mergeMeshes", ".", "../hull", "-overwrite", cwd="background")
    workflow.add_command("merge-rudder", "mergeMeshes", ".", "../rudder", "-overwrite", cwd="background")
    workflow.add_command("create-overset-sets", "topoSet", "-dict", "system/topoSetDict.cHullRudder", cwd="background")
    workflow.add_command("set-fields-hull", "setFields", "-dict", "system/setFieldsDict.1", cwd="background")
    workflow.add_command("set-fields-rudder", "setFields", "-dict", "system/setFieldsDict.2", cwd="background")
    workflow.add_command("decompose-propulsion", "decomposePar", "-force", cwd="background")
    workflow.add_command("run-self-propulsion", "mpirun", "--oversubscribe", "-np", str(processors), "overInterDyMFoam", "-parallel", cwd="background")
    workflow.add_copy("select-turning-control", "background/system/controlDict.turning", "background/system/controlDict")
    workflow.add_copy("select-turning-motion", "background/constant/dynamicMeshDict.turning", "background/constant/dynamicMeshDict")
    workflow.add_copy("copy-maneuver-state", "background/maneuvers", "background/processor0/10/uniform/maneuvers")
    workflow.add_command("run-turning", "mpirun", "--oversubscribe", "-np", str(processors), "overInterDyMFoam", "-parallel", cwd="background")
    return workflow


def propeller_mrf_workflow(root: str | Path, processors: int = 4) -> OpenFOAMWorkflow:
    """Build the simulation stage of a steady MRF propeller study.

    ``root`` is the already-meshed simulation case; its mesh preparation can
    use cfMesh or another mesher as long as it creates the configured rotor
    cell zone and interfaces.
    """
    if processors < 1:
        raise ValueError("processors must be at least one")
    workflow = OpenFOAMWorkflow(root, "steady-mrf-propeller")
    workflow.add_remove("clean-parallel-output", "processor0", "processor1", "processor2", "processor3", "postProcessing", "logs")
    workflow.add_copy("restore-initial-fields", "0.orig", "0")
    workflow.add_command("decompose", "decomposePar")
    workflow.add_command("solve", "mpirun", "--oversubscribe", "-np", str(processors), "rhoSimpleFoam", "-parallel")
    workflow.add_command("reconstruct", "reconstructPar")
    workflow.add_command("wall-shear-stress", "rhoSimpleFoam", "-postProcess", "-func", "wallShearStress")
    return workflow


def dtc_overset_workflow(root: str | Path, processors: int = 4) -> OpenFOAMWorkflow:
    """Build an overset moving-hull workflow with mesh preparation and solve."""
    if processors < 1:
        raise ValueError("processors must be at least one")
    workflow = OpenFOAMWorkflow(root, "dtc-moving-overset")
    workflow.add_copy("hull-surface", "hull/hull.stl", "hull/constant/triSurface/hull.stl")
    workflow.add_command("hull-block-mesh", "blockMesh", cwd="hull")
    workflow.add_command("hull-topo-set", "topoSet", "-dict", "system/topoSetDict", cwd="hull")
    workflow.add_command("hull-refine", "refineMesh", "-dict", "system/refineMeshDict", "-overwrite", cwd="hull")
    workflow.add_command("hull-features", "surfaceFeatureExtract", cwd="hull")
    workflow.add_command("hull-decompose", "decomposePar", cwd="hull")
    workflow.add_command("hull-snappy", "mpirun", "--oversubscribe", "-np", str(processors), "snappyHexMesh", "-parallel", "-overwrite", cwd="hull")
    workflow.add_command("hull-reconstruct", "redistributePar", "-reconstruct", "-constant", "-overwrite", cwd="hull")
    workflow.add_command("hull-renumber", "renumberMesh", "-constant", "-overwrite", cwd="hull")
    workflow.add_command("background-block-mesh", "blockMesh", cwd="background")
    for iteration in range(1, 6):
        workflow.add_command(f"background-topo-set-{iteration}", "topoSet", "-dict", f"system/topoSetDict.{iteration}", cwd="background")
        workflow.add_command(f"background-refine-{iteration}", "refineMesh", "-dict", "system/refineMeshDict", "-overwrite", cwd="background")
    workflow.add_copy("restore-initial-fields", "background/0.orig", "background/0")
    workflow.add_command("merge-hull", "mergeMeshes", ".", "../hull", "-overwrite", cwd="background")
    workflow.add_command("set-overset-zones", "topoSet", cwd="background")
    workflow.add_command("set-fields", "setFields", cwd="background")
    workflow.add_command("decompose", "decomposePar", "-force", cwd="background")
    workflow.add_command("solve", "mpirun", "--oversubscribe", "-np", str(processors), "overInterDyMFoam", "-parallel", cwd="background")
    workflow.add_command("reconstruct", "reconstructPar", cwd="background")
    return workflow
