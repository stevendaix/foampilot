"""Declarative execution workflows for generated OpenFOAM cases."""

from foampilot.workflows.openfoam import (
    CommandStep,
    CopyStep,
    OpenFOAMWorkflow,
    RemoveStep,
    RestoreInitialFieldsStep,
    StepResult,
)
from foampilot.workflows.marine import (
    dtc_openfoam13_workflow,
    dtc_overset_workflow,
    maneuvering_turning_workflow,
    propeller_mrf_workflow,
    write_mrf_properties,
    write_openfoam13_rigid_body_mover,
    write_overset_dynamic_mesh,
)

__all__ = [
    "CommandStep",
    "CopyStep",
    "OpenFOAMWorkflow",
    "RemoveStep",
    "RestoreInitialFieldsStep",
    "StepResult",
    "dtc_openfoam13_workflow",
    "dtc_overset_workflow",
    "maneuvering_turning_workflow",
    "propeller_mrf_workflow",
    "write_mrf_properties",
    "write_openfoam13_rigid_body_mover",
    "write_overset_dynamic_mesh",
]
