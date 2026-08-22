"""Declarative execution workflows for generated OpenFOAM cases."""

from foampilot.workflows.openfoam import (
    CommandStep,
    CopyStep,
    OpenFOAMWorkflow,
    RemoveStep,
    StepResult,
)
from foampilot.workflows.marine import (
    dtc_overset_workflow,
    maneuvering_turning_workflow,
    propeller_mrf_workflow,
    write_mrf_properties,
    write_overset_dynamic_mesh,
)

__all__ = [
    "CommandStep",
    "CopyStep",
    "OpenFOAMWorkflow",
    "RemoveStep",
    "StepResult",
    "dtc_overset_workflow",
    "maneuvering_turning_workflow",
    "propeller_mrf_workflow",
    "write_mrf_properties",
    "write_overset_dynamic_mesh",
]
