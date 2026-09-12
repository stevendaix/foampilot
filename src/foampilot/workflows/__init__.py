"""Workflows for FoamPilot.

Workflows are imperative Python scripts that compose generic capabilities
from core/ into domain-specific simulation pipelines. They are the
entry points for users running simulations.

Structure:
- workflows/medical/   : Medical and biomechanical flows (aorta, vessels, etc.)
- workflows/marine/    : Marine and offshore CFD (muffler, propulsion, etc.)
- workflows/urban/     : Urban canopy and neighborhood CFD
- workflows/energy/    : Energy sector applications (heat exchangers, etc.)
- workflows/wind/      : Wind energy and floating turbines
"""
from foampilot.workflows.medical import *
from foampilot.workflows.urban import *
from foampilot.workflows.wind import *

__all__ = []