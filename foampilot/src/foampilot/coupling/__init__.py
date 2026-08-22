"""Coupling helpers for external OpenFOAM participants."""

from .external_coupled import (
    CoupledPatchData,
    ExternalCoupledTemperature,
    ExternalCouplingTimeout,
)

__all__ = [
    "CoupledPatchData",
    "ExternalCoupledTemperature",
    "ExternalCouplingTimeout",
]
