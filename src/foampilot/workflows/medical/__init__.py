"""Medical and biomechanical flow workflows.

This module contains workflow scripts for medical and biomechanical
simulations including vascular flows, windkessel models, etc.
"""
from foampilot.workflows.medical.windkessel import Windkessel, WindkesselModel

__all__ = ["Windkessel", "WindkesselModel"]