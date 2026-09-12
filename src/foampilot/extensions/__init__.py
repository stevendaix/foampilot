"""Domain-specific extensions for FoamPilot.

This module contains domain-specific capabilities that extend the core
functionality for particular application areas:

- marine: Marine and offshore CFD
- urban: Urban canopy and building aerodynamics
- energy: Energy sector applications
- medical: Medical and biomechanical flows

These extensions are not part of core/ because they are specialized
domain knowledge rather than generic capabilities.
"""
from foampilot.extensions.marine import *
from foampilot.extensions.urban import *
from foampilot.extensions.energy import *

__all__ = []