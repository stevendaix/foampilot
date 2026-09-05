"""Reusable OpenFOAM dictionary capabilities.

The classes are internal building blocks for high-level FoamPilot objects.
"""
from .writer import DictionaryWriter, FoamDict

__all__ = ["DictionaryWriter", "FoamDict"]
