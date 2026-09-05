"""Reusable technical core for FoamPilot v3."""

from .case import CaseLayout, DEFAULT_CASE_DIRECTORIES, create_case_structure
from .dictionaries import DictionaryWriter

__all__ = [
    "CaseLayout",
    "DEFAULT_CASE_DIRECTORIES",
    "DictionaryWriter",
    "create_case_structure",
]
