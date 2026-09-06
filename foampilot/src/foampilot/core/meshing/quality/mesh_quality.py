"""Mesh quality analysis for Gmsh meshes.

This module re-exports the main quality analyzer and data classes
from :mod:`foampilot.core.meshing.quality.gmsh_quality` to provide a
short, example-friendly import path.
"""

from __future__ import annotations

from foampilot.core.meshing.quality.gmsh_quality import (
    ElementQuality,
    GmshQualityAnalyzer,
    QualityReport,
    QualityThresholds,
)

__all__ = [
    "ElementQuality",
    "GmshQualityAnalyzer",
    "QualityReport",
    "QualityThresholds",
]
