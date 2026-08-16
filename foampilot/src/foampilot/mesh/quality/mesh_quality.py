"""Mesh quality analysis for Gmsh meshes.

This module re-exports the main quality analyzer and data classes
from :mod:`foampilot.mesh.quality.gmsh_quality` to provide a
short, example-friendly import path.
"""

from __future__ import annotations

from foampilot.mesh.quality.gmsh_quality import (
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
