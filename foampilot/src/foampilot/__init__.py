"""Public Foampilot namespace with lazy optional imports."""
from __future__ import annotations

import importlib
import numpy as np

# Compatibility aliases for older transitive dependencies such as nptyping.
_COMPAT_ALIASES = {
    "bool8": "bool_", "object0": "object_", "int0": "int8", "uint0": "uint8",
    "float_": "float64", "longfloat": "longdouble", "singlecomplex": "complex64",
    "complex_": "complex128", "cfloat": "complex128", "clongfloat": "clongdouble",
    "longcomplex": "clongdouble", "void0": "void", "string_": "bytes_",
    "bytes0": "bytes_", "unicode_": "str_", "str0": "str_",
}
for _alias, _target in _COMPAT_ALIASES.items():
    if not hasattr(np, _alias) and hasattr(np, _target):
        setattr(np, _alias, getattr(np, _target))

_LAZY_ATTRS = {
    "Meshing": ("foampilot.base", "Meshing"),
    "CaseBuilder": ("foampilot.base", "CaseBuilder"),
    "create_case_structure": ("foampilot.base", "create_case_structure"),
    "Solver": ("foampilot.solver", "Solver"),
    "ConstantDirectory": ("foampilot.constant.constantDirectory", "ConstantDirectory"),
    "SystemDirectory": ("foampilot.system.SystemDirectory", "SystemDirectory"),
    "Boundary": ("foampilot.boundaries.boundaries_dict", "Boundary"),
    "BoundaryFileHandler": ("foampilot.commons.read_polymesh", "BoundaryFileHandler"),
    "STLAnalyzer": ("foampilot.commons", "STLAnalyzer"),
    "BlockMesher": ("foampilot.mesh", "BlockMesher"),
    "GmshMesher": ("foampilot.mesh", "GmshMesher"),
    "SnappyMesher": ("foampilot.mesh", "SnappyMesher"),
    "DirectOpenFOAMExporter": ("foampilot.mesh", "DirectOpenFOAMExporter"),
    "GmshQualityAnalyzer": ("foampilot.mesh", "GmshQualityAnalyzer"),
    "QualityThresholds": ("foampilot.mesh", "QualityThresholds"),
    "QualityReport": ("foampilot.mesh", "QualityReport"),
    "ElementQuality": ("foampilot.mesh", "ElementQuality"),
    "CheckMeshParser": ("foampilot.mesh", "CheckMeshParser"),
    "QualityGate": ("foampilot.mesh", "QualityGate"),
    "OpenFOAMQualityAnalyzer": ("foampilot.mesh", "OpenFOAMQualityAnalyzer"),
    "AdaptiveMeshImprover": ("foampilot.mesh", "AdaptiveMeshImprover"),
    "write_rotating_zone": ("foampilot.mesh", "write_rotating_zone"),
    "write_mesh_motion": ("foampilot.mesh", "write_mesh_motion"),
    "restore_initial_fields": ("foampilot.mesh", "restore_initial_fields"),
    "latex_pdf": ("foampilot.report", "latex_pdf"),
    "ScientificDocument": ("foampilot.report", "ScientificDocument"),
    "TypstRenderer": ("foampilot.report", "TypstRenderer"),
    "ValueWithUnit": ("foampilot.utilities", "ValueWithUnit"),
    "FluidMechanics": ("foampilot.utilities", "FluidMechanics"),
    "Functions": ("foampilot.utilities", "Functions"),
    "ResidualsPost": ("foampilot.utilities", "ResidualsPost"),
    "HumanGeometry": ("foampilot.utilities", "HumanGeometry"),
    "OpenFOAMDictAddFile": ("foampilot.utilities", "OpenFOAMDictAddFile"),
    "CSVFoamIntegrator": ("foampilot.utilities", "CSVFoamIntegrator"),
    "WeatherFileEPW": ("foampilot.utilities", "WeatherFileEPW"),
    "AortaSurfaceCleaner": ("foampilot.utilities", "AortaSurfaceCleaner"),
    "AortaCapMethod": ("foampilot.utilities", "AortaCapMethod"),
    "create_closed_aorta_mesh": ("foampilot.utilities", "create_closed_aorta_mesh"),
    "WindkesselModel": ("foampilot.model_addon.windkessel", "WindkesselModel"),
}

__all__ = sorted(_LAZY_ATTRS)


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_ATTRS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'foampilot' has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value
