"""Public Foampilot namespace with lazy optional imports."""
from __future__ import annotations

import importlib
import numpy as np

# Compatibility aliases required by older transitive dependencies such as nptyping.
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

# Lazy imports pointing to canonical core/ paths
_LAZY_ATTRS = {
    "Meshing": ("foampilot.core.base.meshing", "Meshing"),
    "CaseBuilder": ("foampilot.core.base.meshing", "CaseBuilder"),
    "create_case_structure": ("foampilot.core.meshing.ops", "create_case_structure"),
    "Solver": ("foampilot.solver", "Solver"),
    "ConstantDirectory": ("foampilot.constant.constantDirectory", "ConstantDirectory"),
    "SystemDirectory": ("foampilot.system.SystemDirectory", "SystemDirectory"),
    "Boundary": ("foampilot.boundaries.boundaries_dict", "Boundary"),
    "BoundaryFileHandler": ("foampilot.core.meshing.read_polymesh", "BoundaryFileHandler"),
    "STLAnalyzer": ("foampilot.core.meshing.stl_analyser", "STLAnalyzer"),
    "BlockMesher": ("foampilot.core.meshing.blockmesh", "BlockMesher"),
    "GmshMesher": ("foampilot.core.meshing.gmsh", "GmshMesher"),
    "SnappyMesher": ("foampilot.core.meshing.snappy", "SnappyMesher"),
    "DirectOpenFOAMExporter": ("foampilot.core.meshing.direct_openfoam_exporter", "DirectOpenFOAMExporter"),
    "GmshQualityAnalyzer": ("foampilot.core.meshing.quality", "GmshQualityAnalyzer"),
    "QualityThresholds": ("foampilot.core.meshing.quality", "QualityThresholds"),
    "QualityReport": ("foampilot.core.meshing.quality", "QualityReport"),
    "ElementQuality": ("foampilot.core.meshing.quality", "ElementQuality"),
    "CheckMeshParser": ("foampilot.core.meshing.quality", "CheckMeshParser"),
    "QualityGate": ("foampilot.core.meshing.quality", "QualityGate"),
    "OpenFOAMQualityAnalyzer": ("foampilot.core.meshing.quality", "OpenFOAMQualityAnalyzer"),
    "AdaptiveMeshImprover": ("foampilot.core.meshing.adaptation", "AdaptiveMeshImprover"),
    "write_rotating_zone": ("foampilot.core.meshing.ops", "write_rotating_zone"),
    "write_mesh_motion": ("foampilot.core.meshing.ops", "write_mesh_motion"),
    "restore_initial_fields": ("foampilot.core.meshing.ops", "restore_initial_fields"),
    "latex_pdf": ("foampilot.core.reporting.latex_pdf", "latex_pdf"),
    "ScientificDocument": ("foampilot.core.reporting.latex_pdf", "ScientificDocument"),
    "TypstRenderer": ("foampilot.core.reporting.typst_pdf", "TypstRenderer"),
    "ValueWithUnit": ("foampilot.core.units.manageunits", "ValueWithUnit"),
    "FluidMechanics": ("foampilot.core.physics.fluids_theory", "FluidMechanics"),
    "Functions": ("foampilot.core.physics.functions", "Functions"),
    "ResidualsPost": ("foampilot.core.postprocessing.residuals", "ResidualsPost"),
    "HumanGeometry": ("foampilot.extensions.medical.make_human", "HumanGeometry"),
    "OpenFOAMDictAddFile": ("foampilot.core.dictionaries.legacy", "OpenFOAMDictAddFile"),
    "CSVFoamIntegrator": ("foampilot.extensions.cfd.coupling_foam_csv", "CSVFoamIntegrator"),
    "WeatherFileEPW": ("foampilot.core.geometry.epw_reader", "WeatherFileEPW"),
    "AortaSurfaceCleaner": ("foampilot.core.geometry.surfaces.stl_cleanup", "AortaSurfaceCleaner"),
    "AortaCapMethod": ("foampilot.core.geometry.surfaces.stl_cleanup", "AortaCapMethod"),
    "create_closed_aorta_mesh": ("foampilot.core.geometry.surfaces.stl_cleanup", "create_closed_aorta_mesh"),
    "WindkesselModel": ("foampilot.workflows.medical.windkessel", "WindkesselModel"),
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