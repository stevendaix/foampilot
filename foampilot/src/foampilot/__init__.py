import numpy as np

# Compatibility aliases for NumPy 2.x consumers used by optional geometry dependencies.
for _name, _value in {
    "bool8": np.bool_, "object0": np.object_, "int0": np.int8, "uint0": np.uint8,
    "float_": np.float64, "longfloat": np.longdouble, "singlecomplex": np.complex64,
    "complex_": np.complex128, "cfloat": np.complex128, "clongfloat": np.clongdouble,
    "longcomplex": np.clongdouble, "void0": np.void, "string_": np.bytes_,
    "bytes0": np.bytes_, "unicode_": np.str_, "str0": np.str_,
}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _value)

# Keep the lightweight OpenFOAM 13 integration importable without optional CFD/geometry
# packages. When the full dependency set is installed, preserve all legacy exports.
try:
    from foampilot.base import CaseBuilder, Meshing, create_case_structure
    from foampilot.solver import Solver
    from foampilot.constant.constantDirectory import ConstantDirectory
    from foampilot.system.SystemDirectory import SystemDirectory
    from foampilot.boundaries.boundaries_dict import Boundary
    from foampilot.commons.read_polymesh import BoundaryFileHandler
    from foampilot.commons import STLAnalyzer
    from foampilot.mesh import (
        BlockMesher, GmshMesher, SnappyMesher, DirectOpenFOAMExporter,
        GmshQualityAnalyzer, QualityThresholds, QualityReport, ElementQuality,
        CheckMeshParser, QualityGate, OpenFOAMQualityAnalyzer,
        AdaptiveMeshImprover, write_rotating_zone, write_mesh_motion,
        restore_initial_fields,
    )
    from foampilot.report import latex_pdf, ScientificDocument, TypstRenderer
    from foampilot.utilities import (
        ValueWithUnit, FluidMechanics, Functions, ResidualsPost, HumanGeometry,
        OpenFOAMDictAddFile, CSVFoamIntegrator, WeatherFileEPW, AortaSurfaceCleaner,
        AortaCapMethod, create_closed_aorta_mesh,
    )
    from foampilot.model_addon.windkessel import WindkesselModel
except ModuleNotFoundError as _optional_dependency_error:
    # The dedicated subpackages remain usable; the original import error is available
    # for diagnostics rather than being silently swallowed by production code.
    OPTIONAL_DEPENDENCY_ERROR = _optional_dependency_error
