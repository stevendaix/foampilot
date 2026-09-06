# Public utility exports.
from foampilot.utilities.dictonnary import OpenFOAMDictAddFile
from foampilot.utilities.epw_weather_reader import WeatherFileEPW
from foampilot.utilities.fluids_theory import FluidMechanics
from foampilot.utilities.manageunits import ValueWithUnit
from foampilot.utilities.function import Functions
from foampilot.utilities.residuals import ResidualsPost
from foampilot.utilities.coupling_foam_csv import CSVFoamIntegrator


def __getattr__(name):
    """Load optional geometry helpers only when explicitly requested."""
    if name == "HumanGeometry":
        from foampilot.utilities.make_human import HumanGeometry
        return HumanGeometry
    if name in {"AortaSurfaceCleaner", "AortaCapMethod", "create_closed_aorta_mesh"}:
        from foampilot.utilities.stl_cleanup import (
            AortaCapMethod,
            AortaSurfaceCleaner,
            create_closed_aorta_mesh,
        )
        return locals()[name]
    if name in {"nifti_to_stl", "nifti_to_stl_multisurface"}:
        from foampilot.utilities.nifti_to_stl import nifti_to_stl, nifti_to_stl_multisurface
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# from foampilot.utilities.import_tbad import prepare_tbad_mesh, save_nifti_to_obj
# from .read_mesh import ValueWithUnit
