# base/__init__.py

# Importer les classes principales de test_solver et meshing
from foampilot.utilities.dictonnary import OpenFOAMDictAddFile
from foampilot.utilities.epw_weather_reader import WeatherFileEPW
from foampilot.utilities.fluids_theory import FluidMechanics
from foampilot.utilities.manageunits import ValueWithUnit
from foampilot.utilities.function import Functions
from foampilot.utilities.residuals import ResidualsPost
try:
    from foampilot.utilities.make_human import HumanGeometry
except ImportError:  # Optional CAD visualization dependency
    HumanGeometry = None
from foampilot.utilities.coupling_foam_csv import CSVFoamIntegrator
try:
    from foampilot.utilities.stl_cleanup import AortaSurfaceCleaner,AortaCapMethod, create_closed_aorta_mesh
except ImportError:  # Optional surface-processing dependencies
    AortaSurfaceCleaner = None
    AortaCapMethod = None
    create_closed_aorta_mesh = None
try:
    from foampilot.utilities.nifti_to_stl import nifti_to_stl, nifti_to_stl_multisurface
except ImportError:  # Optional medical-image dependency
    nifti_to_stl = None
    nifti_to_stl_multisurface = None
# from foampilot.utilities.import_tbad import prepare_tbad_mesh, save_nifti_to_obj
# from .read_mesh import ValueWithUnit
