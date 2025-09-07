# base/__init__.py

# Importer les classes principales de test_solver et meshing
from foampilot.utilities.dictonnary import OpenFOAMDictAddFile
from foampilot.utilities.epw_weather_reader import WeatherFileEPW
from foampilot.utilities.fluids_theory import FluidMechanics
from foampilot.utilities.manageunits import Quantity
from foampilot.utilities.function import Functions
from foampilot.utilities.residuals import ResidualsPost
# from .read_mesh import Quantity
