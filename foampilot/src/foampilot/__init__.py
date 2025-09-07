# project/__init__.py

# Importer tous les modules nécessaires
from foampilot.base import  Meshing
from foampilot.solver import  incompressibleFluid
from foampilot.constant.constantDirectory import ConstantDirectory
from foampilot.system.SystemDirectory import SystemDirectory
from foampilot.boundaries.boundaries_dict import Boundary
from foampilot.commons.read_polymesh import BoundaryFileHandler
from foampilot.report import latex_pdf
# from .utilities import Utilities

