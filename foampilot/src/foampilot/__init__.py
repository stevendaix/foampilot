import numpy as np

# --- START MONKEY-PATCH FOR NUMPY 2.0 COMPATIBILITY (nptyping dependency) ---
# These aliases were removed or renamed in NumPy 2.0.
# We define them here to prevent nptyping (a dependency of classy_blocks) from failing on import.
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
if not hasattr(np, 'object0'):
    np.object0 = np.object_
if not hasattr(np, 'int0'):
    np.int0 = np.int8
if not hasattr(np, 'uint0'):
    np.uint0 = np.uint8
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'longfloat'):
    np.longfloat = np.longdouble
if not hasattr(np, 'singlecomplex'):
    np.singlecomplex = np.complex64
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128
if not hasattr(np, 'cfloat'):
    np.cfloat = np.complex128
if not hasattr(np, 'clongfloat'):
    np.clongfloat = np.clongdouble
if not hasattr(np, 'longcomplex'):
    np.longcomplex = np.clongdouble
if not hasattr(np, 'void0'):
    np.void0 = np.void
if not hasattr(np, 'string_'):
    np.string_ = np.bytes_
if not hasattr(np, 'bytes0'):
    np.bytes0 = np.bytes_
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_
if not hasattr(np, 'str0'):
    np.str0 = np.str_
# --- END MONKEY-PATCH ---

# project/__init__.py

# Importer tous les modules nécessaires
from foampilot.base import  Meshing
from foampilot.solver import  Solver
from foampilot.constant.constantDirectory import ConstantDirectory
from foampilot.system.SystemDirectory import SystemDirectory
from foampilot.boundaries.boundaries_dict import Boundary
from foampilot.commons.read_polymesh import BoundaryFileHandler
from foampilot.report import latex_pdf
from foampilot.utilities import Quantity, FluidMechanics