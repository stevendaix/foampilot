# foampilot/solver/__init__.py

# Importer les classes de solvers
from foampilot.solver.base_solver import BaseSolver
from foampilot.solver.incompressible_fluid import incompressibleFluid
from foampilot.solver.fluid import Fluid
from foampilot.solver.incompressible_vof import IncompressibleVoF
from foampilot.solver.solid import Solid

# Liste des classes disponibles pour une utilisation facile
__all__ = [
    "BaseSolver",
    "incompressibleFluid",
    "Fluid",
    "IncompressibleVoF",
    "Solid",
]
