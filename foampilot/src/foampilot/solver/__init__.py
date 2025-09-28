# Importer les classes de solvers spécifiques (pour usage interne ou avancé)
from foampilot.solver.base_solver import BaseSolver
from foampilot.solver.incompressible_fluid import incompressibleFluid
from foampilot.solver.fluid import Fluid
from foampilot.solver.incompressible_vof import IncompressibleVoF
from foampilot.solver.solid import Solid

# Importer la classe Solver générique (interface principale pour l'utilisateur)
from foampilot.solver.solver import Solver

# Liste des classes disponibles pour une utilisation facile
__all__ = [
    "Solver",  # Interface principale recommandée
    "BaseSolver",  # Pour un usage avancé ou interne
    "incompressibleFluid",
    "Fluid",
    "IncompressibleVoF",
    "Solid",
]
