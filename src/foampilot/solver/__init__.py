# Importer les classes de solvers spécifiques (pour usage interne ou avancé)
from foampilot.solver.base_solver import BaseSolver
# from foampilot.solver.incompressible_fluid import incompressibleFluid
# from foampilot.solver.fluid import Fluid
# from foampilot.solver.incompressible_vof import IncompressibleVoF
# from foampilot.solver.solid import Solid

# Importer la classe Solver générique (interface principale pour l'utilisateur)
from foampilot.solver.solver import Solver
from foampilot.solver.marine_case import MarineCaseConfig
from foampilot.solver.marine_controls import PropellerCommand, RudderCommand, write_marine_controls
from foampilot.solver.marine_forces import PropellerForceModel, RudderForceModel, write_force_model
from foampilot.solver.marine_actuation_disk import ActuationDiskSource, actuation_disk_from_propeller, write_actuation_disk

# Importer le module de suivi de convergence
# Convergence monitoring is handled by SimulationReport in foampilot.reporting

# Liste des classes disponibles pour une utilisation facile
__all__ = [
    "Solver",
    "BaseSolver",
    "MarineCaseConfig",
    "PropellerCommand",
    "RudderCommand",
    "write_marine_controls",
    "PropellerForceModel",
    "RudderForceModel",
    "write_force_model",
    "ActuationDiskSource",
    "actuation_disk_from_propeller",
    "write_actuation_disk",
    "incompressibleFluid",
    "Fluid",
    "IncompressibleVoF",
    "Solid",
]
