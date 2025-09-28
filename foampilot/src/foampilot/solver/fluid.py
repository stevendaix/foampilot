from pathlib import Path
from foampilot.solver.base_solver import BaseSolver

class Fluid(BaseSolver):
    """
    Represents an OpenFOAM case configured to run with the fluid solver.
    """

    def __init__(self, case_path: str | Path, solver_name: str = "fluid"):
        super().__init__(case_path, solver_name)

    def update_case_specific_attributes(self):
        """
        Update or define attributes specific to the fluid case.
        """
        self.compressible = True
        self.with_gravity = True
        print("🔧 Setting up fluid-specific attributes")

# Enregistrer la classe dans le dictionnaire SOLVER_CLASSES de BaseSolver
BaseSolver.SOLVER_CLASSES["fluid"] = Fluid
