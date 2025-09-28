from pathlib import Path
from foampilot.solver.base_solver import BaseSolver

class Solid(BaseSolver):
    """
    Represents an OpenFOAM case configured to run with the solid solver.
    """

    def __init__(self, case_path: str | Path, solver_name: str = "solid"):
        super().__init__(case_path, solver_name)

    def update_case_specific_attributes(self):
        """
        Update or define attributes specific to the solid case.
        """
        self.compressible = False
        self.with_gravity = False
        print("🔧 Setting up solid-specific attributes")

# Enregistrer la classe dans le dictionnaire SOLVER_CLASSES de BaseSolver
BaseSolver.SOLVER_CLASSES["solid"] = Solid
