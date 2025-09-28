from pathlib import Path
import subprocess
from foampilot.solver.base_solver import BaseSolver


class incompressibleFluid(BaseSolver):
    """
    Represents an OpenFOAM case configured to run with the incompressibleFluid solver.

    This class reuses BaseSolver for common behavior but keeps solver-specific hooks.
    """

    def __init__(self, path_case: str | Path):
        super().__init__(path_case, solver_name="incompressibleFluid")

    def update_case_specific_attributes(self):
        """
        Update or define attributes specific to the incompressibleFluid case.
        """
        # Exemple: définir flags par défaut spécifiques au solver
        self.compressible = False
        self.with_gravity = True
        print("Updating incompressibleFluid-specific attributes")

    def run_simulation(self, log_filename: str = "log.incompressibleFluid"):
        """
        Run incompressibleFluid using the project's preferred launcher.
        """
        try:
            # Use project's foamRun wrapper with explicit solver if desired
            self.run_command(["foamRun", "-solver", "incompressibleFluid"], log_filename)
        except subprocess.CalledProcessError:
            # keep the same error behavior as before
            print(f"❌ Simulation échouée. Vérifie le log dans {self.case_path / log_filename}")
            raise RuntimeError(f"Simulation échouée (voir {self.case_path / log_filename})")