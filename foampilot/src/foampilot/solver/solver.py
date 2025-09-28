from pathlib import Path
from typing import Optional
from foampilot.solver.base_solver import BaseSolver

class Solver:
    """
    Generic solver class that adapts based on user-defined attributes.
    """

    def __init__(self, case_path: str | Path):
        self.case_path = Path(case_path)
        self._solver: Optional[BaseSolver] = None
        self._compressible: bool = False
        self._with_gravity: bool = False
        self._is_vof: bool = False
        self._is_solid: bool = False

        # Create the initial solver based on default attributes
        self._update_solver()

    @property
    def compressible(self) -> bool:
        return self._compressible

    @compressible.setter
    def compressible(self, value: bool):
        self._compressible = value
        self._update_solver()

    @property
    def with_gravity(self) -> bool:
        return self._with_gravity

    @with_gravity.setter
    def with_gravity(self, value: bool):
        self._with_gravity = value
        self._update_solver()

    @property
    def is_vof(self) -> bool:
        return self._is_vof

    @is_vof.setter
    def is_vof(self, value: bool):
        self._is_vof = value
        self._update_solver()

    @property
    def is_solid(self) -> bool:
        return self._is_solid

    @is_solid.setter
    def is_solid(self, value: bool):
        self._is_solid = value
        self._update_solver()

    def _update_solver(self):
        """
        Update the solver instance based on current attributes.
        """
        if self._is_solid:
            solver_name = "solid"
        elif self._is_vof:
            solver_name = "incompressibleVoF" if not self._compressible else "compressibleVoF"
        else:
            solver_name = "incompressibleFluid" if not self._compressible else "fluid"

        # Create a new solver instance
        self._solver = BaseSolver.create(self.case_path, solver_name)

        # Update solver flags
        self._solver.compressible = self._compressible
        self._solver.with_gravity = self._with_gravity

    def setup_case(self):
        """
        Prepare the case directory and apply solver-specific attributes.
        """
        if self._solver is None:
            raise RuntimeError("Solver is not initialized.")
        self._solver.setup_case()

    def run_simulation(self):
        """
        Run the simulation.
        """
        if self._solver is None:
            raise RuntimeError("Solver is not initialized.")
        self._solver.run_simulation()

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying solver instance.
        """
        if self._solver is None:
            raise RuntimeError("Solver is not initialized.")
        return getattr(self._solver, name)
