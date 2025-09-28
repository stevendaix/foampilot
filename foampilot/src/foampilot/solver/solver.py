from pathlib import Path
from typing import Optional, Callable, List
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
        self._error_handlers: List[Callable[[str], None]] = []
        self._event_handlers: List[Callable[[str, str], None]] = []

        # Create the initial solver based on default attributes
        self._update_solver()

    def add_error_handler(self, handler: Callable[[str], None]):
        """Add a custom error handler."""
        self._error_handlers.append(handler)

    def add_event_handler(self, handler: Callable[[str, str], None]):
        """Add a custom event handler."""
        self._event_handlers.append(handler)

    def _notify_error(self, message: str):
        """Notify all error handlers."""
        print(f"⚠️ Error: {message}")
        for handler in self._error_handlers:
            handler(message)

    def _notify_event(self, event_type: str, message: str):
        """Notify all event handlers."""
        print(f"🔔 Event: {event_type} - {message}")
        for handler in self._event_handlers:
            handler(event_type, message)

    @property
    def compressible(self) -> bool:
        return self._compressible

    @compressible.setter
    def compressible(self, value: bool):
        if self._is_solid and value:
            self._notify_error("Solid simulations are always incompressible.")
            return
        self._compressible = value
        self._update_solver()

    @property
    def with_gravity(self) -> bool:
        return self._with_gravity

    @with_gravity.setter
    def with_gravity(self, value: bool):
        if self._is_solid and value:
            self._notify_error("Gravity cannot be enabled for solid simulations.")
            return
        self._with_gravity = value
        self._update_solver()

    @property
    def is_vof(self) -> bool:
        return self._is_vof

    @is_vof.setter
    def is_vof(self, value: bool):
        if self._is_solid and value:
            self._notify_error("A simulation cannot be both VoF and solid.")
            return
        self._is_vof = value
        self._update_solver()

    @property
    def is_solid(self) -> bool:
        return self._is_solid

    @is_solid.setter
    def is_solid(self, value: bool):
        if self._is_vof and value:
            self._notify_error("A simulation cannot be both VoF and solid.")
            return
        if value:
            self._compressible = False  # Force incompressible for solids
            self._with_gravity = False  # Force no gravity for solids
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

        # Notify that the solver is about to change
        old_solver_name = self._solver.solver_name if self._solver else "None"
        self._notify_event("solver_change", f"Changing solver from {old_solver_name} to {solver_name}")

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
            error_msg = "Solver is not initialized."
            self._notify_error(error_msg)
            raise RuntimeError(error_msg)
        self._solver.setup_case()
        self._notify_event("case_setup", "Case setup completed.")

    def run_simulation(self):
        """
        Run the simulation.
        """
        if self._solver is None:
            error_msg = "Solver is not initialized."
            self._notify_error(error_msg)
            raise RuntimeError(error_msg)
        self._solver.run_simulation()
        self._notify_event("simulation_run", "Simulation started.")

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying solver instance.
        """
        if self._solver is None:
            error_msg = f"Cannot access attribute '{name}': Solver is not initialized."
            self._notify_error(error_msg)
            raise RuntimeError(error_msg)
        return getattr(self._solver, name)
