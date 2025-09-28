from pathlib import Path
from typing import Optional, Callable, List
from foampilot.solver.base_solver import BaseSolver

class Solver:
    """
    Generic solver class that adapts based on user-defined attributes.
    Supports transient incompressible fluids with PIMPLE.
    """

    def __init__(self, case_path: str | Path):
        self.case_path = Path(case_path)
        self._solver: Optional[BaseSolver] = None
        self._compressible: bool = False
        self._with_gravity: bool = False
        self._is_vof: bool = False
        self._is_solid: bool = False
        self._energy_user: Optional[bool] = None
        self._transient: bool = False  # Nouveau flag
        self._error_handlers: List[Callable[[str], None]] = []
        self._event_handlers: List[Callable[[str, str], None]] = []

        self._update_solver()

    # ---------- Handlers ----------
    def add_error_handler(self, handler: Callable[[str], None]):
        self._error_handlers.append(handler)

    def add_event_handler(self, handler: Callable[[str, str], None]):
        self._event_handlers.append(handler)

    def _notify_error(self, message: str):
        print(f"⚠️ Error: {message}")
        for handler in self._error_handlers:
            handler(message)

    def _notify_event(self, event_type: str, message: str):
        print(f"🔔 Event: {event_type} - {message}")
        for handler in self._event_handlers:
            handler(event_type, message)

    # ---------- Properties ----------
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
            self._compressible = False
            self._with_gravity = False
            self._energy_user = False
        self._is_solid = value
        self._update_solver()

    @property
    def energy_activated(self) -> bool:
        if self._compressible or self._with_gravity:
            return True
        return self._energy_user if self._energy_user is not None else False

    @energy_activated.setter
    def energy_activated(self, value: bool):
        self._energy_user = value
        self._update_solver()

    @property
    def transient(self) -> bool:
        return self._transient

    @transient.setter
    def transient(self, value: bool):
        self._transient = value
        self._update_solver()

    # ---------- Solver selection ----------
    def _update_solver(self):
        if self._is_solid:
            solver_name = "solid"
        elif self._is_vof:
            solver_name = "incompressibleVoF" if not self._compressible else "compressibleVoF"
        else:
            if self.energy_activated:
                solver_name = "fluid"
            else:
                solver_name = "incompressibleFluid"

        old_solver_name = self._solver.solver_name if self._solver else "None"
        self._notify_event("solver_change", f"Changing solver from {old_solver_name} to {solver_name}")

        # Création du solver avec le flag transient
        self._solver = BaseSolver.create(
            self.case_path, 
            solver_name, 
            transient=self._transient
        )

        self._solver.compressible = self._compressible
        self._solver.with_gravity = self._with_gravity

    # ---------- Public methods ----------
    def setup_case(self):
        if self._solver is None:
            msg = "Solver is not initialized."
            self._notify_error(msg)
            raise RuntimeError(msg)
        self._solver.setup_case()
        self._notify_event("case_setup", "Case setup completed.")

    def run_simulation(self):
        if self._solver is None:
            msg = "Solver is not initialized."
            self._notify_error(msg)
            raise RuntimeError(msg)
        self._solver.run_simulation()
        self._notify_event("simulation_run", "Simulation started.")

    def __getattr__(self, name):
        if self._solver is None:
            msg = f"Cannot access attribute '{name}': Solver is not initialized."
            self._notify_error(msg)
            raise RuntimeError(msg)
        return getattr(self._solver, name)