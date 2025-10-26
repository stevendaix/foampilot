from pathlib import Path
from typing import Optional, Callable, List, Any
from foampilot.solver.base_solver import BaseSolver
from foampilot.boundaries.boundaries_dict import Boundary

class Solver:
    """Generic solver manager with automatic solver selection."""

    def __init__(self, case_path: str | Path):
        self.case_path = Path(case_path)
        self._solver: Optional[BaseSolver] = None

        # Flags
        self._compressible = False
        self._with_gravity = False
        self._is_vof = False
        self._is_solid = False
        self._energy_user: Optional[bool] = None
        self._transient = False
        self._turbulence_model = "kEpsilon"

        # Handlers
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
        for h in self._error_handlers:
            h(message)

    def _notify_event(self, event_type: str, message: str):
        print(f"🔔 Event: {event_type} - {message}")
        for h in self._event_handlers:
            h(event_type, message)

    # ---------- Properties ----------
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
        if value:
            self._compressible = False
            self._with_gravity = False
            self._energy_user = False
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

    @property
    def turbulence_model(self) -> str:
        return self._turbulence_model

    @turbulence_model.setter
    def turbulence_model(self, value: str):
        self._turbulence_model = value
        if self.boundary:
            self.boundary.turbulence_model = value
        self._update_solver()

    # ---------- Solver selection ----------
    def _update_solver(self):
        if self._is_solid:
            solver_name = "solidDisplacement"
        elif self._is_vof:
            solver_name = "incompressibleVoF" if not self._compressible else "compressibleVoF"
        else:
            solver_name = "fluid" if self.energy_activated else "incompressibleFluid"

        old_solver = self._solver.solver_name if self._solver else "None"
        self._notify_event("solver_change", f"Changing solver from {old_solver} to {solver_name}")

        self._solver = BaseSolver(
            case_path=self.case_path,
            solver_name=solver_name,
            compressible=self._compressible,
            with_gravity=self._with_gravity,
            is_vof=self._is_vof,
            is_solid=self._is_solid,
            energy_activated=self.energy_activated,
            transient=self._transient,
            turbulence_model=self._turbulence_model,
        )

        self.boundary = Boundary(self._solver, fields_manager=self._solver.fields_manager)

    # ---------- Delegate methods ----------
    def setup_case(self):
        self._solver.setup_case()

    def run_simulation(self):
        self._solver.run_simulation()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._solver, name)
