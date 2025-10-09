from pathlib import Path
from typing import Optional, Callable, List, Any
from foampilot.solver.base_solver import BaseSolver
from foampilot.boundaries.boundaries_dict import Boundary
from foampilot.utilities.manageunits import Quantity

class Solver:
    """
    Generic solver class that adapts based on user-defined attributes.
    Supports transient incompressible fluids with PIMPLE, VoF, and solid simulations.
    Automatically configures fields and boundary conditions using CaseFieldsManager.
    """

    def __init__(self, case_path: str | Path):
        self.case_path = Path(case_path)
        self._solver: Optional[BaseSolver] = None
        self._compressible: bool = False
        self._with_gravity: bool = False
        self._is_vof: bool = False
        self._is_solid: bool = False
        self._energy_user: Optional[bool] = None
        self._transient: bool = False
        self._turbulence_model: str = "kEpsilon"
        self._error_handlers: List[Callable[[str], None]] = []
        self._event_handlers: List[Callable[[str, str], None]] = []

        # Initialize boundary with default turbulence model
        self.boundary = Boundary(self, fields_manager=None, turbulence_model=self._turbulence_model)

        # Update solver and fields
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
        """Update solver instance and configure fields based on current flags."""
        # Determine solver name
        if self._is_solid:
            solver_name = "solidDisplacement"
        elif self._is_vof:
            solver_name = "incompressibleVoF" if not self._compressible else "compressibleVoF"
        else:
            solver_name = "fluid" if self.energy_activated else "incompressibleFluid"

        old_solver_name = self._solver.solver_name if self._solver else "None"
        self._notify_event("solver_change", f"Changing solver from {old_solver_name} to {solver_name}")

        # Create solver with all flags
        self._solver = BaseSolver.create(
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

        # Update solver attributes
        self._solver.compressible = self._compressible
        self._solver.with_gravity = self._with_gravity
        self._solver.is_vof = self._is_vof
        self._solver.is_solid = self._is_solid
        self._solver.energy_activated = self.energy_activated
        self._solver.transient = self._transient
        self._solver.turbulence_model = self._turbulence_model

        # Update boundary with new solver and fields
        self.boundary = Boundary(
            self._solver,
            fields_manager=self._solver.fields_manager,
            turbulence_model=self._turbulence_model,
        )

        self._notify_event("solver_updated", f"Solver updated to {solver_name} with current flags.")

    # ---------- Boundary setup ----------
    def setup_boundaries(self):
        """Initialize boundary conditions based on current solver and fields."""
        try:
            self.boundary.initialize_boundary()
            self._notify_event("boundary_setup", "Boundary conditions initialized.")
        except Exception as e:
            self._notify_error(f"Failed to initialize boundaries: {str(e)}")
            raise

    # ---------- Public methods ----------
    def setup_case(self):
        """Set up the entire case directory structure and files."""
        if self._solver is None:
            msg = "Solver is not initialized."
            self._notify_error(msg)
            raise RuntimeError(msg)
        self._solver.setup_case()
        self._notify_event("case_setup", "Case setup completed.")

    def run_simulation(self):
        """Run the simulation with current configuration."""
        if self._solver is None:
            msg = "Solver is not initialized."
            self._notify_error(msg)
            raise RuntimeError(msg)
        self._solver.run_simulation()
        self._notify_event("simulation_run", "Simulation started.")

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the solver instance."""
        if self._solver is None:
            msg = f"Cannot access attribute '{name}': Solver is not initialized."
            self._notify_error(msg)
            raise RuntimeError(msg)
        return getattr(self._solver, name)
