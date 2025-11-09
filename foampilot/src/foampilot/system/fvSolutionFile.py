from typing import Dict, Optional, Any, List
from foampilot.base.openFOAMFile import OpenFOAMFile

class FvSolutionFile(OpenFOAMFile):
    """
    Représente le fichier fvSolution dans OpenFOAM, avec configuration automatique
    basée sur :
      - Les attributs du parent (simulation_type, algorithm, energy_variable, transient)
      - Les champs disponibles (via CaseFieldsManager)
    """

    DEFAULT_TOLERANCE: float = 1e-6
    DEFAULT_REL_TOL: float = 0.1
    DEFAULT_RELAXATION_FACTOR: float = 0.7

    def __init__(
        self,
        parent: Any,
        fields_manager: Optional[Any] = None,
        solvers: Optional[Dict[str, Dict[str, str]]] = None,
        SIMPLE: Optional[Dict[str, Any]] = None,
        PIMPLE: Optional[Dict[str, Any]] = None,
        relaxationFactors: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        self.parent = parent
        self.fields_manager = fields_manager
        self.solvers = self._init_solvers(solvers)

        algo = getattr(self.parent, "algorithm", "SIMPLE")
        transient = getattr(self.parent, "transient", False)

        if transient or algo == "PIMPLE":
            self.PIMPLE = self._init_pimple(PIMPLE)
            self.SIMPLE = None
        else:
            self.SIMPLE = self._init_simple(SIMPLE)
            self.PIMPLE = None

        self.relaxationFactors = self._init_relaxation_factors(relaxationFactors)

        # Configure solvers and relaxation based on available fields
        if self.fields_manager:
            self._configure_from_fields()

        super().__init__(
            object_name="fvSolution",
            solvers=self.solvers,
            SIMPLE=self.SIMPLE,
            PIMPLE=self.PIMPLE,
            relaxationFactors=self.relaxationFactors,
        )

    def _configure_from_fields(self) -> None:
        """Configure les solveurs et facteurs de relaxation en fonction des champs disponibles."""
        if not self.fields_manager:
            return

        field_names = self.fields_manager.get_field_names()
        sim_type = getattr(self.parent, "simulation_type", "incompressible")
        energy_active = getattr(self.parent, "energy_activated", False)
        energy_var = getattr(self.parent, "energy_variable", "e")
        algo = getattr(self.parent, "algorithm", "SIMPLE")
        transient = getattr(self.parent, "transient", False)

        # --- Solvers ---
        # Pression : p ou p_rgh
        pressure_field = "p_rgh" if "p_rgh" in field_names else "p"
        if pressure_field not in self.solvers:
            self.solvers[pressure_field] = self._default_solver(
                solver="GAMG",
                tolerance=self.DEFAULT_TOLERANCE,
                relTol=self.DEFAULT_REL_TOL,
                smoother="GaussSeidel",
                nPreSweeps="0",
                nPostSweeps="2",
                cacheAgglomeration="on",
                agglomerator="faceAreaPair",
                nCellsInCoarsestLevel="10",
                mergeLevels="1",
            )

        # Vitesse
        if "U" in field_names and "U" not in self.solvers:
            self.solvers["U"] = self._default_solver(
                solver="smoothSolver",
                smoother="symGaussSeidel",
                tolerance=1e-5,
                relTol=self.DEFAULT_REL_TOL,
            )

        # Turbulence (k, epsilon, omega, nut)
        for field in ["k", "epsilon", "omega", "nut"]:
            if field in field_names and field not in self.solvers:
                self.solvers[field] = self._default_solver(
                    solver="smoothSolver",
                    smoother="symGaussSeidel",
                    tolerance=1e-5,
                    relTol=self.DEFAULT_REL_TOL,
                )

        # Énergie (T, e, h)
        if energy_active and energy_var in field_names and energy_var not in self.solvers:
            self.solvers[energy_var] = self._default_solver(
                solver="smoothSolver",
                smoother="symGaussSeidel",
                tolerance=1e-5,
                relTol=self.DEFAULT_REL_TOL,
            )

        # VoF : alpha.water, alpha.air, etc.
        for field in field_names:
            if field.startswith("alpha.") and field not in self.solvers:
                self.solvers[field] = self._default_solver(
                    solver="smoothSolver",
                    smoother="symGaussSeidel",
                    tolerance=1e-5,
                    relTol=self.DEFAULT_REL_TOL,
                )

        # PIMPLE : champs finals
        if (algo == "PIMPLE" or transient) and (sim_type in ["compressible", "incompressible"] or energy_active):
            final_fields = []
            if pressure_field in self.solvers:
                final_fields.append(pressure_field)
            if "U" in self.solvers:
                final_fields.append("U")
            if energy_active and energy_var in self.solvers:
                final_fields.append(energy_var)
            if "k" in self.solvers:
                final_fields.append("k")
            if "epsilon" in self.solvers:
                final_fields.append("epsilon")

            for field in final_fields:
                final_key = f"{field}Final"
                if final_key not in self.solvers:
                    self.solvers[final_key] = {f"${field}": "", "relTol": "0"}

        # --- Relaxation factors ---
        if "fields" not in self.relaxationFactors:
            self.relaxationFactors["fields"] = {}
        if pressure_field in field_names:
            self.relaxationFactors["fields"][pressure_field] = str(self.DEFAULT_RELAXATION_FACTOR)

        if "equations" not in self.relaxationFactors:
            self.relaxationFactors["equations"] = {}

        for field in ["U", "k", "epsilon", "omega"]:
            if field in field_names:
                self.relaxationFactors["equations"][field] = str(self.DEFAULT_RELAXATION_FACTOR)

        if energy_active and energy_var in field_names:
            self.relaxationFactors["equations"][energy_var] = str(self.DEFAULT_RELAXATION_FACTOR)

        for field in field_names:
            if field.startswith("alpha."):
                self.relaxationFactors["equations"][field] = str(self.DEFAULT_RELAXATION_FACTOR)

    # -------------------------
    # Solvers (méthodes existantes)
    # -------------------------
    def _init_solvers(self, solvers: Optional[Dict[str, Dict[str, str]]]) -> Dict[str, Dict[str, str]]:
        if solvers is not None:
            return solvers.copy()

        solvers = {
            "p": self._default_solver(
                solver="GAMG",
                tolerance=self.DEFAULT_TOLERANCE,
                relTol=self.DEFAULT_REL_TOL,
                smoother="GaussSeidel",
                nPreSweeps="0",
                nPostSweeps="2",
                cacheAgglomeration="on",
                agglomerator="faceAreaPair",
                nCellsInCoarsestLevel="10",
                mergeLevels="1",
            ),
            "U": self._default_solver(
                solver="smoothSolver",
                smoother="symGaussSeidel",
                tolerance=1e-5,
                relTol=self.DEFAULT_REL_TOL,
            ),
            "k": self._default_solver(
                solver="smoothSolver",
                smoother="symGaussSeidel",
                tolerance=1e-5,
                relTol=self.DEFAULT_REL_TOL,
            ),
            "epsilon": self._default_solver(
                solver="smoothSolver",
                smoother="symGaussSeidel",
                tolerance=1e-5,
                relTol=self.DEFAULT_REL_TOL,
            ),
        }

        self._extend_solvers_for_simulation_type(solvers)
        return solvers

    def _default_solver(self, solver: str, **kwargs: Any) -> Dict[str, str]:
        return {
            "solver": solver,
            "tolerance": str(kwargs.get("tolerance", self.DEFAULT_TOLERANCE)),
            "relTol": str(kwargs.get("relTol", self.DEFAULT_REL_TOL)),
            **{k: str(v) for k, v in kwargs.items() if k not in ("tolerance", "relTol")},
        }

    def _extend_solvers_for_simulation_type(self, solvers: Dict[str, Dict[str, str]]) -> None:
        sim_type = getattr(self.parent, "simulation_type", "incompressible")
        algo = getattr(self.parent, "algorithm", "SIMPLE")
        transient = getattr(self.parent, "transient", False)
        energy_active = getattr(self.parent, "energy_activated", False)
        energy_var = getattr(self.parent, "energy_variable", "e")

        if sim_type == "boussinesq":
            solvers["T"] = self._default_solver(
                solver="smoothSolver",
                smoother="symGaussSeidel",
                tolerance=1e-5,
                relTol=self.DEFAULT_REL_TOL,
            )

        if sim_type == "compressible" or energy_active:
            solvers[energy_var] = self._default_solver(
                solver="smoothSolver",
                smoother="symGaussSeidel",
                tolerance=1e-5,
                relTol=self.DEFAULT_REL_TOL,
            )
            if sim_type == "compressible":
                solvers["rho"] = self._default_solver(
                    solver="smoothSolver",
                    smoother="symGaussSeidel",
                    tolerance=1e-6,
                    relTol=self.DEFAULT_REL_TOL,
                )

        if (algo == "PIMPLE" or transient) and (sim_type in ["compressible", "incompressible"] or energy_active):
            solvers["pFinal"] = {"$p": "", "relTol": "0"}
            solvers["UFinal"] = {"$U": "", "relTol": "0"}
            if sim_type == "compressible" or energy_active:
                solvers[f"{energy_var}Final"] = {"$" + energy_var: "", "relTol": "0"}
            solvers["kFinal"] = {"$k": "", "relTol": "0"}
            solvers["epsilonFinal"] = {"$epsilon": "", "relTol": "0"}

    # -------------------------
    # SIMPLE (méthode existante)
    # -------------------------
    def _init_simple(self, SIMPLE: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if SIMPLE is not None:
            return SIMPLE.copy()

        SIMPLE = {
            "nNonOrthogonalCorrectors": "0",
            "residualControl": {
                "p": "1e-2",
                "U": "1e-4",
                "k": "1e-4",
                "epsilon": "1e-4",
            },
        }

        sim_type = getattr(self.parent, "simulation_type", "incompressible")
        energy_active = getattr(self.parent, "energy_activated", False)
        energy_var = getattr(self.parent, "energy_variable", "e")

        if sim_type == "boussinesq":
            SIMPLE["residualControl"]["T"] = "1e-4"
        elif sim_type == "compressible" or energy_active:
            SIMPLE["residualControl"][energy_var] = "1e-4"
            if sim_type == "compressible":
                SIMPLE["residualControl"]["rho"] = "1e-4"

        return SIMPLE

    # -------------------------
    # PIMPLE (méthode existante)
    # -------------------------
    def _init_pimple(self, PIMPLE: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if PIMPLE is not None:
            return PIMPLE.copy()
        return {
            "momentumPredictor": "yes",
            "nOuterCorrectors": "1",
            "nCorrectors": "2",
            "nNonOrthogonalCorrectors": "0",
            "pRefCell": "0",
            "pRefValue": "0",
        }

    # -------------------------
    # Relaxation factors (méthode existante)
    # -------------------------
    def _init_relaxation_factors(self, relaxationFactors: Optional[Dict[str, Dict[str, str]]]) -> Dict[str, Dict[str, str]]:
        if relaxationFactors is not None:
            return relaxationFactors.copy()

        relaxationFactors = {
            "fields": {"p": str(self.DEFAULT_RELAXATION_FACTOR)},
            "equations": {
                "U": str(self.DEFAULT_RELAXATION_FACTOR),
                "k": str(self.DEFAULT_RELAXATION_FACTOR),
                "epsilon": str(self.DEFAULT_RELAXATION_FACTOR),
            },
        }

        sim_type = getattr(self.parent, "simulation_type", "incompressible")
        energy_active = getattr(self.parent, "energy_activated", False)
        energy_var = getattr(self.parent, "energy_variable", "e")

        if sim_type == "boussinesq":
            relaxationFactors["equations"]["T"] = str(self.DEFAULT_RELAXATION_FACTOR)
        elif sim_type == "compressible" or energy_active:
            relaxationFactors["equations"][energy_var] = str(self.DEFAULT_RELAXATION_FACTOR)
            if sim_type == "compressible":
                relaxationFactors["equations"]["rho"] = str(self.DEFAULT_RELAXATION_FACTOR)

        return relaxationFactors

    # -------------------------
    # Export / Import (méthodes existantes)
    # -------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = {"solvers": self.solvers, "relaxationFactors": self.relaxationFactors}
        if self.SIMPLE is not None:
            d["SIMPLE"] = self.SIMPLE
        if self.PIMPLE is not None:
            d["PIMPLE"] = self.PIMPLE
        return d

    def write(self, filepath):
        """Write the fvSolution file."""
        self.write_file(filepath)

    @classmethod
    def from_dict(cls, config: Dict[str, Dict[str, str]], parent: Any) -> "FvSolutionFile":
        return cls(
            parent=parent,
            solvers=config.get("solvers"),
            SIMPLE=config.get("SIMPLE"),
            PIMPLE=config.get("PIMPLE"),
            relaxationFactors=config.get("relaxationFactors"),
        )