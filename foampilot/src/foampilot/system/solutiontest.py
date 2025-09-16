from typing import Dict, Optional, Any
from foampilot.base.openFOAMFile import OpenFOAMFile

class FvSolutionFile(OpenFOAMFile):
    """
    Represents the fvSolution file in OpenFOAM, with automatic configuration
    based on parent Foam attributes (simulation_type, algorithm, energy_variable).
    """

    DEFAULT_TOLERANCE: float = 1e-6
    DEFAULT_REL_TOL: float = 0.1
    DEFAULT_RELAXATION_FACTOR: float = 0.7

    def __init__(
        self,
        parent: Any,
        solvers: Optional[Dict[str, Dict[str, str]]] = None,
        SIMPLE: Optional[Dict[str, Any]] = None,
        PIMPLE: Optional[Dict[str, Any]] = None,
        relaxationFactors: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        self.parent = parent
        self.solvers = self._init_solvers(solvers)

        if getattr(self.parent, "algorithm", "SIMPLE") == "SIMPLE":
            self.SIMPLE = self._init_simple(SIMPLE)
            self.PIMPLE = None
        else:
            self.PIMPLE = self._init_pimple(PIMPLE)
            self.SIMPLE = None

        self.relaxationFactors = self._init_relaxation_factors(relaxationFactors)

        super().__init__(
            object_name="fvSolution",
            solvers=self.solvers,
            SIMPLE=self.SIMPLE,
            PIMPLE=self.PIMPLE,
            relaxationFactors=self.relaxationFactors,
        )

    # -------------------------
    # Solvers
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
        if self.parent.simulation_type == "boussinesq":
            solvers["T"] = self._default_solver(
                solver="smoothSolver",
                smoother="symGaussSeidel",
                tolerance=1e-5,
                relTol=self.DEFAULT_REL_TOL,
            )

        elif self.parent.simulation_type == "compressible":
            energy = getattr(self.parent, "energy_variable", "e")
            solvers[energy] = self._default_solver(
                solver="smoothSolver",
                smoother="symGaussSeidel",
                tolerance=1e-5,
                relTol=self.DEFAULT_REL_TOL,
            )
            solvers["rho"] = self._default_solver(
                solver="smoothSolver",
                smoother="symGaussSeidel",
                tolerance=1e-6,
                relTol=self.DEFAULT_REL_TOL,
            )
            # Spécifique compressible / PIMPLE : p_rgh
            if getattr(self.parent, "algorithm", "SIMPLE") == "PIMPLE":
                solvers["p_rgh"] = {
                    "solver": "PCG",
                    "preconditioner": "DIC",
                    "tolerance": "1e-8",
                    "relTol": "0.01",
                }
                solvers["p_rghFinal"] = {
                    "$p_rgh": "",
                    "relTol": "0",
                }
                solvers["UFinal"] = {"$U": "", "relTol": "0"}
                solvers[f"{energy}Final"] = {"$" + energy: "", "relTol": "0"}
                solvers["kFinal"] = {"$k": "", "relTol": "0"}
                solvers["epsilonFinal"] = {"$epsilon": "", "relTol": "0"}

    # -------------------------
    # SIMPLE
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
        if self.parent.simulation_type == "boussinesq":
            SIMPLE["residualControl"]["T"] = "1e-4"
        elif self.parent.simulation_type == "compressible":
            energy = getattr(self.parent, "energy_variable", "e")
            SIMPLE["residualControl"][energy] = "1e-4"
            SIMPLE["residualControl"]["rho"] = "1e-4"
        return SIMPLE

    # -------------------------
    # PIMPLE
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
    # Relaxation factors
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
        if self.parent.simulation_type == "boussinesq":
            relaxationFactors["equations"]["T"] = str(self.DEFAULT_RELAXATION_FACTOR)
        elif self.parent.simulation_type == "compressible":
            energy = getattr(self.parent, "energy_variable", "e")
            relaxationFactors["equations"][energy] = str(self.DEFAULT_RELAXATION_FACTOR)
            relaxationFactors["equations"]["rho"] = str(self.DEFAULT_RELAXATION_FACTOR)
        return relaxationFactors

    # -------------------------
    # Export / Import
    # -------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = {"solvers": self.solvers, "relaxationFactors": self.relaxationFactors}
        if self.SIMPLE is not None:
            d["SIMPLE"] = self.SIMPLE
        if self.PIMPLE is not None:
            d["PIMPLE"] = self.PIMPLE
        return d

    @classmethod
    def from_dict(cls, config: Dict[str, Any], parent: Any) -> "FvSolutionFile":
        return cls(
            parent=parent,
            solvers=config.get("solvers"),
            SIMPLE=config.get("SIMPLE"),
            PIMPLE=config.get("PIMPLE"),
            relaxationFactors=config.get("relaxationFactors"),
        )