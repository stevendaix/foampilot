# system/fvSolutionFile.py
from foampilot.base.openFOAMFile import OpenFOAMFile

class FvSolutionFile(OpenFOAMFile):
    """
    A class representing the fvSolution file in OpenFOAM.

    The configuration adapts automatically depending on the parent Foam attributes:
        - simulation_type: "incompressible", "boussinesq", "compressible"
        - energy_variable (if compressible): "e", "h", "T"
    """

    def __init__(self, parent, solvers=None, SIMPLE=None, relaxationFactors=None):
        """
        Args:
            parent: Foam instance exposing attributes:
                - parent.simulation_type
                - parent.energy_variable (if compressible)
        """
        self.parent = parent

        # === Default solvers ===
        if solvers is None:
            solvers = {
                "p": {
                    "solver": "GAMG",
                    "tolerance": "1e-06",
                    "relTol": "0.1",
                    "smoother": "GaussSeidel",
                    "nPreSweeps": "0",
                    "nPostSweeps": "2",
                    "cacheAgglomeration": "on",
                    "agglomerator": "faceAreaPair",
                    "nCellsInCoarsestLevel": "10",
                    "mergeLevels": "1"
                },
                "U": {
                    "solver": "smoothSolver",
                    "smoother": "symGaussSeidel",
                    "tolerance": "1e-05",
                    "relTol": "0.1"
                },
                "k": {
                    "solver": "smoothSolver",
                    "smoother": "symGaussSeidel",
                    "tolerance": "1e-05",
                    "relTol": "0.1"
                },
                "epsilon": {
                    "solver": "smoothSolver",
                    "smoother": "symGaussSeidel",
                    "tolerance": "1e-05",
                    "relTol": "0.1"
                }
            }

            # === Extend depending on simulation type ===
            if self.parent.simulation_type == "boussinesq":
                solvers["T"] = {
                    "solver": "smoothSolver",
                    "smoother": "symGaussSeidel",
                    "tolerance": "1e-05",
                    "relTol": "0.1"
                }

            elif self.parent.simulation_type == "compressible":
                energy = getattr(self.parent, "energy_variable", "e")
                solvers[energy] = {
                    "solver": "smoothSolver",
                    "smoother": "symGaussSeidel",
                    "tolerance": "1e-05",
                    "relTol": "0.1"
                }
                solvers["rho"] = {
                    "solver": "smoothSolver",
                    "smoother": "symGaussSeidel",
                    "tolerance": "1e-06",
                    "relTol": "0.1"
                }

        # === SIMPLE algorithm ===
        if SIMPLE is None:
            SIMPLE = {
                "nNonOrthogonalCorrectors": "0",
                "residualControl": {
                    "p": "1e-2",
                    "U": "1e-4",
                    "k": "1e-4",
                    "epsilon": "1e-4"
                }
            }

            if self.parent.simulation_type == "boussinesq":
                SIMPLE["residualControl"]["T"] = "1e-4"

            elif self.parent.simulation_type == "compressible":
                energy = getattr(self.parent, "energy_variable", "e")
                SIMPLE["residualControl"][energy] = "1e-4"
                SIMPLE["residualControl"]["rho"] = "1e-4"

        # === Relaxation factors ===
        if relaxationFactors is None:
            relaxationFactors = {
                "fields": {"p": "0.3"},
                "equations": {"U": "0.7", "k": "0.7", "epsilon": "0.7"}
            }

            if self.parent.simulation_type == "boussinesq":
                relaxationFactors["equations"]["T"] = "0.7"

            elif self.parent.simulation_type == "compressible":
                energy = getattr(self.parent, "energy_variable", "e")
                relaxationFactors["equations"][energy] = "0.7"
                relaxationFactors["equations"]["rho"] = "0.7"

        # Call parent constructor
        super().__init__(
            object_name="fvSolution",
            solvers=solvers,
            SIMPLE=SIMPLE,
            relaxationFactors=relaxationFactors
        )

    def to_dict(self):
        """Export to OpenFOAM dictionary structure."""
        return {
            'solvers': self.solvers,
            'SIMPLE': self.SIMPLE,
            'relaxationFactors': self.relaxationFactors
        }

    @classmethod
    def from_dict(cls, config, parent):
        """Build instance from dict + Foam parent."""
        return cls(
            parent=parent,
            solvers=config.get('solvers', {}),
            SIMPLE=config.get('SIMPLE', {}),
            relaxationFactors=config.get('relaxationFactors', {})
        )