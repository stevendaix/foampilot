# system/fvSolutionFile.py
from foampilot.base.openFOAMFile import OpenFOAMFile

class FvSolutionFile(OpenFOAMFile):
    """
    A class representing the fvSolution file in OpenFOAM.
    
    This class handles the creation and manipulation of the fvSolution file which defines
    the solution algorithms and solver controls for an OpenFOAM simulation. It inherits
    from OpenFOAMFile and provides specific functionality for solver configuration.

    Attributes:
        solvers (dict): Dictionary containing solver settings for each field.
        SIMPLE (dict): SIMPLE algorithm control parameters.
        relaxationFactors (dict): Under-relaxation factors for fields and equations.
    """
    
    def __init__(self, solvers=None, SIMPLE=None, relaxationFactors=None):
        """
        Initialize the FvSolutionFile with solver configuration.

        Args:
            solvers: Dictionary of solver configurations for each field (default: {
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
                },
                "R": {
                    "solver": "smoothSolver",
                    "smoother": "symGaussSeidel",
                    "tolerance": "1e-05",
                    "relTol": "0.1"
                },
                "nuTilda": {
                    "solver": "smoothSolver",
                    "smoother": "symGaussSeidel",
                    "tolerance": "1e-05",
                    "relTol": "0.1"
                }
            }).
            SIMPLE: SIMPLE algorithm settings (default: {
                "nNonOrthogonalCorrectors": "0",
                "residualControl": {
                    "p": "1e-2",
                    "U": "1e-4",
                    "k": "1e-4",
                    "epsilon": "1e-4"
                }
            }).
            relaxationFactors: Under-relaxation factors (default: {
                "fields": {"p": "0.3"},
                "equations": {"U": "0.7", "k": "0.7", "epsilon": "0.7"}
            }).
        """
        # Initialize solver configurations with default values if None provided
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
                },
                "R": {
                    "solver": "smoothSolver",
                    "smoother": "symGaussSeidel",
                    "tolerance": "1e-05",
                    "relTol": "0.1"
                },
                "nuTilda": {
                    "solver": "smoothSolver",
                    "smoother": "symGaussSeidel",
                    "tolerance": "1e-05",
                    "relTol": "0.1"
                }
            }

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

        if relaxationFactors is None:
            relaxationFactors = {
                "fields": {"p": "0.3"},
                "equations": {"U": "0.7", "k": "0.7", "epsilon": "0.7"}
            }

        # Call parent class constructor with all configurations
        super().__init__(
            object_name="fvSolution",
            solvers=solvers,
            SIMPLE=SIMPLE,
            relaxationFactors=relaxationFactors
        )

    def to_dict(self):
        """
        Convert the solver configuration to a dictionary.
        
        Returns:
            dict: A dictionary containing all solver settings with their current configuration.
                  The structure matches the OpenFOAM fvSolution format with three main sections:
                  - solvers: Per-field solver configurations
                  - SIMPLE: Algorithm controls
                  - relaxationFactors: Under-relaxation parameters
        """
        return {
            'solvers': self.solvers,
            'SIMPLE': self.SIMPLE,
            'relaxationFactors': self.relaxationFactors
        }

    @classmethod
    def from_dict(cls, config):
        """
        Create a FvSolutionFile instance from a configuration dictionary.
        
        This class method allows creating a FvSolutionFile instance by providing a dictionary
        with solver configurations. Missing sections will use empty dictionaries.

        Args:
            config (dict): Dictionary containing solver configuration. Possible keys:
                - solvers: Dictionary of solver configurations
                - SIMPLE: SIMPLE algorithm settings
                - relaxationFactors: Under-relaxation factors
                
        Returns:
            FvSolutionFile: A new instance initialized with the provided configurations.
        """
        # Get each configuration section from dict or use empty dict if not provided
        solvers = config.get('solvers', {})
        SIMPLE = config.get('SIMPLE', {})
        relaxationFactors = config.get('relaxationFactors', {})

        # Create and return new instance
        return cls(
            solvers=solvers,
            SIMPLE=SIMPLE,
            relaxationFactors=relaxationFactors
        )