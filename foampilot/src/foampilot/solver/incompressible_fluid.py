from foampilot.system.SystemDirectory import SystemDirectory
from foampilot.mesh.BlockMeshFile import BlockMeshFile
from foampilot.boundaries.boundaries_dict import Boundary
from foampilot.constant.constantDirectory import ConstantDirectory
from pathlib import Path
import subprocess


class incompressibleFluid():
    """
    Represents an OpenFOAM case configured to run with the incompressibleFluid solver.

    This class organizes the main structure of an OpenFOAM case directory by 
    initializing key components: 'system', 'constant', and boundary condition
    dictionaries. It also provides functionality to update case-specific parameters
    and to execute the incompressibleFluid simulation.

    Attributes:
        case_path (Path): The path to the OpenFOAM case directory.
        system (SystemDirectory): Manages the 'system' folder and its files.
        constant (ConstantDirectory): Manages the 'constant' folder and its files.
        boundary (Boundary): Manages boundary condition dictionaries.
    """

    def __init__(self, path_case: str | Path):
        """
        Initialize a SimpleFoam case from the given directory path.

        Args:
            path_case (str or Path): Path to the OpenFOAM case directory.
        """
        self.case_path = Path(path_case)

        # Initialize subcomponents with the current instance as context
        self.system = SystemDirectory(self)
        self.constant = ConstantDirectory(self)
        self.boundary = Boundary(self)

    def update_case_specific_attributes(self):
        """
        Update or define attributes specific to the SimpleFoam case.

        This method can be extended to handle additional parameters or logic
        related to the simulation configuration (e.g., turbulence models, solvers).
        """
        print("Updating incompressibleFluid-specific attributes")

        
    def run_simulation(self, log_filename="log.incompressibleFluid"):
        """
        Run the incompressibleFluid solver in the case directory and write output to a log file.

        Args:
            log_filename (str): Name of the log file (default: "log.incompressibleFluid")
        """
        if not self.case_path.exists() or not self.case_path.is_dir():
            raise FileNotFoundError(f"Case path {self.case_path} does not exist or is not a directory.")

        log_path = self.case_path / log_filename

        try:
            print(f"Lancement de la simulation incompressibleFluid dans {self.case_path}")
            with open(log_path, "w", encoding="utf-8") as log_file:
                result = subprocess.run(
                    ["foamRun", "-solver", "incompressibleFluid"],
                    cwd=self.case_path,
                    text=True,
                    stdout=log_file,   # redirige stdout
                    stderr=subprocess.STDOUT,  # fusionne stderr dans stdout
                    check=True
                )
            print(f"✅ Simulation incompressibleFluid terminée avec succès. Log écrit dans {log_path}")
        except subprocess.CalledProcessError:
            print(f"❌ Simulation échouée. Vérifie le log dans {log_path}")
            raise RuntimeError(f"Simulation échouée (voir {log_path})")
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise
