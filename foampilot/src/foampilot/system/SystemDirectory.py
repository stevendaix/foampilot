import os
from pathlib import Path
from foampilot.system.controlDictFile import ControlDictFile
from foampilot.system.fvSchemesFile import FvSchemesFile
from foampilot.system.fvSolutionFile import FvSolutionFile
from foampilot.base.openFOAMFile import OpenFOAMFile
import subprocess

class SystemDirectory:
    """
    A class to manage the system directory of an OpenFOAM case.
    
    This class handles the creation, configuration, and management of all system files
    in an OpenFOAM case, including controlDict, fvSchemes, and fvSolution. It also provides
    methods to run OpenFOAM utilities like topoSet and createPatch.

    Attributes:
        parent: The parent case object.
        controlDict (ControlDictFile): The controlDict file handler.
        fvSchemes (FvSchemesFile): The fvSchemes file handler.
        fvSolution (FvSolutionFile): The fvSolution file handler.
        additional_files (dict): Dictionary of additional system files.
    """

    def __init__(self, parent):
        """
        Initialize the SystemDirectory with default file handlers.

        Args:
            parent: The parent case object that owns this system directory.
        """
        self.parent = parent 
        self.controlDict = ControlDictFile()
        self.fvSchemes = FvSchemesFile()
        self.fvSolution = FvSolutionFile()
        self.additional_files = {}
        self.fvSchemes = FvSchemesFile(parent=parent, fields_manager=getattr(parent, "fields_manager", None))

    # ... (le reste de la classe reste inchangé)


    def write(self):
        """
        Write all system files to the case directory.
        
        Creates the system directory if it doesn't exist and writes:
        - controlDict
        - fvSchemes
        - fvSolution
        - Any additional files that were added
        
        The files are written to <case_path>/system/ directory.
        """
        base_path = Path(self.parent.case_path) 
        system_path = Path(base_path) / 'system'
        system_path.mkdir(parents=True, exist_ok=True)

        # Write main system files
        self.controlDict.write(system_path / 'controlDict')
        self.fvSchemes.write(system_path / 'fvSchemes')
        self.fvSolution.write(system_path / 'fvSolution')

        # Write any additional files that were added
        for file_name, file in self.additional_files.items():
            file.write(system_path / file_name)

    def add_dict_file(self, file_name, file_content):
        """
        Add an additional file to the system directory.

        Args:
            file_name (str): The name of the file to add (e.g., 'transportProperties').
            file_content (dict): The content of the file as a dictionary.
        """
        self.additional_files[file_name] = OpenFOAMFile(object_name=file_name, **file_content)

    def to_dict(self):
        """
        Convert the system directory configuration to a dictionary.
        
        Returns:
            dict: A dictionary containing the configurations of:
                - controlDict
                - fvSchemes
                - fvSolution
        """
        return {
            'controlDict': self.controlDict.to_dict(),
            'fvSchemes': self.fvSchemes.to_dict(),
            'fvSolution': self.fvSolution.to_dict()
        }

    def from_dict(self, config):
        """
        Load system directory configuration from a dictionary.
        
        Args:
            config (dict): Dictionary containing configurations for:
                - controlDict
                - fvSchemes
                - fvSolution
        """
        self.controlDict = ControlDictFile.from_dict(config.get('controlDict', {}))
        self.fvSchemes = FvSchemesFile.from_dict(config.get('fvSchemes', {}))
        self.fvSolution = FvSolutionFile.from_dict(config.get('fvSolution', {}))

    def run_topoSet(self):
        """
        Execute the topoSet utility in the case directory.
        
        Runs the OpenFOAM topoSet command which handles cell sets and face sets.
        
        Raises:
            FileNotFoundError: If the case path does not exist.
            NotADirectoryError: If the case path is not a directory.
            RuntimeError: If the topoSet command fails to execute.
        """
        base_path = Path(self.parent.case_path)
        if not base_path.exists():
            raise FileNotFoundError(f"The case path '{base_path}' does not exist.")
        if not base_path.is_dir():
            raise NotADirectoryError(f"The case path '{base_path}' is not a directory.")

        try:
            print(f"Running 'topoSet' in: {base_path}")
            result = subprocess.run(
                ["topoSet"],
                cwd=base_path,
                text=True,
                capture_output=True,
                check=True
            )
            print("topoSet executed successfully.")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error executing topoSet: {e.stderr}")
            raise RuntimeError(f"topoSet failed with error: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    def run_createPatch(self, overwrite=True):
        """
        Execute the createPatch utility in the case directory.
        
        Runs the OpenFOAM createPatch command which handles patch creation and modification.
        
        Args:
            overwrite (bool): Whether to add the -overwrite flag (default: True).
            
        Raises:
            FileNotFoundError: If the case path does not exist.
            NotADirectoryError: If the case path is not a directory.
            RuntimeError: If the createPatch command fails to execute.
        """
        base_path = Path(self.parent.case_path)
        if not base_path.exists():
            raise FileNotFoundError(f"The case path '{base_path}' does not exist.")
        if not base_path.is_dir():
            raise NotADirectoryError(f"The case path '{base_path}' is not a directory.")

        cmd = ["createPatch"]
        if overwrite:
            cmd.append("-overwrite")

        try:
            print(f"Running '{' '.join(cmd)}' in: {base_path}")
            result = subprocess.run(
                cmd,
                cwd=base_path,
                text=True,
                capture_output=True,
                check=True
            )
            print("createPatch executed successfully.")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error executing createPatch: {e.stderr}")
            raise RuntimeError(f"createPatch failed with error: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    def write_functions_file(self, includes=None, filename="functions"):
        """
        Create a 'functions' file in the system directory with the given includes.

        Args:
            includes (list): List of function files to include (default: 
                             ["fieldAverage", "referencePressure", "runTimeControls"])
            filename (str): Name of the file to create (default: "functions")
            version (str): OpenFOAM version for the header (default: "12")
        """
        if includes is None:
            includes = ["fieldAverage", "referencePressure", "runTimeControls"]

        base_path = Path(self.parent.case_path)
        system_path = base_path / "system"
        system_path.mkdir(parents=True, exist_ok=True)

        path = system_path / filename

        header = f"""
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      functions;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""

        body = "\n".join([f'#include "{inc}"' for inc in includes])

        footer = """

// ************************************************************************* //
"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(header + body + footer)

        print(f"✅ Fichier {path} créé avec {len(includes)} includes.")
