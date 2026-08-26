import os
import logging
from pathlib import Path
from foampilot.system.controlDictFile import ControlDictFile
from foampilot.system.fvSchemesFile import FvSchemesFile
from foampilot.system.fvSolutionFile import FvSolutionFile
from foampilot.base.openFOAMFile import OpenFOAMFile
import subprocess
from foampilot.system.decomposeParDictFile import DecomposeParDictFile

logger = logging.getLogger(__name__)

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
        self.controlDict = ControlDictFile(parent=parent)
      
        self.additional_files = {}
        self.fvSchemes = FvSchemesFile(parent=parent, fields_manager=getattr(parent, "fields_manager", None))
      
        self.fvSolution = FvSolutionFile(parent=parent, fields_manager=getattr(parent, "fields_manager", None))
      
        self.decomposeParDict = None 
        




    def import_reference_file(self, source_path: str | Path, filename: str | None = None) -> Path:
        """Import a complete OpenFOAM system dictionary without lossy parsing."""
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        target_name = filename or source.name
        target = Path(self.parent.case_path) / "system" / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        return target

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

        # Write functions file with scalarTransport when energy is activated
        if getattr(self.parent, "energy_activated", False):
            self._write_functions_file(system_path)

        # Write decomposeParDict if created
        if self.decomposeParDict is not None:
           self.decomposeParDict.write(system_path / "decomposeParDict")

        # Write any additional files that were added
        for file_name, file in self.additional_files.items():
            file.write(system_path / file_name)
        
        return system_path

    def _write_functions_file(self, system_path: Path) -> None:
        """Write the ``system/functions`` file with a ``scalarTransport``
        functionObject so that the incompressible flow solver also evolves
        the temperature field ``T`` as a passive scalar.

        This is the OpenFOAM 13 mechanism for adding temperature transport
        to the ``incompressibleFluid`` solver module (which is otherwise
        isothermal).  The ``scalarTransport`` functionObject reads the
        volumetric flux ``phi`` produced by the flow solver at each time
        step and solves the advection-diffusion equation for ``T``.
        """
        parent = self.parent
        energy_var = getattr(parent, "energy_variable", "T")
        field_names = getattr(parent, "fields_manager", None)
        field_list = field_names.get_field_names() if field_names else []

        if energy_var not in field_list:
            return

        # Determine thermal diffusivity: use nu/Pr if Pr is available
        constant_dir = getattr(parent, "constant", None)
        transport_props = getattr(constant_dir, "_transportProperties", None) if constant_dir else None
        if transport_props is not None:
            transport_props._configure_attributes()
        pr = 0.85
        if transport_props is not None:
            try:
                pr_raw = transport_props.attributes.get("Pr", None)
                if pr_raw is not None:
                    pr = float(pr_raw)
            except (TypeError, ValueError):
                pass

        nu_val = 1e-5
        if transport_props is not None:
            try:
                nu_raw = transport_props.attributes.get("nu", None)
                if nu_raw is not None:
                    nu_val = float(nu_raw)
            except (TypeError, ValueError):
                pass

        D = nu_val / pr

        content = (
            "FoamFile\n"
            "{\n"
            "    format      ascii;\n"
            "    class       dictionary;\n"
            "    location    \"system\";\n"
            "    object      functions;\n"
            "}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
            "\n"
            f'#includeFunc scalarTransport({energy_var}, diffusivity=constant, D = {D:g})\n'
            "\n"
            "// ************************************************************************* //\n"
        )
        functions_path = system_path / "functions"
        functions_path.write_text(content)
        logger.info("Wrote functions file: %s", functions_path)

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


    def ensure_decomposeParDict(self, nb_proc: int):
        """
        Create a decomposeParDict file handler if not present.
        """
        if self.decomposeParDict is None:
            self.decomposeParDict = DecomposeParDictFile(parent=self.parent, nb_proc=nb_proc)
        else:
            self.decomposeParDict.set_nb_proc(nb_proc)

    def write_functions_file(self, system_path: Path, *, rigid_body: bool = False) -> None:
        """Write the ``system/functions`` file.

        Parameters
        ----------
        system_path:
            Target directory for the ``functions`` file.
        rigid_body:
            If ``True``, append a ``rigidBodyForces`` functionObject
            using ``librigidBodyForces.so`` for the ``hull`` body.
        """
        base_content = ""
        energy_var = getattr(self.parent, "energy_variable", "T")
        field_names = getattr(self.parent, "fields_manager", None)
        field_list = field_names.get_field_names() if field_names else []

        if energy_var in field_list:
            constant_dir = getattr(self.parent, "constant", None)
            transport_props = getattr(constant_dir, "_transportProperties", None) if constant_dir else None
            if transport_props is not None:
                transport_props._configure_attributes()
            pr = 0.85
            nu_val = 1e-5
            if transport_props is not None:
                try:
                    pr_raw = transport_props.attributes.get("Pr", None)
                    if pr_raw is not None:
                        pr = float(pr_raw)
                except (TypeError, ValueError):
                    pass
                try:
                    nu_raw = transport_props.attributes.get("nu", None)
                    if nu_raw is not None:
                        nu_val = float(nu_raw)
                except (TypeError, ValueError):
                    pass
            D = nu_val / pr
            base_content += f'#includeFunc scalarTransport({energy_var}, diffusivity=constant, D = {D:g})\n'

        if rigid_body:
            base_content += (
                "rigidBodyForces\n"
                "{\n"
                "    type            rigidBodyForces;\n"
                "    libs            (\"librigidBodyForces.so\");\n"
                "    body            hull;\n"
                "    patches         (hull);\n"
                "    log             on;\n"
                "    writeControl    timeStep;\n"
                "    writeInterval   1;\n"
                "}\n"
            )

        if not base_content:
            return

        content = (
            "FoamFile\n"
            "{\n"
            "    format      ascii;\n"
            "    class       dictionary;\n"
            "    location    \"system\";\n"
            "    object      functions;\n"
            "}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
            "\n"
        )
        content += base_content
        content += "\n// ************************************************************************* //\n"
        functions_path = system_path / "functions"
        functions_path.write_text(content)
        logger.info("Wrote functions file: %s", functions_path)

    def write_set_fields_dict(self, system_path: Path, *, zones: list | None = None) -> None:
        """Write a minimal ``system/setFieldsDict``.

        Parameters
        ----------
        system_path:
            Target directory.
        zones:
            Optional list of box zones. Each item is a dict with:
            ``name``, ``min``, ``max``, ``field``, ``value``.
        """
        content = (
            "FoamFile\n"
            "{\n"
            "    format      ascii;\n"
            "    class       dictionary;\n"
            "    location    \"system\";\n"
            "    object      setFieldsDict;\n"
            "}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
            "\n"
            "defaultFieldValues\n"
            "(\n"
            ");\n"
            "\n"
        )

        if zones:
            content += "regions\n"
            content += "(\n"
            for zone in zones:
                content += (
                    f"    {zone['name']}\n"
                    "    {\n"
                    f"        box ({zone['min'][0]} {zone['min'][1]} {zone['min'][2]}) ({zone['max'][0]} {zone['max'][1]} {zone['max'][2]});\n"
                    f"        field {zone['field']};\n"
                    f"        value {zone['value']};\n"
                    "    }\n"
                )
            content += ");\n"

        content += "\n// ************************************************************************* //\n"
        set_fields_path = system_path / "setFieldsDict"
        set_fields_path.write_text(content)
        logger.info("Wrote setFieldsDict: %s", set_fields_path)

    def write_refine_mesh_dict(self, system_path: Path, *, zones: list | None = None) -> None:
        """Write a minimal ``system/refineMeshDict``.

        Parameters
        ----------
        system_path:
            Target directory.
        zones:
            Optional list of box refinement zones. Each item is a dict
            with ``name``, ``min``, ``max``, and ``level``.
        """
        content = (
            "FoamFile\n"
            "{\n"
            "    format      ascii;\n"
            "    class       dictionary;\n"
            "    location    \"system\";\n"
            "    object      refineMeshDict;\n"
            "}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
            "\n"
            "refineMesh  true;\n"
            "\n"
        )

        if zones:
            content += "locations\n"
            content += "(\n"
            for zone in zones:
                content += (
                    f"    {zone['name']}\n"
                    "    {\n"
                    f"        box ({zone['min'][0]} {zone['min'][1]} {zone['min'][2]}) ({zone['max'][0]} {zone['max'][1]} {zone['max'][2]});\n"
                    f"        level {zone['level']};\n"
                    "    }\n"
                )
            content += ");\n"

        content += "\n// ************************************************************************* //\n"
        refine_path = system_path / "refineMeshDict"
        refine_path.write_text(content)
        logger.info("Wrote refineMeshDict: %s", refine_path)

    def write_mesh_quality_dict(self, system_path: Path, *, quality: dict | None = None) -> None:
        """Write ``system/meshQualityDict``.

        Parameters
        ----------
        system_path:
            Target directory.
        quality:
            Optional dict overriding quality controls.
        """
        default_quality = {
            "maxNonOrtho": 75,
            "maxBoundarySkewness": 20,
            "maxInternalSkewness": 4,
            "maxConcave": 80,
            "minVol": 1e-13,
            "minTetQuality": 1e-15,
            "minArea": -1,
            "minTwist": 0.02,
            "minDeterminant": 0.001,
            "minFaceWeight": 0.05,
            "minVolRatio": 0.01,
            "minTriangleTwist": -1,
            "minFlatness": 0.5,
            "nSmoothScale": 4,
            "errorReduction": 0.75,
            "debug": 0,
            "writeFlags": "()",
            "mergeTolerance": 1e-6,
        }
        if quality:
            default_quality.update(quality)

        content = (
            "FoamFile\n"
            "{\n"
            "    format      ascii;\n"
            "    class       dictionary;\n"
            "    location    \"system\";\n"
            "    object      meshQualityDict;\n"
            "}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
            "\n"
            "meshQualityControls\n"
            "{\n"
        )
        for key, value in default_quality.items():
            if isinstance(value, str):
                content += f"    {key:<20} {value};\n"
            elif isinstance(value, bool):
                content += f"    {key:<20} {str(value).lower()};\n"
            else:
                content += f"    {key:<20} {value};\n"
        content += "}\n"
        content += "\n// ************************************************************************* //\n"

        quality_path = system_path / "meshQualityDict"
        quality_path.write_text(content)
        logger.info("Wrote meshQualityDict: %s", quality_path)



    def rename_dictionary_entries(self, dictionary: str | Path, renames: dict[str, str]) -> Path:
        """Rename dictionary entries through the OpenFOAM ``foamDictionary`` utility."""
        dictionary_path = Path(dictionary)
        if not dictionary_path.is_absolute():
            dictionary_path = Path(self.parent.case_path) / dictionary_path
        if not dictionary_path.is_file():
            raise FileNotFoundError(dictionary_path)
        mapping = ", ".join(f"{old}={new}" for old, new in renames.items())
        self.run_utility(
            "foamDictionary",
            [str(dictionary_path), "-rename", mapping],
            log_filename=f"log.foamDictionary.rename.{dictionary_path.name}",
        )
        return dictionary_path

    def remove_dictionary_entries(self, dictionary: str | Path, entries: list[str]) -> Path:
        """Remove entries from an OpenFOAM dictionary through ``foamDictionary``."""
        dictionary_path = Path(dictionary)
        if not dictionary_path.is_absolute():
            dictionary_path = Path(self.parent.case_path) / dictionary_path
        if not dictionary_path.is_file():
            raise FileNotFoundError(dictionary_path)
        for entry in entries:
            self.run_utility(
                "foamDictionary",
                [str(dictionary_path), "-remove", "-entry", entry],
                log_filename=f"log.foamDictionary.remove.{dictionary_path.name}.{entry.replace('/', '_')}",
            )
        return dictionary_path

    def update_dictionary_entries(self, dictionary: str | Path, entries: dict[str, str]) -> Path:
        """Update OpenFOAM dictionary entries through ``foamDictionary``."""
        dictionary_path = Path(dictionary)
        if not dictionary_path.is_absolute():
            dictionary_path = Path(self.parent.case_path) / dictionary_path
        if not dictionary_path.is_file():
            raise FileNotFoundError(dictionary_path)
        for entry, value in entries.items():
            self.run_utility(
                "foamDictionary",
                [str(dictionary_path), "-entry", entry, "-set", str(value)],
                log_filename=f"log.foamDictionary.{dictionary_path.name}.{entry.replace('/', '_')}",
            )
        return dictionary_path

    def replace_file_text(self, file: str | Path, old: str, new: str, count: int = -1) -> Path:
        """Replace text in a case file through a FoamPilot-managed API.

        This is intended for deterministic post-processing of files generated
        by OpenFOAM utilities when a reference tutorial applies a small text
        transformation that is not safely expressible through a dictionary
        parser.
        """
        path = Path(file)
        if not path.is_absolute():
            path = Path(self.parent.case_path) / path
        if not path.is_file():
            raise FileNotFoundError(path)
        content = path.read_text(encoding="utf-8")
        if old not in content:
            raise ValueError(f"Text not found in {path}: {old!r}")
        path.write_text(content.replace(old, new, count), encoding="utf-8")
        return path

    def run_utility(self, utility: str, args=None, log_filename=None) -> Path:
        """Run an OpenFOAM utility in the case directory through FoamPilot.

        ``args`` contains only utility arguments; the case directory is selected
        automatically and stdout/stderr are persisted in a deterministic log.
        """
        base_path = Path(self.parent.case_path)
        if not base_path.is_dir():
            raise NotADirectoryError(f"The case path '{base_path}' is not a directory.")
        cmd = [utility, *(str(arg) for arg in (args or []))]
        log_path = base_path / (log_filename or f"log.{utility}")
        try:
            result = subprocess.run(cmd, cwd=base_path, text=True,
                                    capture_output=True, check=True)
        except subprocess.CalledProcessError as exc:
            log_path.write_text((exc.stdout or "") + "\n" + (exc.stderr or ""))
            raise RuntimeError(f"{utility} failed: {exc.stderr}") from exc
        log_path.write_text(result.stdout + "\n" + result.stderr)
        return log_path

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
            logger.info(f"Running 'topoSet' in: {base_path}")
            result = subprocess.run(
                ["topoSet"],
                cwd=base_path,
                text=True,
                capture_output=True,
                check=True
            )
            logger.info("topoSet executed successfully.")
            logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error executing topoSet: {e.stderr}")
            raise RuntimeError(f"topoSet failed with error: {e.stderr}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
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
            logger.info(f"Running '{' '.join(cmd)}' in: {base_path}")
            result = subprocess.run(
                cmd,
                cwd=base_path,
                text=True,
                capture_output=True,
                check=True
            )
            logger.info("createPatch executed successfully.")
            logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error executing createPatch: {e.stderr}")
            raise RuntimeError(f"createPatch failed with error: {e.stderr}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
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

        logger.info(f"Fichier {path} créé avec {len(includes)} includes.")