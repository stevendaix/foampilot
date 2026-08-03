import logging
from pathlib import Path
from foampilot.base.openFOAMFile import OpenFOAMFile
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class ControlDictFile(OpenFOAMFile):
    """
    Class representing the controlDict file in OpenFOAM.

    If a parent Solver is provided, the OpenFOAM solver application
    ('application') is automatically retrieved from the parent.
    """

    def __init__(
        self,
        parent: Optional[Any] = None,
        application: Optional[str] = None,
        startFrom: str = "startTime",
        startTime: float = 0,
        stopAt: str = "endTime",
        endTime: float = 5000,
        deltaT: float = 1,
        writeControl: str = "timeStep",
        writeInterval: float = 100,
        purgeWrite: int = 10,
        writeFormat: str = "ascii",
        writePrecision: int = 6,
        writeCompression: str = "off",
        timeFormat: str = "general",
        timePrecision: int = 6,
        runTimeModifiable: bool = True,
        libs: Optional[List[str]] = None,
        adaptiveTimeStep: Optional[Dict[str, Any]] = None,
        functions: Optional[List[str]] = None,
        region_solvers: Optional[Dict[str, str]] = None,
        sub_solver: Optional[str] = None,

 ):

        # Retrieve solver name from parent if application not explicitly provided
        if application is None and parent is not None:
            # Assume parent has a property `solver_name` (from Solver class)
            application = getattr(parent, "solver_name", "incompressibleFluid")

        if libs is not None:
            if isinstance(libs, str):
                libs = [libs]
            elif not all(isinstance(lib, str) for lib in libs):
                raise TypeError("libs must be a list of strings")

        self.libs = libs or []
        self.adaptiveTimeStep = adaptiveTimeStep or {}
        self.functions = functions or []
        self.region_solvers = region_solvers or {}
        self.sub_solver = sub_solver

        # Call parent constructor with all parameters
        super().__init__(
            object_name="controlDict",
            application=application,
            startFrom=startFrom,
            startTime=startTime,
            stopAt=stopAt,
            endTime=endTime,
            deltaT=deltaT,
            writeControl=writeControl,
            writeInterval=writeInterval,
            purgeWrite=purgeWrite,
            writeFormat=writeFormat,
            writePrecision=writePrecision,
            writeCompression=writeCompression,
            timeFormat=timeFormat,
            timePrecision=timePrecision,
            runTimeModifiable=runTimeModifiable,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the controlDict parameters to a dictionary.
        """
        result = {
            'application': self.application,
            'startFrom': self.startFrom,
            'startTime': self.startTime,
            'stopAt': self.stopAt,
            'endTime': self.endTime,
            'deltaT': self.deltaT,
            'writeControl': self.writeControl,
            'writeInterval': self.writeInterval,
            'purgeWrite': self.purgeWrite,
            'writeFormat': self.writeFormat,
            'writePrecision': self.writePrecision,
            'writeCompression': self.writeCompression,
            'timeFormat': self.timeFormat,
            'timePrecision': self.timePrecision,
            'runTimeModifiable': self.runTimeModifiable,
            "libs": self.libs,
        }
        if self.functions:
            result["functions"] = self.functions
        if self.region_solvers:
            result["regionSolvers"] = self.region_solvers
        if self.sub_solver:
            result["subSolver"] = self.sub_solver
        return result

    @classmethod
    def from_dict(cls, config: Dict[str, Any], parent: Optional[Any] = None) -> "ControlDictFile":
        """
        Create a ControlDictFile instance from a dictionary and optional parent.
        """
        return cls(
            parent=parent,
            application=config.get('application'),
            startFrom=config.get('startFrom', "startTime"),
            startTime=config.get('startTime', 0),
            stopAt=config.get('stopAt', "endTime"),
            endTime=config.get('endTime', 5000),
            deltaT=config.get('deltaT', 1),
            writeControl=config.get('writeControl', "timeStep"),
            writeInterval=config.get('writeInterval', 100),
            purgeWrite=config.get('purgeWrite', 10),
            writeFormat=config.get('writeFormat', "ascii"),
            writePrecision=config.get('writePrecision', 6),
            writeCompression=config.get('writeCompression', "off"),
            timeFormat=config.get('timeFormat', "general"),
            timePrecision=config.get('timePrecision', 6),
            runTimeModifiable=config.get('runTimeModifiable', True),
            libs=config.get('libs',()),
            functions=config.get('functions'),
            region_solvers=config.get('regionSolvers'),
            sub_solver=config.get('subSolver'),
        )

    def add_library(self, lib_name: str):
        """Add a library to the controlDict."""
        if lib_name not in self.libs:
            self.libs.append(lib_name)

    def set_adaptive_time_step(
        self,
        adjustTimeStep: bool = True,
        maxCo: float = 0.8,
        maxAlphaCo: float = 1.2,
        maxDeltaT: float = 0.001,
        minDeltaT: float = 1e-7
    ):
        """Set adaptive time stepping parameters."""
        self.adaptiveTimeStep = {
            "adjustTimeStep": adjustTimeStep,
            "maxCo": maxCo,
            "maxAlphaCo": maxAlphaCo,
            "maxDeltaT": maxDeltaT,
            "minDeltaT": minDeltaT
        }

    def set_region_solvers(self, region_solvers: Dict[str, str]):
        """Configure multi-region solvers for CHT cases.

        Args:
            region_solvers: Dict mapping region names to solver
                types, e.g. {"fluid": "fluid", "solid": "solid"}.
        """
        self.region_solvers = region_solvers

    def add_function(self, function_name: str):
        """Add a functionObject include to the controlDict."""
        if function_name not in self.functions:
            self.functions.append(function_name)

    def write(self, filepath):
        write_attrs = self.attributes.copy()

        if self.adaptiveTimeStep:
            write_attrs.update(self.adaptiveTimeStep)
        if self.libs:
            includes_lib = "\n".join([f'"{fname}"' for fname in self.libs])
            write_attrs["libs"] = f'\n(\n{includes_lib} \n)'

        if getattr(self, "region_solvers", None):
            write_attrs["regionSolvers"] = self.region_solvers

        filepath = Path(filepath)
        with open(filepath, "w", encoding="utf-8") as file:
            file.write("FoamFile\n{\n")
            for key, value in self.header.items():
                file.write(f"    {key}     {value};\n")
            file.write("}\n\n")
            self._write_attributes(file, write_attrs)