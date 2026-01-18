from foampilot.base.openFOAMFile import OpenFOAMFile
from typing import Optional, Dict, Any,List,Tuple


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
        return {
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
            "libs": self.libs
        }

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

    def write(self, filepath):
        """
        Write the controlDict file with proper formatting for 'libs' and 'functions'.

        Uses OpenFOAMFile.write_file() for all other attributes.
        """
        # Faire une copie temporaire des attributs
        write_attrs = self.attributes.copy()

        # Gestion adaptive time stepping
        if self.adaptiveTimeStep:
            write_attrs.update(self.adaptiveTimeStep)
        # Gestion des libs
        if self.libs:
            includes_lib = "\n".join([f'"{fname}"' for fname in self.libs])
            write_attrs["libs"] = f'\n(\n{includes_lib} \n)'


        # Remplacer temporairement attributes
        old_attrs = self.attributes
        self.attributes = write_attrs

        # Écrire le fichier via OpenFOAMFile
        super().write_file(filepath)

        # Restaurer attributes originaux
        self.attributes = old_attrs