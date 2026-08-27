from pathlib import Path
from typing import Optional, Any
from foampilot.base.openFOAMFile import OpenFOAMFile


class MomentumTransportFile(OpenFOAMFile):
    """
    OpenFOAM ``constant/momentumTransport`` file.

    In OpenFOAM 13 the ``turbulenceProperties`` file (which used to contain
    ``simulationType``) was renamed to ``momentumTransport``.  This class
    generates the full file including the ``RAS`` or ``LES`` sub-dictionary
    when appropriate.
    """

    def __init__(
        self,
        parent: Optional[Any] = None,
        simulationType: str = "laminar",
        RASModel: Optional[str] = None,
        LESModel: Optional[str] = None,
    ):
        self.parent = parent
        self.simulationType = simulationType  # Keep original case (RAS, LES, laminar)
        self._RASModel = RASModel
        self._LESModel = LESModel

        data = {
            "object_name": "momentumTransport",
            "simulationType": self.simulationType,
        }

        super().__init__(**data)

        if self.simulationType == "laminar":
            return

        sim_type_lower = self.simulationType.lower()
        if sim_type_lower == "ras":
            self.attributes["RAS"] = {
                "model": self._RASModel or "kEpsilon",
                "turbulence": "on",
            }
        elif sim_type_lower == "les":
            model = self._LESModel or "Smagorinsky"
            self.attributes["LES"] = {
                "model": model,
                "turbulence": "on",
                "delta": "cubeRootVol" if str(model).lower() == "keqn" else "vanDriest",
            }

    def write(self, filepath: Path):
        self.write_file(filepath)
