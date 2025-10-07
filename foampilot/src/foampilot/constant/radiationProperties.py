from foampilot.base.openFOAMFile import OpenFOAMFile
from pathlib import Path


class RadiationPropertiesFile(OpenFOAMFile):
    """
    Represents the OpenFOAM `radiationProperties` file inside the `constant/` directory.

    Supports radiation models: P1 and fvDOM.
    """

    SUPPORTED_MODELS = ["P1", "fvDOM"]

    def __init__(
        self,
        model: str = "P1",
        absorptivity: float = 0.5,
        emissivity: float = 0.5,
        E: float = 0.0,
        solver_freq: int | float = None,
        nPhi: int = 3,
        nTheta: int = 5,
        tolerance: float = 1e-3,
        maxIter: int = 10,
    ):
        super().__init__(object_name="radiationProperties")
        self.model = model
        self.absorptivity = absorptivity
        self.emissivity = emissivity
        self.E = E
        self.solver_freq = solver_freq or (1 if model.lower() == "p1" else 10)
        self.nPhi = nPhi
        self.nTheta = nTheta
        self.tolerance = tolerance
        self.maxIter = maxIter

        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported radiation model '{model}'. Supported: {self.SUPPORTED_MODELS}")

        self._configure_attributes()

    def _configure_attributes(self):
        """
        Define attributes based on the chosen radiation model.
        """
        self.attributes["radiationModel"] = self.model
        self.attributes["solverFreq"] = self.solver_freq
        self.attributes["absorptionEmissionModel"] = "constant"

        self.attributes["constantCoeffs"] = {
            "absorptivity": self.absorptivity,
            "emissivity": self.emissivity,
            "E": self.E,
        }

        if self.model.lower() == "fvdom":
            self.attributes["fvDOMCoeffs"] = {
                "nPhi": self.nPhi,
                "nTheta": self.nTheta,
                "tolerance": self.tolerance,
                "maxIter": self.maxIter,
            }

        self.attributes["scatterModel"] = "none"
        self.attributes["sootModel"] = "none"

    def write(self, base_path: Path | str):
        """
        Write the radiationProperties file under base_path/constant/.
        """
        base_path = Path(base_path)
        file_path = base_path / self.object_name
        self.write_file(file_path)
        print(f"✅ radiationProperties écrit ({self.model}) → {file_path}")


class FvModelsFile(OpenFOAMFile):
    """
    Represents the `fvModels` file inside the `constant/` directory,
    automatically enabling the radiation model.
    """

    def __init__(self):
        super().__init__(object_name="fvModels")
        self.attributes["radiation"] = {
            "type": "radiation",
            "libs": ["libradiationModels.so"],
        }

    def write(self, base_path: Path | str):
        """
        Write the fvModels file under base_path/constant/.
        """
        base_path = Path(base_path)
        file_path = base_path / self.object_name
        self.write_file(file_path)
        print(f"✅ fvModels écrit → {file_path}")