from foampilot.base.openFOAMFile import OpenFOAMFile
from pathlib import Path
from typing import Optional, Any, Dict, Union


class RadiationPropertiesFile(OpenFOAMFile):
    """
    Represents the OpenFOAM `radiationProperties` file inside the `constant/` directory.

    Supports dynamic configuration based on:
      - CaseFieldsManager (for field detection)
      - Radiation models: P1, fvDOM, and S2S (new)
      - Automatic detection of radiation fields (e.g., K, G)

    Examples
    --------
    >>> # Default P1 model
    >>> radiation = RadiationPropertiesFile()
    >>> radiation.write("./constant")

    >>> # fvDOM model with custom parameters
    >>> radiation = RadiationPropertiesFile(
    >>>     model="fvDOM",
    >>>     nPhi=5,
    >>>     nTheta=7,
    >>>     tolerance=1e-4
    >>> )
    >>> radiation.write("./constant")
    """

    SUPPORTED_MODELS = ["P1", "fvDOM", "S2S"]

    def __init__(
        self,
        parent: Optional[Any] = None,
        model: str = "P1",
        absorptivity: float = 0.5,
        emissivity: float = 0.5,
        E: float = 0.0,
        solver_freq: Optional[Union[int, float]] = None,
        nPhi: int = 3,
        nTheta: int = 5,
        tolerance: float = 1e-3,
        maxIter: int = 10,
        scatterModel: str = "none",
        sootModel: str = "none",
    ):
        """
        Initialize a `radiationProperties` file.

        Parameters
        ----------
        parent : Any, optional
            Parent object with `fields_manager` (for dynamic field detection).
        model : str, optional
            Radiation model: "P1" (default), "fvDOM", or "S2S".
        absorptivity : float, optional
            Absorptivity coefficient (default: 0.5).
        emissivity : float, optional
            Emissivity coefficient (default: 0.5).
        E : float, optional
            Emission coefficient (default: 0.0).
        solver_freq : int or float, optional
            Solver frequency (default: 1 for P1, 10 for others).
        nPhi : int, optional
            Number of phi divisions for fvDOM (default: 3).
        nTheta : int, optional
            Number of theta divisions for fvDOM (default: 5).
        tolerance : float, optional
            Solver tolerance (default: 1e-3).
        maxIter : int, optional
            Maximum iterations (default: 10).
        scatterModel : str, optional
            Scatter model (default: "none").
        sootModel : str, optional
            Soot model (default: "none").
        """
        super().__init__(object_name="radiationProperties")
        self.parent = parent
        self.model = model
        self.absorptivity = absorptivity
        self.emissivity = emissivity
        self.E = E
        self.solver_freq = solver_freq or (1 if model.lower() == "p1" else 10)
        self.nPhi = nPhi
        self.nTheta = nTheta
        self.tolerance = tolerance
        self.maxIter = maxIter
        self.scatterModel = scatterModel
        self.sootModel = sootModel

        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported radiation model '{model}'. Supported: {self.SUPPORTED_MODELS}")

        # Configure attributes based on model and fields
        self._configure_attributes()

        # Override with dynamic fields if parent has CaseFieldsManager
        if self.parent and hasattr(self.parent, "fields_manager"):
            self._configure_from_fields()

    def _configure_attributes(self):
        """
        Define attributes based on the chosen radiation model.
        """
        self.attributes = {
            "radiationModel": self.model,
            "solverFreq": self.solver_freq,
            "absorptionEmissionModel": "constant",
            "constantCoeffs": {
                "absorptivity": self.absorptivity,
                "emissivity": self.emissivity,
                "E": self.E,
            },
            "scatterModel": self.scatterModel,
            "sootModel": self.sootModel,
        }

        if self.model.lower() == "fvdom":
            self.attributes["fvDOMCoeffs"] = {
                "nPhi": self.nPhi,
                "nTheta": self.nTheta,
                "tolerance": self.tolerance,
                "maxIter": self.maxIter,
            }
        elif self.model.lower() == "s2s":
            self.attributes["S2SCoeffs"] = {
                "tolerance": self.tolerance,
                "maxIter": self.maxIter,
            }

    def _configure_from_fields(self):
        """
        Override configuration based on fields available in CaseFieldsManager.
        """
        if not hasattr(self.parent, "fields_manager"):
            return

        fields_manager = self.parent.fields_manager
        field_names = fields_manager.get_field_names()

        # Detect radiation-related fields (e.g., K, G)
        if "K" in field_names:
            # Adjust solver frequency if radiation field is present
            self.attributes["solverFreq"] = min(5, self.solver_freq)  # More frequent updates

        # Detect energy field (T) to enable radiation-energy coupling
        if "T" in field_names:
            self.attributes["absorptionEmissionModel"] = "semiTransparent"
            self.attributes["constantCoeffs"]["E"] = 1.0  # Enable emission

    def write(self, base_path: Union[Path, str]):
        """
        Write the radiationProperties file under base_path/constant/.
        """
        base_path = Path(base_path)
        file_path = base_path / "constant" / self.object_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_file(file_path)
        print(f"✅ radiationProperties écrit ({self.model}) → {file_path}")


from foampilot.base.openFOAMFile import OpenFOAMFile
from pathlib import Path
from typing import Optional, Any, Dict, Union

class FvModelsFile(OpenFOAMFile):
    """
    Represents the `fvModels` file inside the `constant/` directory.
    Automatically enables the radiation model and supports dynamic configuration.

    Examples
    --------
    >>> # Default radiation model
    >>> fv_models = FvModelsFile()
    >>> fv_models.write("./constant")

    >>> # With custom radiation library
    >>> fv_models = FvModelsFile(radiation_libs=["libradiationModels.so", "libmyCustomRadiation.so"])
    >>> fv_models.write("./constant")
    """

    def __init__(
        self,
        parent: Optional[Any] = None,
        radiation_libs: Optional[list] = None,
        additional_models: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        """
        Initialize an `fvModels` file.

        Parameters
        ----------
        parent : Any, optional
            Parent object with `fields_manager` (for dynamic field detection).
        radiation_libs : list, optional
            List of radiation libraries to load (default: ["libradiationModels.so"]).
        additional_models : dict, optional
            Additional models to include in fvModels (e.g., {"combustion": {"type": "combustion"}}).
        """
        super().__init__(object_name="fvModels")
        self.parent = parent
        self.radiation_libs = radiation_libs or ["libradiationModels.so"]
        self.additional_models = additional_models or {}

        # Configure attributes
        self._configure_attributes()

        # Override with dynamic fields if parent has CaseFieldsManager
        if self.parent and hasattr(self.parent, "fields_manager"):
            self._configure_from_fields()

    def _configure_attributes(self):
        """
        Define attributes for fvModels.
        """
        self.attributes = {
            "radiation": {
                "type": "radiation",
                "libs": self.radiation_libs,
            }
        }

        # Add additional models
        self.attributes.update(self.additional_models)

    def _configure_from_fields(self):
        """
        Override configuration based on fields available in CaseFieldsManager.
        """
        if not hasattr(self.parent, "fields_manager"):
            return

        fields_manager = self.parent.fields_manager
        field_names = fields_manager.get_field_names()

        # Detect combustion fields to add combustion model
        if any(field.startswith("Y") for field in field_names):  # e.g., YFuel, YO2
            self.attributes["combustion"] = {
                "type": "combustion",
                "libs": ["libcombustionModels.so"],
            }

        # Detect turbulence fields to ensure radiation-turbulence coupling
        if any(field in field_names for field in ["k", "epsilon", "omega", "nut"]):
            self.attributes["radiation"]["turbulence"] = "yes"

    def write(self, base_path: Union[Path, str]):
        """
        Write the fvModels file under base_path/constant/.
        """
        base_path = Path(base_path)
        file_path = base_path / "constant" / self.object_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_file(file_path)
        print(f"✅ fvModels écrit → {file_path}")
