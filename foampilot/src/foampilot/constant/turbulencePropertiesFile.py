from foampilot.base.openFOAMFile import OpenFOAMFile
from typing import Any
from pathlib import Path




class TurbulencePropertiesFile(OpenFOAMFile):
    """
    Represents an OpenFOAM `turbulenceProperties` configuration file.

    This class allows setting up turbulence models, simulation type, 
    and related parameters. It inherits from :class:`OpenFOAMFile`.

    Attributes
    ----------
    AVAILABLE_MODELS : dict
        Dictionary mapping user-friendly model names to internal OpenFOAM codes.
    """

    # Dictionary of available turbulence models
    AVAILABLE_MODELS = {
        "k-epsilon": "kEpsilon",
        "k-kl-omega": "kkLOmega",
        "SST": "kOmegaSST",
        "Realizable k-epsilon": "realizableKE",
        "RNG k-epsilon": "RNGkEpsilon",
        "Spalart-Allmaras": "SpalartAllmaras",
        "v2-f": "v2f"
    }

    def __init__(self, parent: Any, simulationType="RAS", RASModel="k-epsilon", turbulence="on", printCoeffs="on"):
        self.parent = parent
        """
        Initialize a turbulenceProperties file.

        Parameters
        ----------
        simulationType : str, optional
            The turbulence simulation type (default: "RAS").
        RASModel : str, optional
            The turbulence model to use. Must be in `AVAILABLE_MODELS` or a custom one (default: "k-epsilon").
        turbulence : str, optional
            Whether turbulence is enabled ("on"/"off") (default: "on").
        printCoeffs : str, optional
            Whether to print model coefficients ("on"/"off") (default: "on").
        """
        # Check if the chosen model is in the available models
        if RASModel in self.AVAILABLE_MODELS:
            RASModel_internal = self.AVAILABLE_MODELS[RASModel]
        else:
            # Allow a custom non-standard model
            print(f"Unknown RAS model '{RASModel}', it will be added as a custom model.")
            RASModel_internal = RASModel

        super().__init__(
            object_name="turbulenceProperties",
            simulationType=simulationType,
            RAS={"RASModel": RASModel_internal, "turbulence": turbulence, "printCoeffs": printCoeffs}
        )

    @classmethod
    def add_turbulence_model(cls, model_name, model_code):
        """
        Adds a new turbulence model to the dictionary of available models.

        Parameters
        ----------
        model_name : str
            User-readable name for the model.
        model_code : str
            Internal code for the model in OpenFOAM.
        """
        cls.AVAILABLE_MODELS[model_name] = model_code

    def write(self, filepath: Path):
        """Write the turbulenceProperties file."""
        self.write_file(filepath)