from foampilot.base.openFOAMFile import OpenFOAMFile
from typing import Any, Optional
from pathlib import Path


class TurbulencePropertiesFile(OpenFOAMFile):

    AVAILABLE_RAS_MODELS = {
        "k-epsilon": "kEpsilon",
        "k-kl-omega": "kkLOmega",
        "SST": "kOmegaSST",
        "Realizable k-epsilon": "realizableKE",
        "RNG k-epsilon": "RNGkEpsilon",
        "Spalart-Allmaras": "SpalartAllmaras",
        "v2-f": "v2f",
    }

    AVAILABLE_LES_MODELS = {
        # futur
        # "Smagorinsky": "Smagorinsky",
    }

    def __init__(
        self,
        parent: Any,
        simulationType: str = "RAS",
        RASModel: Optional[str] = "k-epsilon",
        LESModel: Optional[str] = None,
        turbulence: str = "on",
        printCoeffs: str = "on",
    ):
        self.parent = parent

        simulationType = simulationType.lower()

        data = {
            "object_name": "turbulenceProperties",
            "simulationType": simulationType,
        }

        # ---- LAMINAR --------------------------------------------------
        if simulationType == "laminar":
            super().__init__(**data)
            return

        # ---- RAS ------------------------------------------------------
        if simulationType == "ras":
            if RASModel is None:
                raise ValueError("RASModel must be provided when simulationType='RAS'")

            model = self.AVAILABLE_RAS_MODELS.get(RASModel, RASModel)

            data["RAS"] = {
                "RASModel": model,
                "turbulence": turbulence,
                "printCoeffs": printCoeffs,
            }

            super().__init__(**data)
            return

        # ---- LES ------------------------------------------------------
        if simulationType == "les":
            if LESModel is None:
                raise ValueError("LESModel must be provided when simulationType='LES'")

            model = self.AVAILABLE_LES_MODELS.get(LESModel, LESModel)

            data["LES"] = {
                "LESModel": model,
                "turbulence": turbulence,
                "printCoeffs": printCoeffs,
            }

            super().__init__(**data)
            return

        raise ValueError(f"Unknown simulationType '{simulationType}'")

    # -----------------------------------------------------------------
    # EXTENSION API
    # -----------------------------------------------------------------
    @classmethod
    def add_turbulence_model(
        cls,
        simulation_type: str,
        model_name: str,
        model_code: str,
    ):
        """
        Register a new turbulence model.

        Parameters
        ----------
        simulation_type : str
            'RAS' or 'LES'
        model_name : str
            User-readable name.
        model_code : str
            OpenFOAM internal model identifier.
        """
        simulation_type = simulation_type.lower()

        if simulation_type == "ras":
            cls.AVAILABLE_RAS_MODELS[model_name] = model_code
            return

        if simulation_type == "les":
            cls.AVAILABLE_LES_MODELS[model_name] = model_code
            return

        raise ValueError(
            "simulation_type must be 'RAS' or 'LES' (laminar has no models)"
        )

    def write(self, filepath: Path):
        self.write_file(filepath)
