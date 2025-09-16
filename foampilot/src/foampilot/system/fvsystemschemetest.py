# system/fvSchemesFile.py
from typing import Dict, Optional, Any
from foampilot.base.openFOAMFile import OpenFOAMFile


class FvSchemesFile(OpenFOAMFile):
    """
    Represents the fvSchemes file in OpenFOAM.

    This class builds schemes depending on parent Foam attributes:
        - simulation_type: "incompressible", "boussinesq", "compressible"
        - energy_variable (compressible only): "e", "h" or "T"
    """

    def __init__(
        self,
        parent: Any,
        ddtSchemes: Optional[Dict[str, str]] = None,
        gradSchemes: Optional[Dict[str, str]] = None,
        divSchemes: Optional[Dict[str, str]] = None,
        laplacianSchemes: Optional[Dict[str, str]] = None,
        interpolationSchemes: Optional[Dict[str, str]] = None,
        snGradSchemes: Optional[Dict[str, str]] = None,
    ) -> None:
        self.parent = parent

        self.ddtSchemes = self._init_ddt_schemes(ddtSchemes)
        self.gradSchemes = self._init_grad_schemes(gradSchemes)
        self.divSchemes = self._init_div_schemes(divSchemes)
        self.laplacianSchemes = self._init_laplacian_schemes(laplacianSchemes)
        self.interpolationSchemes = self._init_interpolation_schemes(interpolationSchemes)
        self.snGradSchemes = self._init_sn_grad_schemes(snGradSchemes)

        super().__init__(
            object_name="fvSchemes",
            ddtSchemes=self.ddtSchemes,
            gradSchemes=self.gradSchemes,
            divSchemes=self.divSchemes,
            laplacianSchemes=self.laplacianSchemes,
            interpolationSchemes=self.interpolationSchemes,
            snGradSchemes=self.snGradSchemes,
        )

    # -------------------------
    # Initialization helpers
    # -------------------------
    def _init_ddt_schemes(self, ddtSchemes: Optional[Dict[str, str]]) -> Dict[str, str]:
        return ddtSchemes.copy() if ddtSchemes else {"default": "Euler"}

    def _init_grad_schemes(self, gradSchemes: Optional[Dict[str, str]]) -> Dict[str, str]:
        return gradSchemes.copy() if gradSchemes else {"default": "Gauss linear"}

    def _init_div_schemes(self, divSchemes: Optional[Dict[str, str]]) -> Dict[str, str]:
        if divSchemes is not None:
            return divSchemes.copy()

        divSchemes = {
            "default": "none",
            "div(phi,U)": "Gauss upwind",
            "div(phi,k)": "Gauss upwind",
            "div(phi,epsilon)": "Gauss upwind",
            "div(((rho*nuEff)*dev2(T(grad(U)))))": "Gauss linear",
        }

        if self.parent.simulation_type == "boussinesq":
            divSchemes["div(phi,T)"] = "Gauss upwind"

        elif self.parent.simulation_type == "compressible":
            energy = getattr(self.parent, "energy_variable", "e")
            if energy in ("e", "h", "T"):
                divSchemes[f"div(phi,{energy})"] = "Gauss upwind"
            divSchemes["div(phi,(p|rho))"] = "Gauss linear"

        return divSchemes

    def _init_laplacian_schemes(self, laplacianSchemes: Optional[Dict[str, str]]) -> Dict[str, str]:
        if laplacianSchemes is not None:
            return laplacianSchemes.copy()

        laplacianSchemes = {"default": "Gauss linear corrected"}

        if self.parent.simulation_type == "boussinesq":
            laplacianSchemes["laplacian(alphaEff,T)"] = "Gauss linear corrected"

        elif self.parent.simulation_type == "compressible":
            energy = getattr(self.parent, "energy_variable", "e")
            laplacianSchemes[f"laplacian(alphaEff,{energy})"] = "Gauss linear corrected"

        return laplacianSchemes

    def _init_interpolation_schemes(self, interpolationSchemes: Optional[Dict[str, str]]) -> Dict[str, str]:
        return interpolationSchemes.copy() if interpolationSchemes else {"default": "linear"}

    def _init_sn_grad_schemes(self, snGradSchemes: Optional[Dict[str, str]]) -> Dict[str, str]:
        return snGradSchemes.copy() if snGradSchemes else {"default": "corrected"}

    # -------------------------
    # Export / Import
    # -------------------------
    def to_dict(self) -> Dict[str, Dict[str, str]]:
        """Export to OpenFOAM dictionary structure."""
        return {
            "ddtSchemes": self.ddtSchemes,
            "gradSchemes": self.gradSchemes,
            "divSchemes": self.divSchemes,
            "laplacianSchemes": self.laplacianSchemes,
            "interpolationSchemes": self.interpolationSchemes,
            "snGradSchemes": self.snGradSchemes,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Dict[str, str]], parent: Any) -> "FvSchemesFile":
        """Build instance from dict + Foam parent."""
        return cls(
            parent=parent,
            ddtSchemes=config.get("ddtSchemes"),
            gradSchemes=config.get("gradSchemes"),
            divSchemes=config.get("divSchemes"),
            laplacianSchemes=config.get("laplacianSchemes"),
            interpolationSchemes=config.get("interpolationSchemes"),
            snGradSchemes=config.get("snGradSchemes"),
        )