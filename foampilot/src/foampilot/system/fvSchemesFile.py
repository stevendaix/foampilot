# system/fvSchemesFile.py
from typing import Dict, Optional, Any
from foampilot.base.openFOAMFile import OpenFOAMFile


class FvSchemesFile(OpenFOAMFile):
    """
    Represents the fvSchemes file in OpenFOAM.

    Automatically builds schemes depending on parent Foam attributes:
        - simulation_type: "incompressible", "boussinesq", "compressible", "vof"
        - energy_variable (compressible only): "e", "h" or "T"
    """

    DEFAULT_DDT = "Euler"
    DEFAULT_GRAD = "Gauss linear"
    DEFAULT_LAPLACIAN = "Gauss linear corrected"
    DEFAULT_INTERP = "linear"
    DEFAULT_SNGRAD = "corrected"

    def __init__(
        self,
        parent: Any,
        ddtSchemes: Optional[Dict[str, str]] = None,
        gradSchemes: Optional[Dict[str, str]] = None,
        divSchemes: Optional[Dict[str, str]] = None,
        laplacianSchemes: Optional[Dict[str, str]] = None,
        interpolationSchemes: Optional[Dict[str, str]] = None,
        snGradSchemes: Optional[Dict[str, str]] = None,
        wallDist: Optional[Dict[str, str]] = None,
    ) -> None:
        self.parent = parent

        self.ddtSchemes = self._init_ddt(ddtSchemes)
        self.gradSchemes = self._init_grad(gradSchemes)
        self.divSchemes = self._init_div(divSchemes)
        self.laplacianSchemes = self._init_laplacian(laplacianSchemes)
        self.interpolationSchemes = self._init_interpolation(interpolationSchemes)
        self.snGradSchemes = self._init_sn_grad(snGradSchemes)
        self.wallDist = self._init_wall_dist(wallDist)

        super().__init__(
            object_name="fvSchemes",
            ddtSchemes=self.ddtSchemes,
            gradSchemes=self.gradSchemes,
            divSchemes=self.divSchemes,
            laplacianSchemes=self.laplacianSchemes,
            interpolationSchemes=self.interpolationSchemes,
            snGradSchemes=self.snGradSchemes,
            wallDist=self.wallDist,
        )

    # -------------------------
    # Initialization helpers
    # -------------------------
    def _init_ddt(self, ddt: Optional[Dict[str, str]]) -> Dict[str, str]:
        return ddt.copy() if ddt else {"default": self.DEFAULT_DDT}

    def _init_grad(self, grad: Optional[Dict[str, str]]) -> Dict[str, str]:
        return grad.copy() if grad else {"default": self.DEFAULT_GRAD}

    def _init_div(self, div: Optional[Dict[str, str]]) -> Dict[str, str]:
        if div:
            return div.copy()

        divSchemes = {"default": "none"}

        sim = getattr(self.parent, "simulation_type", "incompressible")
        energy = getattr(self.parent, "energy_variable", "e")

        # Standard fields
        divSchemes.update({
            "div(phi,U)": "Gauss upwind",
            "div(phi,k)": "Gauss upwind",
            "div(phi,epsilon)": "Gauss upwind",
            "div(((rho*nuEff)*dev2(T(grad(U)))))": "Gauss linear",
        })

        # Specific simulation types
        if sim == "boussinesq":
            divSchemes["div(phi,T)"] = "Gauss upwind"

        elif sim == "compressible":
            if energy in ("e", "h", "T"):
                divSchemes[f"div(phi,{energy})"] = "Gauss upwind"
            divSchemes["div(phi,(p|rho))"] = "Gauss linear"

        elif sim == "vof":
            divSchemes["div(rhoPhi,U)"] = "Gauss upwind"
            divSchemes["div(phi,alpha)"] = "Gauss MPLIC"
            # Optional extra fields
            divSchemes.setdefault("div(phi,omega)", "Gauss upwind")
            divSchemes.setdefault("div(phi,k)", "Gauss upwind")

        return divSchemes

    def _init_laplacian(self, lap: Optional[Dict[str, str]]) -> Dict[str, str]:
        if lap:
            return lap.copy()

        lapSchemes = {"default": self.DEFAULT_LAPLACIAN}
        sim = getattr(self.parent, "simulation_type", "incompressible")
        energy = getattr(self.parent, "energy_variable", "e")

        if sim == "boussinesq":
            lapSchemes["laplacian(alphaEff,T)"] = self.DEFAULT_LAPLACIAN
        elif sim == "compressible":
            lapSchemes[f"laplacian(alphaEff,{energy})"] = self.DEFAULT_LAPLACIAN

        return lapSchemes

    def _init_interpolation(self, interp: Optional[Dict[str, str]]) -> Dict[str, str]:
        return interp.copy() if interp else {"default": self.DEFAULT_INTERP}

    def _init_sn_grad(self, sn: Optional[Dict[str, str]]) -> Dict[str, str]:
        return sn.copy() if sn else {"default": self.DEFAULT_SNGRAD}

    def _init_wall_dist(self, wall: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        sim = getattr(self.parent, "simulation_type", "incompressible")
        if sim == "vof" and wall is None:
            return {"method": "meshWave"}
        return wall.copy() if wall else None

    # -------------------------
    # Export / Import
    # -------------------------
    def to_dict(self) -> Dict[str, Dict[str, str]]:
        d = {
            "ddtSchemes": self.ddtSchemes,
            "gradSchemes": self.gradSchemes,
            "divSchemes": self.divSchemes,
            "laplacianSchemes": self.laplacianSchemes,
            "interpolationSchemes": self.interpolationSchemes,
            "snGradSchemes": self.snGradSchemes,
        }
        if self.wallDist:
            d["wallDist"] = self.wallDist
        return d

    @classmethod
    def from_dict(cls, config: Dict[str, Dict[str, str]], parent: Any) -> "FvSchemesFile":
        return cls(
            parent=parent,
            ddtSchemes=config.get("ddtSchemes"),
            gradSchemes=config.get("gradSchemes"),
            divSchemes=config.get("divSchemes"),
            laplacianSchemes=config.get("laplacianSchemes"),
            interpolationSchemes=config.get("interpolationSchemes"),
            snGradSchemes=config.get("snGradSchemes"),
            wallDist=config.get("wallDist"),
        )