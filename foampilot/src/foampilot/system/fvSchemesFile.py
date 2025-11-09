from typing import Dict, Optional, Any, List
from foampilot.base.openFOAMFile import OpenFOAMFile

class FvSchemesFile(OpenFOAMFile):
    """
    Represents the fvSchemes file in OpenFOAM.

    Automatically builds schemes depending on:
      - Parent Foam attributes (simulation_type, energy_variable)
      - CaseFieldsManager (dynamic field detection)
    """

    DEFAULT_DDT = "Euler"
    DEFAULT_GRAD = "Gauss linear"
    DEFAULT_LAPLACIAN = "Gauss linear corrected"
    DEFAULT_INTERP = "linear"
    DEFAULT_SNGRAD = "corrected"

    def __init__(
        self,
        parent: Any,
        fields_manager: Optional[Any] = None,
        ddtSchemes: Optional[Dict[str, str]] = None,
        gradSchemes: Optional[Dict[str, str]] = None,
        divSchemes: Optional[Dict[str, str]] = None,
        laplacianSchemes: Optional[Dict[str, str]] = None,
        interpolationSchemes: Optional[Dict[str, str]] = None,
        snGradSchemes: Optional[Dict[str, str]] = None,
        wallDist: Optional[Dict[str, str]] = None,
    ) -> None:
        self.parent = parent
        self.fields_manager = fields_manager

        # Initialize default schemes
        self.ddtSchemes = self._init_ddt(ddtSchemes)
        self.gradSchemes = self._init_grad(gradSchemes)
        self.divSchemes = self._init_div(divSchemes)
        self.laplacianSchemes = self._init_laplacian(laplacianSchemes)
        self.interpolationSchemes = self._init_interpolation(interpolationSchemes)
        self.snGradSchemes = self._init_sn_grad(snGradSchemes)
        self.wallDist = self._init_wall_dist(wallDist)

        # Configure schemes based on available fields
        if self.fields_manager:
            self._configure_from_fields()

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

    def _configure_from_fields(self) -> None:
        """Configure schemes based on fields available in CaseFieldsManager."""
        if not self.fields_manager:
            return

        field_names = self.fields_manager.get_field_names()
        sim_type = getattr(self.parent, "simulation_type", "incompressible")
        energy_var = getattr(self.parent, "energy_variable", "e")

        # --- divSchemes ---
        if "U" in field_names:
            # Override default U scheme if gravity is active (p_rgh)
            pressure_field = "p_rgh" if "p_rgh" in field_names else "p"
            self.divSchemes["div(phi,U)"] = "bounded Gauss linearUpwind grad(U)"

            # Add turbulence terms if present
            if any(f in field_names for f in ["k", "epsilon", "omega", "nut"]):
                self.divSchemes["div(phi,k)"] = "bounded Gauss upwind"
                self.divSchemes["div(phi,epsilon)"] = "bounded Gauss upwind"
                self.divSchemes.setdefault("div(phi,omega)", "bounded Gauss upwind")
                self.divSchemes["div((nuEff*dev2(T(grad(U)))))"] = "Gauss linear"

        # VoF-specific schemes
        if sim_type == "vof" and any(f.startswith("alpha.") for f in field_names):
            self.divSchemes["div(phi,alpha)"] = "Gauss MPLIC"
            self.divSchemes["div(rhoPhi,U)"] = "Gauss upwind"

            # Add interface compression for each alpha field
            for field in field_names:
                if field.startswith("alpha."):
                    phase = field.split(".")[1]
                    self.divSchemes[f"div(phirb,{field})"] = f"Gauss interfaceCompression {phase}"

        # Energy schemes
        if "T" in field_names:
            if sim_type == "boussinesq":
                self.divSchemes["div(phi,T)"] = "bounded Gauss upwind"
            elif sim_type == "compressible":
                self.divSchemes[f"div(phi,{energy_var})"] = "bounded Gauss upwind"

        # --- laplacianSchemes ---
        if "T" in field_names:
            if sim_type == "boussinesq":
                self.laplacianSchemes["laplacian(alphaEff,T)"] = self.DEFAULT_LAPLACIAN
            elif sim_type == "compressible":
                self.laplacianSchemes[f"laplacian(alphaEff,{energy_var})"] = self.DEFAULT_LAPLACIAN

        # Turbulence diffusion
        if any(f in field_names for f in ["k", "epsilon", "omega"]):
            for field in ["k", "epsilon", "omega"]:
                if field in field_names:
                    self.laplacianSchemes[f"laplacian(nuEff,{field})"] = self.DEFAULT_LAPLACIAN

        # --- wallDist ---
        if sim_type == "vof" and self.wallDist is None:
            self.wallDist = {"method": "meshWave"}

    # -------------------------
    # Initialization helpers (unchanged)
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
    # Export / Import (unchanged)
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

    def write(self, filepath):
        """Write the fvSchemes file."""
        self.write_file(filepath)

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