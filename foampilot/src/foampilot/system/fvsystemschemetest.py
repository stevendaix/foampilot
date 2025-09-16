# system/fvSchemesFile.py
from foampilot.base.openFOAMFile import OpenFOAMFile

class FvSchemesFile(OpenFOAMFile):
    """
    fvSchemes configuration adaptable to simulation type.
    
    It builds schemes depending on parent Foam attributes:
        - simulation_type: "incompressible", "boussinesq", "compressible"
        - energy_variable (compressible only): "e", "h" or "T"
    """

    def __init__(self, parent,
                 ddtSchemes=None, gradSchemes=None, divSchemes=None,
                 laplacianSchemes=None, interpolationSchemes=None, snGradSchemes=None):
        """
        Args:
            parent: Foam instance (or compatible) exposing attributes:
                - parent.simulation_type
                - parent.energy_variable (if compressible)
        """
        self.parent = parent

        # === Defaults ===
        if ddtSchemes is None:
            ddtSchemes = {"default": "Euler"}

        if gradSchemes is None:
            gradSchemes = {"default": "Gauss linear"}

        if divSchemes is None:
            divSchemes = {"default": "none",
                          "div(phi,U)": "Gauss upwind"}

            # --- Handle physics ---
            if self.parent.simulation_type == "boussinesq":
                divSchemes["div(phi,T)"] = "Gauss upwind"

            elif self.parent.simulation_type == "compressible":
                energy = getattr(self.parent, "energy_variable", "e")
                if energy in ("e", "h", "T"):
                    divSchemes[f"div(phi,{energy})"] = "Gauss upwind"
                divSchemes["div(phi,(p|rho))"] = "Gauss linear"

            # turbulence terms
            divSchemes.update({
                "div(phi,k)": "Gauss upwind",
                "div(phi,epsilon)": "Gauss upwind",
                "div(((rho*nuEff)*dev2(T(grad(U)))))": "Gauss linear"
            })

        if laplacianSchemes is None:
            laplacianSchemes = {"default": "Gauss linear corrected"}

            if self.parent.simulation_type == "boussinesq":
                laplacianSchemes["laplacian(alphaEff,T)"] = "Gauss linear corrected"

            elif self.parent.simulation_type == "compressible":
                energy = getattr(self.parent, "energy_variable", "e")
                laplacianSchemes[f"laplacian(alphaEff,{energy})"] = "Gauss linear corrected"

        if interpolationSchemes is None:
            interpolationSchemes = {"default": "linear"}

        if snGradSchemes is None:
            snGradSchemes = {"default": "corrected"}

        # Call parent class constructor
        super().__init__(
            object_name="fvSchemes",
            ddtSchemes=ddtSchemes,
            gradSchemes=gradSchemes,
            divSchemes=divSchemes,
            laplacianSchemes=laplacianSchemes,
            interpolationSchemes=interpolationSchemes,
            snGradSchemes=snGradSchemes
        )

    def to_dict(self):
        """Export to OpenFOAM dictionary structure."""
        return {
            'ddtSchemes': self.ddtSchemes,
            'gradSchemes': self.gradSchemes,
            'divSchemes': self.divSchemes,
            'laplacianSchemes': self.laplacianSchemes,
            'interpolationSchemes': self.interpolationSchemes,
            'snGradSchemes': self.snGradSchemes
        }

    @classmethod
    def from_dict(cls, config, parent):
        """Build instance from dict + Foam parent."""
        return cls(
            parent=parent,
            ddtSchemes=config.get('ddtSchemes', {}),
            gradSchemes=config.get('gradSchemes', {}),
            divSchemes=config.get('divSchemes', {}),
            laplacianSchemes=config.get('laplacianSchemes', {}),
            interpolationSchemes=config.get('interpolationSchemes', {}),
            snGradSchemes=config.get('snGradSchemes', {})
        )