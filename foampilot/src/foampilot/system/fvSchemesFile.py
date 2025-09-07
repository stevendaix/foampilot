# system/fvSchemesFile.py
from foampilot.base.openFOAMFile import OpenFOAMFile

class FvSchemesFile(OpenFOAMFile):
    """
    A class representing the fvSchemes file in OpenFOAM.
    
    This class handles the creation and manipulation of the fvSchemes file which defines
    the numerical schemes used in an OpenFOAM simulation. It inherits from OpenFOAMFile
    and provides specific functionality for finite volume schemes configuration.

    Attributes:
        ddtSchemes (dict): Time derivative schemes configuration.
        gradSchemes (dict): Gradient schemes configuration.
        divSchemes (dict): Divergence schemes configuration.
        laplacianSchemes (dict): Laplacian schemes configuration.
        interpolationSchemes (dict): Interpolation schemes configuration.
        snGradSchemes (dict): Surface normal gradient schemes configuration.
    """
    
    def __init__(self, ddtSchemes=None, gradSchemes=None, divSchemes=None,
                 laplacianSchemes=None, interpolationSchemes=None, snGradSchemes=None):
        """
        Initialize the FvSchemesFile with numerical schemes configuration.

        Args:
            ddtSchemes: Time derivative schemes (default: {"default": "steadyState"}).
            gradSchemes: Gradient schemes (default: {"default": "Gauss linear"}).
            divSchemes: Divergence schemes (default: {
                "default": "none",
                "div(phi,U)": "bounded Gauss upwind",
                "turbulence": "bounded Gauss upwind",
                "div(phi,k)": "$turbulence",
                "div(phi,epsilon)": "$turbulence",
                "div((nuEff*dev2(T(grad(U)))))": "Gauss linear"
            }).
            laplacianSchemes: Laplacian schemes (default: {"default": "Gauss linear corrected"}).
            interpolationSchemes: Interpolation schemes (default: {"default": "linear"}).
            snGradSchemes: Surface normal gradient schemes (default: {"default": "corrected"}).
        """
        # Initialize schemes with default values if None provided
        if ddtSchemes is None:
            ddtSchemes = {"default": "steadyState"}
        if gradSchemes is None:
            gradSchemes = {"default": "Gauss linear"}
        if divSchemes is None:
            divSchemes = {
                "default": "none",
                "div(phi,U)": "bounded Gauss upwind",
                "turbulence": "bounded Gauss upwind",
                "div(phi,k)": "$turbulence",
                "div(phi,epsilon)": "$turbulence",
                "div((nuEff*dev2(T(grad(U)))))": "Gauss linear"
            }
        if laplacianSchemes is None:
            laplacianSchemes = {"default": "Gauss linear corrected"}
        if interpolationSchemes is None:
            interpolationSchemes = {"default": "linear"}
        if snGradSchemes is None:
            snGradSchemes = {"default": "corrected"}

        # Call parent class constructor with all schemes
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
        """
        Convert the schemes configuration to a dictionary.
        
        Returns:
            dict: A dictionary containing all schemes with their current configuration.
                  The dictionary structure matches the OpenFOAM fvSchemes format.
        """
        return {
            'ddtSchemes': self.ddtSchemes,
            'gradSchemes': self.gradSchemes,
            'divSchemes': self.divSchemes,
            'laplacianSchemes': self.laplacianSchemes,
            'interpolationSchemes': self.interpolationSchemes,
            'snGradSchemes': self.snGradSchemes
        }

    @classmethod
    def from_dict(cls, config):
        """
        Create a FvSchemesFile instance from a configuration dictionary.
        
        This class method allows creating a FvSchemesFile instance by providing a dictionary
        with schemes configuration. Missing schemes will use empty dictionaries.

        Args:
            config (dict): Dictionary containing schemes configuration. Possible keys:
                - ddtSchemes: Time derivative schemes
                - gradSchemes: Gradient schemes
                - divSchemes: Divergence schemes
                - laplacianSchemes: Laplacian schemes
                - interpolationSchemes: Interpolation schemes
                - snGradSchemes: Surface normal gradient schemes
                
        Returns:
            FvSchemesFile: A new instance initialized with the provided schemes.
        """
        # Get each scheme from config or use empty dict if not provided
        ddtSchemes = config.get('ddtSchemes', {})
        gradSchemes = config.get('gradSchemes', {})
        divSchemes = config.get('divSchemes', {})
        laplacianSchemes = config.get('laplacianSchemes', {})
        interpolationSchemes = config.get('interpolationSchemes', {})
        snGradSchemes = config.get('snGradSchemes', {})

        # Create and return new instance
        return cls(
            ddtSchemes=ddtSchemes,
            gradSchemes=gradSchemes,
            divSchemes=divSchemes,
            laplacianSchemes=laplacianSchemes,
            interpolationSchemes=interpolationSchemes,
            snGradSchemes=snGradSchemes
        )