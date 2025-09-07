# constant/TransportPropertiesFile.py

from foampilot.base.openFOAMFile import OpenFOAMFile

class TransportPropertiesFile(OpenFOAMFile):
    """
    Represents the OpenFOAM `transportProperties` configuration file.

    This file defines the transport model and fluid properties, such as
    viscosity, required for CFD simulations.

    Examples
    --------
    Create a Newtonian transport model with kinematic viscosity:

    ```python
    from foampilot.constant.TransportPropertiesFile import TransportPropertiesFile

    # Default Newtonian fluid with nu = 1e-05
    transport = TransportPropertiesFile()

    # Custom viscosity
    transport = TransportPropertiesFile(nu="5e-06")
    ```

    Notes
    -----
    - This class inherits from :class:`foampilot.base.openFOAMFile.OpenFOAMFile`.
    - Parameters are stored in the OpenFOAM dictionary format.
    """

    def __init__(self, transportModel: str = "Newtonian", nu: str = "1e-05"):
        """
        Initialize a `transportProperties` file.

        Parameters
        ----------
        transportModel : str, optional
            The transport model used in the simulation (default: `"Newtonian"`).
        nu : str, optional
            The kinematic viscosity (m²/s) as a string (default: `"1e-05"`).

        Attributes
        ----------
        object_name : str
            The OpenFOAM dictionary name (`"transportProperties"`).
        """
        super().__init__(
            object_name="transportProperties",
            transportModel=transportModel,
            nu=nu
        )