# constant/TransportPropertiesFile.py

from foampilot.base.openFOAMFile import OpenFOAMFile
from foampilot.utilities.manageunits import Quantity

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
    from foampilot.utilities.manageunits import Quantity

    # Default Newtonian fluid with nu = 1e-05 m²/s
    transport = TransportPropertiesFile()

    # Custom viscosity using Quantity
    transport = TransportPropertiesFile(nu=Quantity(5e-6, "m^2/s"))
    ```

    Notes
    -----
    - This class inherits from :class:`foampilot.base.openFOAMFile.OpenFOAMFile`.
    - Parameters are stored in the OpenFOAM dictionary format.
    - The kinematic viscosity can be provided either as a string or as a Quantity object.
      If provided as a Quantity, it must have units compatible with m²/s.
    """

    def __init__(self, transportModel: str = "Newtonian", nu: str | Quantity = "1e-05"):
        """
        Initialize a `transportProperties` file.

        Parameters
        ----------
        transportModel : str, optional
            The transport model used in the simulation (default: `"Newtonian"`).
        nu : Union[str, Quantity], optional
            The kinematic viscosity. Can be either:
            - A string representing the value in m²/s (default: `"1e-05"`)
            - A Quantity object with units compatible with m²/s

        Attributes
        ----------
        object_name : str
            The OpenFOAM dictionary name (`"transportProperties"`).

        Raises
        ------
        ValueError
            If a Quantity is provided with incompatible units.
        """
        # Handle Quantity object if provided
        if isinstance(nu, Quantity):
            # Check if the units are compatible with m²/s
            if not nu.quantity.check('[length]^2/[time]'):
                raise ValueError("Kinematic viscosity must have units compatible with m²/s")
            # Convert to numeric value in m²/s
            nu = float(nu.get_in('m^2/s'))

        super().__init__(
            object_name="transportProperties",
            transportModel=transportModel,
            nu=nu
        )
