# foampilot/constant/gravityFile.py

from foampilot.utilities.manageunits import Quantity
from foampilot.utilities.openfoamfile import OpenFOAMFile


class GravityFile(OpenFOAMFile):
    """
    Represents the gravity vector file 'constant/g' in OpenFOAM.
    Supports:
    - a tuple of 3 Quantity values
    - a single Quantity + axis ("x", "y", "z")
    """

    def __init__(self, value=None, axis=None, dimensions="[0 1 -2 0 0 0 0]"):
        """
        Initialize GravityFile.

        Args:
            value (Quantity | tuple[Quantity]): Gravity value(s).
            axis (str | None): Axis for single Quantity ("x", "y", "z").
            dimensions (str): OpenFOAM dimensions string.
        """
        if isinstance(value, tuple):
            # tuple de 3 Quantity
            vec = value
        elif isinstance(value, Quantity) and axis:
            # single Quantity + axis
            if axis.lower() == "x":
                vec = (value, Quantity(0, str(value.quantity.units)), Quantity(0, str(value.quantity.units)))
            elif axis.lower() == "y":
                vec = (Quantity(0, str(value.quantity.units)), value, Quantity(0, str(value.quantity.units)))
            elif axis.lower() == "z":
                vec = (Quantity(0, str(value.quantity.units)), Quantity(0, str(value.quantity.units)), value)
            else:
                raise ValueError("axis must be 'x', 'y' or 'z'")
        else:
            # valeur par défaut : g en Y (en m/s^2)
            vec = (
                Quantity(0, "m/s^2"),
                Quantity(-9.81, "m/s^2"),
                Quantity(0, "m/s^2"),
            )

        super().__init__(
            object_name="g",
            dimensions=dimensions,
            value=vec
        )
        self.header["class"] = "uniformDimensionedVectorField"
        self.header["location"] = "constant"

    def _format_value(self, key, value):
        """
        Override for 'value' since gravity is a vector of Quantities.
        """
        if key == "value":
            if isinstance(value, tuple) and all(isinstance(v, Quantity) for v in value):
                unit = "m/s^2"
                vals = [format(v.get_in(unit), ".15g") for v in value]
                return f"({vals[0]} {vals[1]} {vals[2]})"
        return super()._format_value(key, value)