
# foampilot/constant/pRefFile.py

from foampilot.utilities.manageunits import Quantity
from foampilot.base.openFOAMFile import OpenFOAMFile
from pathlib import Path




class PRefFile(OpenFOAMFile):
    """
    Represents the reference pressure file 'constant/pRef' in OpenFOAM.
    Supports a single Quantity value.
    """

    def __init__(self, value: Quantity = None, dimensions="[1 -1 -2 0 0 0 0]"):
        """
        Initialize PRefFile.

        Args:
            value (Quantity): Reference pressure (default = 1e5 Pa).
            dimensions (str): Dimensions string for pressure.
        """
        if value is None:
            value = Quantity(1e5, "Pa")

        super().__init__(
            object_name="pRef",
            dimensions=dimensions,
            value=value
        )
        self.header["class"] = "uniformDimensionedScalarField"
        self.header["location"] = "constant"

    def _format_value(self, key, value):
        """
        Override for 'value' since pRef expects a scalar Quantity.
        """
        if key == "value" and isinstance(value, Quantity):
            # toujours en Pascal (Pa) via get_in()
            return format(value.get_in("Pa"), ".15g")
        return super()._format_value(key, value)

    def write(self, filepath: Path):
        """Write the pRef file."""
        self.write_file(filepath)