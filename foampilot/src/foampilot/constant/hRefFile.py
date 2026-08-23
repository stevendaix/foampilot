
# foampilot/constant/hRefFile.py

from foampilot.utilities.manageunits import ValueWithUnit
from foampilot.base.openFOAMFile import OpenFOAMFile
from pathlib import Path


class HRefFile(OpenFOAMFile):
    """
    Represents the reference height file ``constant/hRef`` in OpenFOAM.

    This file is used with ``p_rgh`` for hydrostatic pressure handling.
    """

    def __init__(self, value: ValueWithUnit = None):
        """
        Initialize HRefFile.

        Args:
            value (ValueWithUnit): Reference height in meters.
                Defaults to ``0 m`` if not provided.
        """
        if value is None:
            value = ValueWithUnit(0, "m")

        super().__init__(
            object_name="hRef",
            dimensions="[0 1 0 0 0 0 0]",
            value=value,
        )
        self.header["class"] = "uniformDimensionedScalarField"
        self.header["location"] = "constant"

    def _format_value(self, key, value):
        """
        Override value formatting to ensure plain numeric output.
        """
        if key == "value" and isinstance(value, ValueWithUnit):
            return format(value.get_in("m"), ".15g")
        return super()._format_value(key, value)

    def write(self, filepath: Path):
        """Write the hRef file."""
        self.write_file(filepath)
