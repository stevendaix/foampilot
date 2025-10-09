from foampilot.utilities.manageunits import Quantity
from foampilot.base.openFOAMFile import OpenFOAMFile
from typing import Optional, Any, Union, Tuple

class GravityFile(OpenFOAMFile):
    """
    Represents the gravity vector file 'constant/g' in OpenFOAM.
    Supports dynamic configuration based on:
      - CaseFieldsManager (for field detection)
      - Custom gravity vectors (tuple of 3 Quantities or single Quantity + axis)
      - Automatic detection of gravity axis based on simulation type

    Examples
    --------
    >>> # Default gravity in Y direction (0 -9.81 0)
    >>> gravity = GravityFile()
    >>> gravity.write("./constant")

    >>> # Custom gravity vector
    >>> from foampilot.utilities.manageunits import Quantity
    >>> gravity = GravityFile(
    >>>     value=(
    >>>         Quantity(0, "m/s^2"),
    >>>         Quantity(-10, "m/s^2"),
    >>>         Quantity(0, "m/s^2")
    >>>     )
    >>> )
    >>> gravity.write("./constant")

    >>> # Single Quantity in Z direction
    >>> gravity = GravityFile(value=Quantity(9.81, "m/s^2"), axis="z")
    >>> gravity.write("./constant")
    """

    def __init__(
        self,
        parent: Optional[Any] = None,
        value: Optional[Union[Quantity, Tuple[Quantity, Quantity, Quantity]]] = None,
        axis: Optional[str] = None,
        dimensions: str = "[0 1 -2 0 0 0 0]"
    ):
        """
        Initialize GravityFile.

        Args:
            parent (Any, optional): Parent object with `fields_manager` for dynamic configuration.
            value (Quantity | tuple[Quantity], optional): Gravity value(s).
            axis (str, optional): Axis for single Quantity ("x", "y", or "z").
            dimensions (str, optional): OpenFOAM dimensions string.
        """
        self.parent = parent

        # Set default gravity vector (0 -9.81 0) if no value provided
        if value is None and axis is None:
            vec = (
                Quantity(0, "m/s^2"),
                Quantity(-9.81, "m/s^2"),
                Quantity(0, "m/s^2"),
            )
        elif isinstance(value, tuple):
            # Tuple of 3 Quantities
            self._validate_quantity_tuple(value)
            vec = value
        elif isinstance(value, Quantity) and axis:
            # Single Quantity + axis
            vec = self._create_vector_from_axis(value, axis)
        else:
            # Default if invalid input
            vec = (
                Quantity(0, "m/s^2"),
                Quantity(-9.81, "m/s^2"),
                Quantity(0, "m/s^2"),
            )

        # Override with dynamic configuration if parent has CaseFieldsManager
        if self.parent and hasattr(self.parent, "fields_manager"):
            vec = self._configure_from_fields(vec)

        super().__init__(
            object_name="g",
            dimensions=dimensions,
            value=vec
        )
        self.header["class"] = "uniformDimensionedVectorField"
        self.header["location"] = "constant"

    def _validate_quantity_tuple(self, vec: Tuple[Quantity, Quantity, Quantity]):
        """
        Validate that all elements in the tuple are Quantities with compatible units.
        """
        for i, v in enumerate(vec):
            if not isinstance(v, Quantity):
                raise ValueError(f"Element {i} in gravity vector must be a Quantity")
            if not v.quantity.check('[length]/[time]^2'):
                raise ValueError(f"Gravity element {i} must have units compatible with m/s^2")

    def _create_vector_from_axis(
        self,
        value: Quantity,
        axis: str
    ) -> Tuple[Quantity, Quantity, Quantity]:
        """
        Create a 3D gravity vector from a single Quantity and axis.
        """
        if not isinstance(value, Quantity):
            raise ValueError("Value must be a Quantity when using axis")
        if not value.quantity.check('[length]/[time]^2'):
            raise ValueError("Gravity value must have units compatible with m/s^2")

        axis = axis.lower()
        if axis == "x":
            return (value, Quantity(0, str(value.quantity.units)), Quantity(0, str(value.quantity.units)))
        elif axis == "y":
            return (Quantity(0, str(value.quantity.units)), value, Quantity(0, str(value.quantity.units)))
        elif axis == "z":
            return (Quantity(0, str(value.quantity.units)), Quantity(0, str(value.quantity.units)), value)
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")

    def _configure_from_fields(
        self,
        default_vec: Tuple[Quantity, Quantity, Quantity]
    ) -> Tuple[Quantity, Quantity, Quantity]:
        """
        Override gravity vector based on fields available in CaseFieldsManager.
        """
        fields_manager = self.parent.fields_manager
        field_names = fields_manager.get_field_names()

        # Example: Detect if simulation is for a rotating machine (e.g., turbine)
        # and adjust gravity accordingly (e.g., centrifugal effects)
        if "omega" in field_names or "MRF" in field_names:
            # For rotating cases, gravity might need adjustment
            # Here we keep default but could add centrifugal components
            return default_vec

        # Detect if gravity should be in Z direction (e.g., for atmospheric simulations)
        if self.parent and hasattr(self.parent, "simulation_type"):
            sim_type = getattr(self.parent, "simulation_type", "").lower()
            if sim_type == "atmospheric":
                return (
                    Quantity(0, "m/s^2"),
                    Quantity(0, "m/s^2"),
                    Quantity(-9.81, "m/s^2"),  # Gravity in Z for atmospheric cases
                )

        # Default case: return input vector
        return default_vec

    def _format_value(self, key: str, value: Union[Tuple[Quantity, Quantity, Quantity], Any]):
        """
        Format the gravity vector for OpenFOAM file writing.
        """
        if key == "value" and isinstance(value, tuple):
            unit = "m/s^2"
            vals = [format(v.get_in(unit), ".15g") for v in value]
            return f"({vals[0]} {vals[1]} {vals[2]})"
        return super()._format_value(key, value)
