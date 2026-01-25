from foampilot.utilities.manageunits import ValueWithUnit
from foampilot.base.openFOAMFile import OpenFOAMFile
from pathlib import Path
from typing import Optional, Any, Union, Tuple

class GravityFile(OpenFOAMFile):
    """
    Represents the gravity vector file 'constant/g' in OpenFOAM.
    Automatically uses the parent's case_path and supports dynamic configuration.

    Features:
    - Uses parent.case_path for file location
    - Supports dynamic gravity configuration via CaseFieldsManager
    - Validates units and provides flexible input (vector or axis-based)
    - Maintains backward compatibility
    """

    def __init__(
        self,
        parent: Any,
        value: Optional[Union[ValueWithUnit, Tuple[ValueWithUnit, ValueWithUnit, ValueWithUnit]]] = None,
        axis: Optional[str] = None,
        dimensions: str = "[0 1 -2 0 0 0 0]"
    ):
        """
        Initialize GravityFile using parent's case_path.

        Args:
            parent (Any): Parent object with case_path and optional fields_manager.
            value (ValueWithUnit | tuple[ValueWithUnit], optional): Gravity value(s).
            axis (str, optional): Axis for single ValueWithUnit ("x", "y", or "z").
            dimensions (str): OpenFOAM dimensions string.
        """
        self.parent = parent
        self.dimensions = dimensions

        # Set default gravity vector (0 -9.81 0) if no value provided
        if value is None and axis is None:
            vec = (
                ValueWithUnit(0, "m/s^2"),
                ValueWithUnit(-9.81, "m/s^2"),
                ValueWithUnit(0, "m/s^2"),
            )
        elif isinstance(value, tuple):
            self._validate_ValueWithUnit_tuple(value)
            vec = value
        elif isinstance(value, ValueWithUnit) and axis:
            vec = self._create_vector_from_axis(value, axis)
        else:
            vec = (
                ValueWithUnit(0, "m/s^2"),
                ValueWithUnit(-9.81, "m/s^2"),
                ValueWithUnit(0, "m/s^2"),
            )

        # Override with dynamic configuration if parent has CaseFieldsManager
        if hasattr(self.parent, "fields_manager"):
            vec = self._configure_from_fields(vec)

        super().__init__(
            object_name="g",
            dimensions=self.dimensions,
            value=vec
        )
        self.header["class"] = "uniformDimensionedVectorField"
        self.header["location"] = "constant"

    def _validate_ValueWithUnit_tuple(self, vec: Tuple[ValueWithUnit, ValueWithUnit, ValueWithUnit]):
        """Validate that all elements in the tuple are Quantities with correct units."""
        for i, v in enumerate(vec):
            if not isinstance(v, ValueWithUnit):
                raise ValueError(f"Gravity vector element {i} must be a ValueWithUnit")
            if not v.ValueWithUnit.check('[length]/[time]^2'):
                raise ValueError(f"Gravity element {i} must have units compatible with m/s^2")

    def _create_vector_from_axis(self, value: ValueWithUnit, axis: str) -> Tuple[ValueWithUnit, ValueWithUnit, ValueWithUnit]:
        """Create 3D gravity vector from single ValueWithUnit + axis."""
        if not isinstance(value, ValueWithUnit):
            raise ValueError("Value must be a ValueWithUnit when using axis")
        if not value.ValueWithUnit.check('[length]/[time]^2'):
            raise ValueError("Gravity value must have units compatible with m/s^2")

        axis = axis.lower()
        unit = str(value.ValueWithUnit.units)
        if axis == "x":
            return (value, ValueWithUnit(0, unit), ValueWithUnit(0, unit))
        elif axis == "y":
            return (ValueWithUnit(0, unit), value, ValueWithUnit(0, unit))
        elif axis == "z":
            return (ValueWithUnit(0, unit), ValueWithUnit(0, unit), value)
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")

    def _configure_from_fields(self, default_vec: Tuple[ValueWithUnit, ValueWithUnit, ValueWithUnit]) -> Tuple[ValueWithUnit, ValueWithUnit, ValueWithUnit]:
        """Adjust gravity vector based on available fields."""
        fields_manager = self.parent.fields_manager
        field_names = fields_manager.get_field_names()

        # Detect rotating cases (MRF/omega) and adjust gravity if needed
        if "omega" in field_names or "MRF" in field_names:
            return default_vec  # Could add centrifugal components here

        # Detect atmospheric cases (adjust gravity direction)
        if hasattr(self.parent, "simulation_type"):
            sim_type = getattr(self.parent, "simulation_type", "").lower()
            if sim_type == "atmospheric":
                return (
                    ValueWithUnit(0, "m/s^2"),
                    ValueWithUnit(0, "m/s^2"),
                    ValueWithUnit(-9.81, "m/s^2"),  # Z-direction for atmospheric
                )

        return default_vec

    def write(self):
        """Write gravity file to parent's constant directory."""
        case_path = Path(getattr(self.parent, "case_path", "."))
        file_path = case_path / "constant" / self.object_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_file(file_path)
        print(f"✅ Gravity file written to {file_path}")
