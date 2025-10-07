from pathlib import Path
from foampilot.base.openFOAMFile import OpenFOAMFile
import re
from foampilot.utilities.manageunits import Quantity
import warnings
from typing import Dict, Optional, List, Union

class Boundary:
    """
    A class to handle boundary conditions for OpenFOAM simulations.

    This class provides methods to set, manage, and write boundary conditions for various fields
    (velocity, pressure, turbulence parameters) in OpenFOAM cases. It supports different types of
    boundary conditions including inlets, outlets, walls, and symmetry planes.

    Attributes:
        parent: The parent OpenFOAM case object (with OpenFOAMVariables dataclass).
        turbulence_model: The turbulence model being used (default: "kEpsilon").
        fields: A dictionary containing boundary conditions for each active field.
    """

    def __init__(self, parent, turbulence_model="kEpsilon"):
        """
        Initialize the Boundary class.

        Args:
            parent: The parent OpenFOAM case object (must have a 'variables' attribute).
            turbulence_model: The turbulence model to use (default: "kEpsilon").
        """
        self.parent = parent
        self.turbulence_model = turbulence_model
        self.fields = {}  # Will be initialized with active fields only

    def load_boundary_names(self, case_path: Path) -> Dict[str, str]:
        """
        Load boundary names and types from the polyMesh/boundary file.

        Args:
            case_path: Path to the OpenFOAM case directory.

        Returns:
            A dictionary mapping patch names to their types.

        Raises:
            FileNotFoundError: If the boundary file doesn't exist.
        """
        boundary_file = case_path / 'constant' / 'polyMesh' / 'boundary'
        if not boundary_file.exists():
            raise FileNotFoundError(f"File {boundary_file} not found.")

        with open(boundary_file, 'r') as f:
            content = f.read()

        pattern = re.compile(r'(\w+)\s*\{\s*[^}]*?type\s+(\w+);', re.DOTALL)
        patches = dict(pattern.findall(content))

        exclude = {'FoamFile', 'format', 'class', 'location', 'object'}
        return {k: v for k, v in patches.items() if k not in exclude}

    def is_field_active(self, field_name: str) -> bool:
        """
        Check if a field is active according to the parent's variables dataclass.

        Args:
            field_name: Name of the field to check.

        Returns:
            True if the field is active, False otherwise.
        """
        if not hasattr(self.parent, 'variables'):
            return False
        if not hasattr(self.parent.variables, field_name):
            return False
        return getattr(self.parent.variables, field_name).active

    def get_default_value(self, field_name: str, unit: Optional[str] = None) -> Optional[Quantity]:
        """
        Get the default value for a field from the parent's variables dataclass.

        Args:
            field_name: Name of the field.
            unit: Optional target unit for conversion.

        Returns:
            The default value as a Quantity, or None if not set.
        """
        if not hasattr(self.parent, 'variables'):
            return None
        if not hasattr(self.parent.variables, field_name):
            return None
        field = getattr(self.parent.variables, field_name)
        return field.get_default_value(unit)

    def initialize_boundary(self):
        """
        Initialize boundary fields with patch names and automatically apply wall conditions.
        Only initializes fields that are active in the parent's variables dataclass.
        """
        case_path = Path(self.parent.case_path)
        patch_types = self.load_boundary_names(case_path)

        # Warn about patches of type 'patch' which might need manual configuration
        patch_type_to_warn = "patch"
        patches_found = [name for name, ptype in patch_types.items() if ptype == patch_type_to_warn]
        if patches_found:
            warnings.warn(
                f"Warning: The following patches are of type '{patch_type_to_warn}': {patches_found}. "
                "Please verify that their boundary conditions are properly defined."
            )

        # Initialize only active fields
        self.fields = {}
        for field_name in ["U", "p", "k", "epsilon", "nut", "T", "alpha", "phi"]:
            if self.is_field_active(field_name):
                self.fields[field_name] = {name: {} for name in patch_types}

        # Automatically apply wall conditions for wall-type patches
        for patch_name, patch_type in patch_types.items():
            if patch_type == "wall":
                self.set_wall(patch_name)
            elif patch_type == "empty":
                self.set_symmetry(patch_name)

    def apply_condition_with_wildcard(self, field: str, pattern: str, condition: dict):
        """
        Apply a condition to all boundaries matching a pattern.

        Args:
            field: The field to apply the condition to.
            pattern: Regular expression pattern to match boundary names.
            condition: Dictionary containing the boundary condition settings.
        """
        if field not in self.fields:
            raise ValueError(f"Field '{field}' is not active or not initialized.")

        for boundary in self.fields[field].keys():
            if re.match(pattern, boundary):
                self.fields[field][boundary] = condition

    def set_velocity_inlet(self, pattern: str, velocity: List[Quantity], turbulence_intensity: Optional[float] = None):
        """
        Set a velocity inlet boundary condition.

        Args:
            pattern: Regular expression pattern to match boundary names.
            velocity: Tuple of 3 Quantity objects (u, v, w) representing velocity components.
            turbulence_intensity: Turbulence intensity value between 0 and 1 (optional).

        Raises:
            ValueError: If velocity components don't have correct units.
            RuntimeError: If velocity field is not active.
        """
        if not self.is_field_active("U"):
            raise RuntimeError("Velocity field (U) is not active for this solver.")

        u, v, w = velocity

        # Verify each component has velocity units
        for comp in (u, v, w):
            if not comp.quantity.check('[length] / [time]'):
                raise ValueError("Each velocity component must have units of length/time.")

        # Format velocity value for OpenFOAM
        velocity_value = f"uniform ({u.get_in('m/s')} {v.get_in('m/s')} {w.get_in('m/s')})"
        condition_U = {"type": "fixedValue", "value": velocity_value}
        self.apply_condition_with_wildcard("U", pattern, condition_U)

        # Pressure: zeroGradient
        if self.is_field_active("p"):
            self.apply_condition_with_wildcard("p", pattern, {"type": "zeroGradient"})

        if self.is_field_active("nut"):
            self.apply_condition_with_wildcard("nut", pattern, {"type": "calculated", "value": f"uniform 0"})

        # Turbulence conditions
        if turbulence_intensity and self.is_field_active("k") and self.is_field_active("epsilon"):
            norm_u = (u.get_in("m/s")**2 + v.get_in("m/s")**2 + w.get_in("m/s")**2) ** 0.5
            k_value = 1.5 * (norm_u * turbulence_intensity) ** 2
            epsilon_value = (k_value ** 1.5) / (0.07 * norm_u)

            self.apply_condition_with_wildcard("k", pattern, {
                "type": "fixedValue",
                "value": f"uniform {k_value}"
            })
            self.apply_condition_with_wildcard("epsilon", pattern, {
                "type": "fixedValue",
                "value": f"uniform {epsilon_value}"
            })
        elif self.is_field_active("k") and self.is_field_active("epsilon"):
            self.apply_condition_with_wildcard("k", pattern, {"type": "zeroGradient"})
            self.apply_condition_with_wildcard("epsilon", pattern, {"type": "zeroGradient"})

    def set_pressure_inlet(self, pattern: str, pressure: Quantity, turbulence_intensity: Optional[float] = None):
        """
        Set a pressure inlet boundary condition.

        Args:
            pattern: Regular expression pattern to match boundary names.
            pressure: Pressure value as a Quantity object.
            turbulence_intensity: Turbulence intensity value between 0 and 1 (optional).

        Raises:
            ValueError: If pressure doesn't have correct units.
            RuntimeError: If pressure field is not active.
        """
        if not self.is_field_active("p"):
            raise RuntimeError("Pressure field (p) is not active for this solver.")

        if not pressure.quantity.check('[mass] / ([length] * [time] ** 2)'):
            raise ValueError("Pressure must have units of mass/(length*time²).")

        # Velocity condition
        if self.is_field_active("U"):
            condition_U = {"type": "zeroGradient"}
            self.apply_condition_with_wildcard("U", pattern, condition_U)

        # Pressure condition
        pressure_value = f"uniform {pressure.get_in('Pa')}"
        condition_p = {"type": "fixedValue", "value": pressure_value}
        self.apply_condition_with_wildcard("p", pattern, condition_p)

        if self.is_field_active("nut"):
            self.apply_condition_with_wildcard("nut", pattern, {"type": "calculated", "value": f"uniform 0"})

        # Turbulence conditions
        if turbulence_intensity and self.is_field_active("k") and self.is_field_active("epsilon"):
            k_value = (1.5 * (pressure.get_in("Pa") * turbulence_intensity) ** 2)
            condition_k = {"type": "fixedValue", "value": f"uniform {k_value}"}
            self.apply_condition_with_wildcard("k", pattern, condition_k)

            epsilon_value = (k_value ** 1.5) / (0.07 * pressure.get_in("Pa"))
            condition_epsilon = {"type": "fixedValue", "value": f"uniform {epsilon_value}"}
            self.apply_condition_with_wildcard("epsilon", pattern, condition_epsilon)
        elif self.is_field_active("k") and self.is_field_active("epsilon"):
            self.apply_condition_with_wildcard("k", pattern, {"type": "zeroGradient"})
            self.apply_condition_with_wildcard("epsilon", pattern, {"type": "zeroGradient"})

    def set_pressure_outlet(self, pattern: str, velocity: List[Quantity]):
        """
        Set a pressure outlet boundary condition.

        Args:
            pattern: Regular expression pattern to match boundary names.
            velocity: Tuple of 3 Quantity objects (u, v, w) representing velocity components.

        Raises:
            RuntimeError: If required fields are not active.
        """
        if not self.is_field_active("U") or not self.is_field_active("p"):
            raise RuntimeError("Velocity (U) and pressure (p) fields must be active for pressure outlet.")

        u, v, w = velocity
        velocity_value = f"uniform ({u.get_in('m/s')} {v.get_in('m/s')} {w.get_in('m/s')})"
        condition_U = {"type": "pressureInletOutletVelocity", "value": velocity_value}
        self.apply_condition_with_wildcard("U", pattern, condition_U)

        condition_p = {"type": "fixedValue", "value": "uniform 0"}
        self.apply_condition_with_wildcard("p", pattern, condition_p)

        if self.is_field_active("k"):
            self.apply_condition_with_wildcard("k", pattern, {"type": "zeroGradient"})
        if self.is_field_active("epsilon"):
            self.apply_condition_with_wildcard("epsilon", pattern, {"type": "zeroGradient"})
        if self.is_field_active("nut"):
            self.apply_condition_with_wildcard("nut", pattern, {"type": "calculated", "value": f"uniform 0"})

    def set_mass_flow_inlet(self, pattern: str, mass_flow_rate: Quantity, density: Quantity):
        """
        Set a mass flow inlet boundary condition.

        Args:
            pattern: Regular expression pattern to match boundary names.
            mass_flow_rate: Mass flow rate as a Quantity object.
            density: Density as a Quantity object.

        Raises:
            ValueError: If units are incorrect.
            RuntimeError: If required fields are not active.
        """
        if not self.is_field_active("U") or not self.is_field_active("p"):
            raise RuntimeError("Velocity (U) and pressure (p) fields must be active for mass flow inlet.")

        if not mass_flow_rate.quantity.check('[mass] / [time]'):
            raise ValueError("Mass flow rate must have units of mass/time.")
        if not density.quantity.check('[mass] / [volume]'):
            raise ValueError("Density must have units of mass/volume.")

        velocity_value = (mass_flow_rate.get_in("kg/s") / density.get_in("kg/m^3"))
        condition_U = {"type": "fixedValue", "value": f"uniform ({velocity_value} 0 0)"}
        self.apply_condition_with_wildcard("U", pattern, condition_U)

        self.apply_condition_with_wildcard("p", pattern, {"type": "zeroGradient"})

        if self.is_field_active("k"):
            self.apply_condition_with_wildcard("k", pattern, {"type": "zeroGradient"})
        if self.is_field_active("epsilon"):
            self.apply_condition_with_wildcard("epsilon", pattern, {"type": "zeroGradient"})
        if self.is_field_active("nut"):
            self.apply_condition_with_wildcard("nut", pattern, {"type": "calculated", "value": f"uniform 0"})

    def set_wall(self, pattern: str, friction: bool = True, velocity: Optional[List[Quantity]] = None):
        """
        Set a wall boundary condition.

        Args:
            pattern: Regular expression pattern to match boundary names.
            friction: If True (default), use no-slip condition. If False, use slip condition.
            velocity: Optional tuple of 3 Quantity objects for fixed velocity wall.

        Raises:
            RuntimeError: If velocity field is not active.
        """
        if not self.is_field_active("U"):
            raise RuntimeError("Velocity field (U) is not active for wall conditions.")

        vel_cond = False
        if velocity is not None:
            u, v, w = velocity
            vel_cond = True
            condition_U = {
                "type": "fixedValue",
                "value": f"uniform ({u.get_in('m/s')} {v.get_in('m/s')} {w.get_in('m/s')})"
            }
        elif not friction:
            condition_U = {"type": "slip"}
        else:
            condition_U = {"type": "noSlip"}

        self.apply_condition_with_wildcard("U", pattern, condition_U)

        if self.is_field_active("p"):
            self.apply_condition_with_wildcard("p", pattern, {"type": "zeroGradient"})

        for field in ["k", "epsilon", "omega"]:
            if self.is_field_active(field):
                self.apply_condition_with_wildcard(
                    field, pattern, self.get_wall_function(field, vel_cond=vel_cond)
                )

        if self.is_field_active("nut"):
            self.apply_condition_with_wildcard("nut", pattern, {
                "type": "nutkWallFunction",
                "Cmu": 0.09,
                "kappa": 0.41,
                "E": 9.8,
                "value": f"uniform 0"
            })

    def set_symmetry(self, pattern: str):
        """
        Set a symmetry (empty) boundary condition.

        Args:
            pattern: Regular expression pattern to match boundary names.
        """
        empty_condition = {"type": "empty"}
        for field in self.fields.keys():  # Only apply to active fields
            self.apply_condition_with_wildcard(field, pattern, empty_condition)

    def set_no_friction_wall(self, pattern: str):
        """
        Set a no-friction (slip) wall boundary condition.

        Args:
            pattern: Regular expression pattern to match boundary names.

        Raises:
            RuntimeError: If required fields are not active.
        """
        if not self.is_field_active("U") or not self.is_field_active("p"):
            raise RuntimeError("Velocity (U) and pressure (p) fields must be active for no-friction wall.")

        condition_U = {"type": "slip"}
        self.apply_condition_with_wildcard("U", pattern, condition_U)
        self.apply_condition_with_wildcard("p", pattern, {"type": "zeroGradient"})

        if self.is_field_active("k"):
            self.apply_condition_with_wildcard("k", pattern, {"type": "zeroGradient"})
        if self.is_field_active("epsilon"):
            self.apply_condition_with_wildcard("epsilon", pattern, {"type": "zeroGradient"})
        if self.is_field_active("nut"):
            self.apply_condition_with_wildcard("nut", pattern, {
                "type": "nutkWallFunction",
                "Cmu": 0.09,
                "kappa": 0.41,
                "E": 9.8,
                "value": f"uniform 0"
            })

    def get_wall_function(self, field: str, vel_cond: bool) -> dict:
        """
        Get the appropriate wall function for a given field.

        Args:
            field: The field name ("k", "epsilon", or "omega").
            vel_cond: Whether velocity condition is applied.

        Returns:
            A dictionary containing the wall function settings.
        """
        if field == "epsilon":
            return {
                "type": "epsilonWallFunction",
                "value": "uniform 0",
                "Cmu": 0.09,
                "kappa": 0.41,
                "E": 9.8
            }
        elif field == "k":
            value = 0.375 if vel_cond else 0
            return {
                "type": "kqRWallFunction",
                "value": f"uniform {value}"
            }
        elif field == "omega":
            return {
                "type": "omegaWallFunction",
                "value": "uniform 0"
            }
        else:
            return {"type": "zeroGradient"}

    def set_uniform_normal_fixed_value_all_fields(self, patch_pattern: str, mode: str = "intakeType3", ref_value: float = 1.2):
        """
        Set uniformNormalFixedValue or surfaceNormalFixedValue condition on all active fields.

        Args:
            patch_pattern: Regular expression pattern to match boundary names.
            mode: Type of condition to apply.
            ref_value: Reference value for the condition.

        Raises:
            ValueError: If invalid mode is specified.
        """
        # Velocity condition based on mode
        if mode == "intakeType1":
            condition_U = {
                "type": "surfaceNormalFixedValue",
                "refValue": f"uniform {ref_value}",
                "ramp": "table ((0 0) (10 1))"
            }
        elif mode == "intakeType2":
            condition_U = {
                "type": "uniformNormalFixedValue",
                "uniformValue": f"table ((0 0) (10 {ref_value}))"
            }
        elif mode == "intakeType3":
            condition_U = {
                "type": "uniformNormalFixedValue",
                "uniformValue": f"constant {ref_value}",
                "ramp": "table ((0 0) (10 1))"
            }
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose from 'intakeType1', 'intakeType2', 'intakeType3'.")

        # Apply to active fields only
        if self.is_field_active("U"):
            self.apply_condition_with_wildcard("U", patch_pattern, condition_U)
        if self.is_field_active("p"):
            self.apply_condition_with_wildcard("p", patch_pattern, {"type": "zeroGradient"})
        if self.is_field_active("k"):
            self.apply_condition_with_wildcard("k", patch_pattern, {"type": "zeroGradient"})
        if self.is_field_active("epsilon"):
            self.apply_condition_with_wildcard("epsilon", patch_pattern, {"type": "zeroGradient"})
        if self.is_field_active("nut"):
            self.apply_condition_with_wildcard("nut", patch_pattern, {
                "type": "calculated",
                "value": "uniform 0"
            })

    def _format_value_for_field(self, field_name: str, value: Union[Quantity, float, List, tuple]) -> str:
        """
        Format a value for writing to an OpenFOAM field file.

        Args:
            field_name: Name of the field.
            value: Value to format (Quantity, float, list, or tuple).

        Returns:
            Formatted string representation of the value.
        """
        if isinstance(value, Quantity):
            unit = OpenFOAMFile.DEFAULT_UNITS.get(field_name, None)
            if unit:
                return format(value.get_in(unit), ".15g")
            return format(float(value.quantity.magnitude), ".15g")

        if isinstance(value, (tuple, list)):
            parts = []
            for v in value:
                if isinstance(v, Quantity):
                    unit = OpenFOAMFile.DEFAULT_UNITS.get(field_name, None)
                    if unit:
                        parts.append(format(v.get_in(unit), ".15g"))
                    else:
                        parts.append(format(float(v.quantity.magnitude), ".15g"))
                else:
                    parts.append(format(float(v), ".15g"))
            return f"({' '.join(parts)})"

        if isinstance(value, bool):
            return "true" if value else "false"

        return format(float(value), ".15g")

    def set_temperature_inlet(self, pattern: str, temperature: Union[Quantity, float], bc_type: str = "fixedValue"):
        """
        Set a temperature inlet boundary condition.

        Args:
            pattern: Regular expression pattern to match boundary names.
            temperature: Temperature value as a Quantity or float (in Kelvin or Celsius).
            bc_type: Boundary condition type (default: "fixedValue").

        Raises:
            RuntimeError: If temperature field is not active.
        """
        if not self.is_field_active("T"):
            raise RuntimeError("Temperature field (T) is not active.")

        if not isinstance(temperature, Quantity):
            temperature = Quantity(float(temperature), "K")
        else:
            if temperature.quantity.units == "degC":
                temperature = Quantity(temperature.get_in("degC") + 273.15, "K")

        T_val = temperature.get_in("K")
        cond = {"type": bc_type, "value": f"uniform {format(T_val, '.15g')}"}
        self.apply_condition_with_wildcard("T", pattern, cond)

    def set_temperature_wall(self, pattern: str, temperature: Union[Quantity, float], bc_type: str = "fixedValue"):
        """
        Set a temperature wall boundary condition.

        Args:
            pattern: Regular expression pattern to match boundary names.
            temperature: Temperature value as a Quantity or float (in Kelvin or Celsius).
            bc_type: Boundary condition type (default: "fixedValue").

        Raises:
            RuntimeError: If temperature field is not active.
        """
        self.set_temperature_inlet(pattern, temperature, bc_type=bc_type)

    def set_temperature_flux(self, pattern: str, heat_flux: Union[Quantity, float], bc_type: str = "externalWallHeatFluxTemperature"):
        """
        Set a temperature flux boundary condition.

        Args:
            pattern: Regular expression pattern to match boundary names.
            heat_flux: Heat flux value as a Quantity or float (in W/m²).
            bc_type: Boundary condition type (default: "externalWallHeatFluxTemperature").

        Raises:
            RuntimeError: If temperature field is not active.
        """
        if not self.is_field_active("T"):
            raise RuntimeError("Temperature field (T) is not active.")

        if not isinstance(heat_flux, Quantity):
            heat_flux = Quantity(float(heat_flux), "W/m^2")

        q = heat_flux.get_in("W/m^2")
        cond = {"type": bc_type, "value": f"uniform {format(q, '.15g')}"}
        self.apply_condition_with_wildcard("T", pattern, cond)

    def write_boundary_file(self, field: str):
        """
        Write the boundary condition file for a specific field.

        Args:
            field: Name of the field to write.

        Raises:
            ValueError: If the field is not active.
        """
        if not self.is_field_active(field):
            raise ValueError(f"Field '{field}' is not active.")

        base_path = Path(self.parent.case_path)
        system_path = base_path / "0"
        system_path.mkdir(parents=True, exist_ok=True)
        file_path = system_path / field

        with open(file_path, "w") as file:
            # === Header ===
            file.write(self.generate_header(field))

            # === Dimensions ===
            unit = OpenFOAMFile.DEFAULT_UNITS.get(field, None)
            if unit:
                q = Quantity(1.0, unit)
                dim_vector = q.get_dimensions_vector()
            else:
                dim_vector = "[0 0 0 0 0 0 0]"
            file.write(f"\ndimensions      {dim_vector};\n")

            # === Internal field ===
            default_value = self.get_default_value(field)
            if default_value is not None:
                if field == "U":
                    internal_field = f"uniform ({default_value.quantity.magnitude[0]} {default_value.quantity.magnitude[1]} {default_value.quantity.magnitude[2]})"
                else:
                    internal_field = f"uniform {default_value.quantity.magnitude}"
            else:
                internal_field = {
                    "U": "uniform (0 0 0)",
                    "epsilon": "uniform 0.125",
                    "k": "uniform 0.375",
                    "T": "uniform 300",
                }.get(field, "uniform 0")
            file.write(f"internalField   {internal_field};\n\n")

            # === Boundary field ===
            file.write("boundaryField\n{\n")
            for boundary, conditions in self.fields[field].items():
                if conditions:
                    file.write(f" {boundary}\n {{\n")
                    for key, value in conditions.items():
                        if key in ("value", "uniformValue", "refValue"):
                            if isinstance(value, str) and (
                                value.strip().startswith("uniform")
                                or value.strip().startswith("(")
                            ):
                                file.write(f" {key:<15} {value};\n")
                            else:
                                formatted = self._format_value_for_field(field, value)
                                file.write(f" {key:<15} uniform {formatted};\n")
                        else:
                            file.write(f" {key:<15} {value};\n")
                    file.write(" }\n\n")
            file.write("}\n\n")

            # === Footer ===
            file.write("// ************************************************************************* //\n")

    def generate_header(self, field: str) -> str:
        """
        Generate the header for an OpenFOAM field file.

        Args:
            field: Name of the field.

        Returns:
            The file header as a string.
        """
        if field == "U":
            class_field = "volVectorField"
        else:
            class_field = "volScalarField"
        return (
            f"FoamFile\n"
            f"{{\n"
            f" version 2.0;\n"
            f" format ascii;\n"
            f" class {class_field};\n"
            f" object {field};\n"
            f"}}\n"
        )
