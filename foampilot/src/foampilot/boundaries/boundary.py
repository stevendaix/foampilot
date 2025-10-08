
from pathlib import Path
import re
import warnings
from foampilot.base.openFOAMFile import OpenFOAMFile
from foampilot.utilities.manageunits import Quantity
from .boundary_conditions_config import BOUNDARY_CONDITIONS_CONFIG, WALL_FUNCTIONS, CONDITION_CALCULATORS

class Boundary:
    """
    A class to handle boundary conditions for OpenFOAM simulations using a configuration-driven approach.
    
    This class provides a flexible way to set, manage, and write boundary conditions for various fields
    in OpenFOAM cases. It supports different types of boundary conditions and turbulence models through an
    external configuration file.
    
    Attributes:
        parent: The parent OpenFOAM case object.
        turbulence_model: The turbulence model being used (e.g., "kEpsilon").
        fields: A dictionary containing boundary conditions for each field.
        config: The configuration for the selected turbulence model.
    """
    
    def __init__(self, parent, fields, turbulence_model="kEpsilon"):
        """
        Initialize the Boundary class.
        
        Args:
            parent: The parent OpenFOAM case object.
            fields: A list of fields to manage (e.g., ["U", "p", "T"]).
            turbulence_model: The turbulence model to use (default: "kEpsilon").
        """
        self.parent = parent
        self.turbulence_model = turbulence_model
        self.fields = {field: {} for field in fields}
        self.config = BOUNDARY_CONDITIONS_CONFIG.get(self.turbulence_model)
        if not self.config:
            raise ValueError(f"Turbulence model '{self.turbulence_model}' is not supported.")

    def load_boundary_names(self, case_path: Path) -> dict[str, str]:
        """
        Load boundary names and types from the polyMesh/boundary file.
        
        Args:
            case_path: Path to the OpenFOAM case directory.
            
        Returns:
            A dictionary mapping patch names to their types.
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

    def initialize_boundary(self):
        """
        Initialize boundary fields with patch names and automatically apply wall/symmetry conditions.
        """
        case_path = Path(self.parent.case_path)
        patch_types = self.load_boundary_names(case_path)

        patch_type_to_warn = "patch"
        patches_found = [name for name, ptype in patch_types.items() if ptype == patch_type_to_warn]

        if patches_found:
            warnings.warn(
                f"Warning: The following patches are of type '{patch_type_to_warn}': {patches_found}. "
                "Please verify that their boundary conditions are properly defined."
            )
            
        for field in self.fields:
            self.fields[field] = {name: {} for name in patch_types}

        for patch_name, patch_type in patch_types.items():
            if patch_type == "wall":
                self.set_condition(patch_name, "wall", friction=True)
            elif patch_type == "empty":
                self.set_condition(patch_name, "symmetry")

    def apply_condition_with_wildcard(self, pattern, condition_type, **kwargs):
        """
        Apply a condition to all boundaries matching a pattern.
        
        Args:
            pattern: Regular expression pattern to match boundary names.
            condition_type: The type of condition to apply (e.g., "velocityInlet").
            **kwargs: Arguments for the condition (e.g., velocity, pressure).
        """
        for boundary in self.fields[next(iter(self.fields))].keys():
            if re.match(pattern, boundary):
                self.set_condition(boundary, condition_type, **kwargs)

    def set_condition(self, boundary_name, condition_type, **kwargs):
        """
        Set a boundary condition based on the configuration.

        Args:
            boundary_name: The name of the boundary to apply the condition to.
            condition_type: The type of condition (e.g., "velocityInlet").
            **kwargs: Arguments for the condition.
        """
        condition_config = self.config.get(condition_type)
        if not condition_config:
            raise ValueError(f"Condition type '{condition_type}' is not defined for model '{self.turbulence_model}'.")

        calculator = CONDITION_CALCULATORS.get(condition_type)
        if calculator:
            if not calculator["validate"](**kwargs):
                raise ValueError(calculator["error_message"])
            calculated_params = calculator["calculate"](**kwargs)
            kwargs.update(calculated_params)

        for field, field_config in condition_config.items():
            if field in self.fields:
                final_config = self._resolve_field_config(field_config, kwargs)
                if final_config:
                    self.fields[field][boundary_name] = self._format_config(final_config, kwargs)

    def _resolve_field_config(self, field_config, kwargs):
        """
        Resolve the specific configuration for a field based on provided arguments.
        """
        if "type" in field_config and field_config["type"] == "wallFunction":
            wall_func_conf = WALL_FUNCTIONS[self.turbulence_model][field_config["function"]]
            if kwargs.get("velocity"):
                return wall_func_conf.get("fixedValue", wall_func_conf.get("default"))
            else:
                return wall_func_conf.get("noSlip", wall_func_conf.get("default"))
        
        # Handle conditional configs (e.g., withTurbulence)
        if kwargs.get("turbulence_intensity") and "withTurbulence" in field_config:
            return field_config["withTurbulence"]
        elif "default" in field_config:
            return field_config["default"]
        
        # Handle friction for walls
        if kwargs.get("friction") is False and "slip" in field_config:
            return field_config["slip"]
        elif "noSlip" in field_config:
            return field_config["noSlip"]

        return field_config

    def _format_config(self, config, params):
        """
        Format the configuration dictionary by substituting placeholders with calculated values.
        """
        formatted_config = {}
        for key, value in config.items():
            if isinstance(value, str):
                try:
                    formatted_config[key] = value.format(**params)
                except KeyError as e:
                    # If a placeholder is not found, keep it as is for OpenFOAM to interpret
                    formatted_config[key] = value
            else:
                formatted_config[key] = value
        return formatted_config

    def write_boundary_conditions(self):
        """
        Write the boundary conditions to their respective files in the 0/ directory.
        """
        for field, boundaries in self.fields.items():
            file_path = self.parent.case_path / "0" / field
            foam_file = OpenFOAMFile(file_path)
            foam_file.set_boundary_field(boundaries)
            foam_file.write()

# Example Usage (for demonstration)
if __name__ == '__main__':
    class MockParent:
        def __init__(self, case_path):
            self.case_path = Path(case_path)

    # Create a dummy case structure
    dummy_case = Path("./dummy_case")
    (dummy_case / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
    (dummy_case / "0").mkdir(exist_ok=True)

    with open(dummy_case / "constant" / "polyMesh" / "boundary", "w") as f:
        f.write("""
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       polyBoundaryMesh;
            location    "constant/polyMesh";
            object      boundary;
        }
        (
            inlet
            {
                type            patch;
                nFaces          20;
                startFace       760;
            }
            outlet
            {
                type            patch;
                nFaces          20;
                startFace       780;
            }
            walls
            {
                type            wall;
                nFaces          400;
                startFace       800;
            }
        )
        """)

    # --- kEpsilon Example ---
    print("--- Running kEpsilon Example ---")
    parent_case = MockParent(dummy_case)
    fields_to_manage = ["U", "p", "k", "epsilon", "nut"]
    boundary_manager = Boundary(parent_case, fields=fields_to_manage, turbulence_model="kEpsilon")
    boundary_manager.initialize_boundary()

    velocity_in = (Quantity(10, "m/s"), Quantity(0, "m/s"), Quantity(0, "m/s"))
    boundary_manager.apply_condition_with_wildcard("inlet", "velocityInlet", velocity=velocity_in, turbulence_intensity=0.05)
    
    velocity_out = (Quantity(0, "m/s"), Quantity(0, "m/s"), Quantity(0, "m/s"))
    boundary_manager.apply_condition_with_wildcard("outlet", "pressureOutlet", velocity=velocity_out)

    boundary_manager.write_boundary_conditions()

    # Verify output
    with open(dummy_case / "0" / "U", "r") as f:
        print("--- U file ---")
        print(f.read())

    # --- kOmegaSST Example ---
    print("\n--- Running kOmegaSST Example ---")
    fields_to_manage_sst = ["U", "p", "k", "omega", "nut"]
    boundary_manager_sst = Boundary(parent_case, fields=fields_to_manage_sst, turbulence_model="kOmegaSST")
    boundary_manager_sst.initialize_boundary()

    boundary_manager_sst.apply_condition_with_wildcard("inlet", "velocityInlet", velocity=velocity_in, turbulence_intensity=0.05)
    boundary_manager_sst.apply_condition_with_wildcard("outlet", "pressureOutlet", velocity=velocity_out)

    boundary_manager_sst.write_boundary_conditions()
    with open(dummy_case / "0" / "omega", "r") as f:
        print("--- omega file ---")
        print(f.read())

    # --- Thermal Example ---
    print("\n--- Running Thermal Example ---")
    fields_to_manage_thermal = ["U", "p", "T"]
    boundary_manager_thermal = Boundary(parent_case, fields=fields_to_manage_thermal, turbulence_model="kEpsilon") # Assuming kEpsilon for flow
    boundary_manager_thermal.initialize_boundary()
    # Note: Thermal conditions would need to be added to the config file to be used.
    # For now, this demonstrates that the field 'T' is created.
    boundary_manager_thermal.write_boundary_conditions()
    with open(dummy_case / "0" / "T", "r") as f:
        print("--- T file ---")
        print(f.read())
