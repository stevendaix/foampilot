import os
from pathlib import Path
from foampilot.utilities.manageunits import Quantity
from typing import Optional, Any, Union, Tuple


class OpenFOAMFile:
    """A base class for managing and writing OpenFOAM configuration files.

    This class provides a structured way to handle OpenFOAM dictionary syntax,
    including headers, nested dictionaries, and unit conversions for physical
    quantities. It supports both generic dictionary files and specific 
    field files (like 'U' or 'p').

    Attributes:
        DEFAULT_UNITS (dict): Mapping of field names to their standard OpenFOAM units.
        FIELD_DIMENSIONS (dict): Mapping of field names to their SI unit dimension strings.
        header (dict): Standard OpenFOAM 'FoamFile' header information.
        attributes (dict): Data content of the file, stored as nested dictionaries or values.
        object_name (str): The name of the OpenFOAM object (e.g., 'controlDict').
    """

    DEFAULT_UNITS = {
        "nu": "m^2/s", "mu": "Pa.s", "rho": "kg/m^3",
        "k": "m^2/s^2", "epsilon": "m^2/s^3", "omega": "1/s",
        "nut": "m^2/s", "mut": "Pa.s",
        "U": "m/s", "p": "Pa", "T": "K",
        "alpha": "m^2/s", "phi": "m^3/s",
        "g": "m/s^2",
        "Re": None, "Pr": None, "Ma": None, "Fo": None, "yPlus": None,
        "porosity": None, "alpha.water": None, "alpha.air": None,
    }

    FIELD_DIMENSIONS = {
        "U": "[0 1 -1 0 0 0 0]",
        "p": "[0 2 -2 0 0 0 0]",
        "k": "[0 2 -2 0 0 0 0]",
        "epsilon": "[0 2 -3 0 0 0 0]",
        "nut": "[0 2 -1 0 0 0 0]"
    }

    def __init__(self, object_name: str, **attributes: Any):
        """Initializes the OpenFOAMFile with a header and specific attributes.

        Args:
            object_name: The name used in the FoamFile 'object' entry.
            **attributes: Arbitrary keyword arguments representing the 
                dictionary entries.
        """
        self.header = {
            "version": "2.0",
            "format": "ascii",
            "class": "dictionary",
            "object": object_name
        }
        self.attributes = dict(attributes)
        self.object_name = object_name

    def __getattr__(self, item: str) -> Any:
        """Allows access to dictionary attributes as object properties.

        Args:
            item: The attribute key to retrieve.

        Returns:
            The value associated with the key in `self.attributes`.

        Raises:
            AttributeError: If the key does not exist in the attributes dictionary.
        """
        if item in self.attributes:
            return self.attributes[item]
        raise AttributeError(f"\'{self.__class__.__name__}\' object has no attribute \'{item}\'")

    def __setattr__(self, key: str, value: Any):
        """Allows modifying existing dictionary attributes via dot notation.

        Args:
            key: The attribute name.
            value: The value to assign.
        """
        if key in ("header", "attributes", "object_name"):
            super().__setattr__(key, value)
        elif "attributes" in self.__dict__ and key in self.attributes:
            self.attributes[key] = value
        else:
            super().__setattr__(key, value)

    def _format_value(self, key: str, value: Any) -> str:
        """Formats Python types into OpenFOAM-compatible string syntax.

        Handles Booleans (true/false), floating point precision, and 
        `Quantity` objects with unit conversion.

        Args:
            key: The attribute key (used to determine target units).
            value: The value to format.

        Returns:
            A string representation of the value for OpenFOAM files.
        """
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return format(value, ".15g")
        if Quantity and isinstance(value, Quantity):
            unit = self.DEFAULT_UNITS.get(key)
            val = value.get_in(unit) if unit else value.magnitude
            if isinstance(val, (int, float)):
                return f'{val:.15g}'
            return str(val)
        return str(value)

    def _write_attributes(self, file: Any, attributes: dict, indent_level: int = 0):
        """Recursively writes dictionary attributes to a file with proper indentation.

        Args:
            file: The file object to write to.
            attributes: The dictionary of attributes to write.
            indent_level: Current indentation depth.
        """
        indent = "    " * indent_level
        for key, value in attributes.items():
            if value is None:
                continue

            # dictionary → block
            if isinstance(value, dict):
                if value:
                    file.write(f'{indent}{key}\n{indent}{{\n')
                    self._write_attributes(file, value, indent_level + 1)
                    file.write(f'{indent}}}\n')
                continue

            # tuple → OpenFOAM list
            if isinstance(value, tuple):
                file.write(f'{indent}{key} (')
                for item in value:
                    file.write(f'{self._format_value(key, item)} ')
                file.write(');\n')
                continue

            # standard key-value
            file.write(f'{indent}{key} {self._format_value(key, value)};\n')

    def write_file(self, filepath: Union[str, Path]):
        """Writes the current object as a standard OpenFOAM dictionary file.

        Args:
            filepath: Destination path where the file will be saved.

        Raises:
            IOError: If the file cannot be written to the disk.
        """
        try:
            filepath = Path(filepath)
            with open(filepath, 'w') as file:
                # header FoamFile
                file.write("FoamFile\n{\n")
                for key, value in self.header.items():
                    file.write(f'    {key}     {value};\n')
                file.write("}\n\n")

                # main content
                self._write_attributes(file, self.attributes)
        except IOError as e:
            print(f"Error writing file {filepath}: {e}")

    def _generate_field_header(self, field: str) -> str:
        """Generates the FoamFile header specific to field files (e.g., volScalarField).

        Args:
            field: The field name (e.g., 'U', 'p').

        Returns:
            A string containing the formatted OpenFOAM header.
        """
        class_field = "volVectorField" if field == "U" else "volScalarField"
        return (
            f"FoamFile\n"
            f"{{\n"
            f"    version     2.0;\n"
            f"    format      ascii;\n"
            f"    class       {class_field};\n"
            f"    object      {field};\n"
            f"}}\n"
        )

    def write_boundary_file(self, field: str, boundaries: dict, case_path: Union[str, Path], internal_field: Optional[str] = None):
        """Writes an OpenFOAM boundary condition file (usually in the '0/' directory).

        This method handles the creation of the directory, dimension definitions,
        and the formatting of boundary patches.

        Args:
            field: Name of the field (e.g., "U", "p", "nut").
            boundaries: Dictionary where keys are patch names and values 
                are dictionaries of BC parameters.
            case_path: Path to the OpenFOAM case root directory.
            internal_field: Override for the `internalField` entry. 
                Defaults to sensible defaults (e.g., "uniform (0 0 0)" for U).
        """
        base_path = Path(case_path)
        folder_0 = base_path / "0"
        folder_0.mkdir(parents=True, exist_ok=True)
        file_path = folder_0 / field

        # Default internal fields
        default_fields = {
            "U": "uniform (0 0 0)",
            "p": "uniform 0",
            "k": "uniform 0.375",
            "epsilon": "uniform 0.125",
            "nut": "uniform 0"
        }
        internal_field = internal_field or default_fields.get(field, "uniform 0")

        # Write file
        with open(file_path, "w") as f:
            f.write(self._generate_field_header(field))
            f.write(f'\ndimensions      {self.FIELD_DIMENSIONS.get(field, "[0 0 0 0 0 0 0]")};\n')
            f.write(f"internalField   {internal_field};\n\n")

            f.write("boundaryField\n{\n")
            for patch, params in boundaries.items():
                f.write(f'    {patch}\n    {{\n')
                for key, value in params.items():
                    f.write(f'        {key:<15} {value};\n')
                f.write("    }\n\n")
            f.write("}\n\n")
            f.write("// ************************************************************************* //\n")

        print(f"[✔] Wrote boundary file: {file_path}")
