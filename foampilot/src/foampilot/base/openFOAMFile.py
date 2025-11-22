import os
from pathlib import Path
from foampilot.utilities.manageunits import Quantity
from typing import Optional, Any, Union, Tuple


class OpenFOAMFile:
    """
    A base class for OpenFOAM configuration files.

    Attributes:
        header (dict): The header information for the OpenFOAM file.
        attributes (dict): The specific attributes for the file, including nested dictionaries.
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

    def __init__(self, object_name, **attributes):
        """
        Initializes the OpenFOAMFile with a header and specific attributes.
        """
        self.header = {
            "version": "2.0",
            "format": "ascii",
            "class": "dictionary",
            "object": object_name
        }
        self.attributes = dict(attributes)
        self.object_name = object_name

    # -------------------------------------------------------------------------
    # Attribute access helpers
    # -------------------------------------------------------------------------
    def __getattr__(self, item):
        if item in self.attributes:
            return self.attributes[item]
        raise AttributeError(f"\'{self.__class__.__name__}\' object has no attribute \'{item}\'")

    def __setattr__(self, key, value):
        if key in ("header", "attributes", "object_name"):
            super().__setattr__(key, value)
        elif "attributes" in self.__dict__ and key in self.attributes:
            self.attributes[key] = value
        else:
            super().__setattr__(key, value)
    # -------------------------------------------------------------------------
    # Internal utilities
    # -------------------------------------------------------------------------
    def _format_value(self, key, value):
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

    def _write_attributes(self, file, attributes, indent_level=0):
        indent = "    " * indent_level
        for key, value in attributes.items():
            if value is None:
                continue

            # dictionnaire → bloc
            if isinstance(value, dict):
                if value:
                    file.write(f'{indent}{key}\n{indent}{{\n')
                    self._write_attributes(file, value, indent_level + 1)
                    file.write(f'{indent}}}\n')
                continue  # éviter d'écrire dict en tant que str

            # tuple → OpenFOAM list
            if isinstance(value, Tuple):
                file.write(f'{indent}{key} (')
                for item in value:
                    file.write(f'{self._format_value(key, item)} ')
                file.write(');\n')
                continue

            # tout le reste
            file.write(f'{indent}{key} {self._format_value(key, value)};\n')

    # -------------------------------------------------------------------------
    # Generic writer
    # -------------------------------------------------------------------------
    def write_file(self, filepath):
        try:
            filepath = Path(filepath)
            with open(filepath, 'w') as file:
                # header FoamFile
                file.write("FoamFile\n{\n")
                for key, value in self.header.items():
                    file.write(f'    {key}     {value};\n')
                file.write("}\n\n")

                # contenu principal
                self._write_attributes(file, self.attributes)
        except IOError as e:
            print(f"Error writing file {filepath}: {e}")
            
    # -------------------------------------------------------------------------
    # Specific: boundary field file
    # -------------------------------------------------------------------------
    def _generate_field_header(self, field):
        """
        Generate the OpenFOAM header for a specific field file.
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

    def write_boundary_file(self, field, boundaries, case_path, internal_field=None):
        """
        Write an OpenFOAM boundary file for the given field.
        
        Args:
            field (str): Field name ("U", "p", "k", "epsilon", etc.)
            boundaries (dict): Dictionary of boundary conditions for this field.
            case_path (Path | str): Base path to the case directory.
            internal_field (str, optional): Override for internal field (default auto).
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