from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Union, Dict
import logging

from foampilot.utilities.manageunits import ValueWithUnit

logger = logging.getLogger(__name__)


class FoamDict:
    """Core dictionary writing capability - unified API for OpenFOAM dictionary files.

    This class provides a fluent interface for creating and writing OpenFOAM
    dictionary files. It wraps dictionary content with proper header generation,
    value formatting, and file writing.

    Attributes:
        object_name: Name in FoamFile header (e.g., "transportProperties")
        base_path: Optional path for writing
        default_data: Optional default values
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

    def __init__(self, object_name: str, base_path: Optional[Path] = None,
                 default_data: Optional[Dict[str, Any]] = None):
        """Initialize FoamDict.

        Args:
            object_name: Name in FoamFile header (e.g., "transportProperties")
            base_path: Optional path for writing
            default_data: Optional default values
        """
        self._object_name = object_name
        self._base_path = base_path
        self._data: Dict[str, Any] = default_data.copy() if default_data else {}
        self._header = {
            "version": "2.0",
            "format": "ascii",
            "class": "dictionary",
            "object": object_name
        }

    @property
    def object_name(self) -> str:
        return self._object_name

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    def __getattr__(self, item: str) -> Any:
        if item.startswith("_"):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")
        if item in self._data:
            return self._data[item]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def __setattr__(self, key: str, value: Any):
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any):
        self._data[key] = value

    def _format_value(self, key: str, value: Any) -> str:
        """Format a value for OpenFOAM key-value pair.

        Args:
            key: The attribute key
            value: The value to format

        Returns:
            A formatted string
        """
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return format(value, ".15g")
        if isinstance(value, ValueWithUnit):
            unit = self.DEFAULT_UNITS.get(key)
            val = value.get_in(unit) if unit else value.magnitude
            if isinstance(val, (int, float)):
                return f'{val:.15g}'
            return str(val)
        return str(value)

    def _write_content(self, file) -> None:
        """Write dictionary content to a file object.

        Args:
            file: File object to write to
        """
        for key, value in self._data.items():
            if value is None:
                continue

            quoted_key = f'"{key}"' if any(c in key for c in '.*|()') else key

            if isinstance(value, dict):
                if value:
                    file.write(f'{quoted_key}\n{{\n')
                    self._write_dict_content(file, value, 1)
                    file.write("}\n")
                continue

            if isinstance(value, tuple):
                parts = []
                for item in value:
                    if isinstance(item, tuple):
                        parts.append("(" + " ".join(str(x) for x in item) + ")")
                    else:
                        parts.append(self._format_value(key, item))
                file.write(f'{quoted_key} ({" ".join(parts)});\n')
                continue

            fmt_val = self._format_value(key, value)
            if "\n" in fmt_val:
                file.write(f'{quoted_key} {fmt_val}\n')
            else:
                file.write(f'{quoted_key} {fmt_val};\n')

    def _write_dict_content(self, file, attributes: dict, indent_level: int = 0) -> None:
        """Recursively write dictionary attributes with proper indentation.

        Args:
            file: File object to write to
            attributes: Dictionary of attributes to write
            indent_level: Current indentation depth
        """
        indent = "    " * indent_level
        for key, value in attributes.items():
            if value is None:
                continue

            if key == "_include_etc":
                file.write(f"{indent}{value}\n\n")
                continue

            quoted_key = f'"{key}"' if any(c in key for c in '.*|()') else key

            if isinstance(value, dict):
                if value:
                    file.write(f'{indent}{quoted_key}\n{indent}{{\n')
                    self._write_dict_content(file, value, indent_level + 1)
                    file.write(f'{indent}}}\n')
                continue

            if isinstance(value, tuple):
                parts = []
                for item in value:
                    if isinstance(item, tuple):
                        parts.append("(" + " ".join(str(x) for x in item) + ")")
                    else:
                        parts.append(self._format_value(key, item))
                file.write(f'{indent}{quoted_key} ({" ".join(parts)});\n')
                continue

            fmt_val = self._format_value(key, value)
            if "\n" in fmt_val:
                file.write(f'{indent}{quoted_key} {fmt_val}\n')
            else:
                file.write(f'{indent}{quoted_key} {fmt_val};\n')

    def write(self, path: Optional[Union[str, Path]] = None, footer: bool = False) -> None:
        """Write this FoamDict to a file.

        Args:
            path: Optional path to write to. If not provided, uses base_path
            footer: Whether to add the OpenFOAM footer (default: False for compatibility)
        """
        if path is None:
            path = self._base_path
        if path is None:
            raise ValueError("No path provided for writing")

        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(filepath, 'w') as file:
                file.write("FoamFile\n{\n")
                for key, value in self._header.items():
                    file.write(f'    {key}     {value};\n')
                file.write("}\n\n")
                self._write_content(file)
                if footer:
                    file.write("\n// ************************************************************************* //\n")
        except IOError as e:
            logger.error(f"Error writing file {filepath}: {e}")
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary representation of the data."""
        return self._data.copy()

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> "FoamDict":
        """Create a FoamDict from a file.

        Note: This is a basic parser. For full parsing, use the OpenFOAMFile class.

        Args:
            filepath: Path to the file to read

        Returns:
            A new FoamDict instance with data from the file
        """
        filepath = Path(filepath)
        obj_name = filepath.name

        fd = cls(object_name=obj_name, base_path=filepath.parent)

        try:
            with open(filepath, 'r') as f:
                content = f.read()
            in_section = False
            current_dict = {}
            dict_stack = [(current_dict, 0)]

            for line in content.split('\n'):
                line = line.rstrip()
                if not line or line.startswith('//') or line.startswith('#'):
                    continue

                if line.startswith('FoamFile'):
                    in_section = True
                    continue

                if in_section:
                    if line == '}':
                        in_section = False
                    continue

                stripped = line.strip()
                if stripped == '{':
                    continue

                if stripped == '}':
                    if dict_stack:
                        dict_stack.pop()
                    continue

                if '=' in stripped:
                    key, val = stripped.split('=', 1)
                    key = key.strip()
                    val = val.rstrip(';').strip()
                    if val.startswith('(') and val.endswith(')'):
                        val = val[1:-1]
                    current_dict[key] = val
                    dict_stack[-1][0][key] = val
                else:
                    parts = stripped.split()
                    if len(parts) >= 2:
                        key = parts[0]
                        rest = ' '.join(parts[1:]).rstrip(';')
                        current_dict[key] = rest
                        dict_stack[-1][0][key] = rest

            fd._data = current_dict
        except Exception as e:
            logger.warning(f"Error reading file {filepath}: {e}")

        return fd

    def update(self, data: Dict[str, Any]) -> "FoamDict":
        """Update data and return self for chaining.

        Args:
            data: Dictionary of data to update

        Returns:
            self for fluent API
        """
        self._data.update(data)
        return self

    def set_header(self, **kwargs) -> "FoamDict":
        """Set header fields.

        Args:
            **kwargs: Header fields to set

        Returns:
            self for fluent API
        """
        self._header.update(kwargs)
        return self


class BoundaryDict:
    """Specialized writer for OpenFOAM boundary condition files in 0/.

    Writes field files with proper FoamFile header, dimensions, internalField,
    and boundaryField blocks using the FoamDict core infrastructure.
    """

    FIELD_DIMENSIONS = {
        "U": "[0 1 -1 0 0 0 0]",
        "p": "[0 2 -2 0 0 0 0]",
        "p_rgh": "[1 -1 -2 0 0 0 0]",
        "k": "[0 2 -2 0 0 0 0]",
        "epsilon": "[0 2 -3 0 0 0 0]",
        "nut": "[0 2 -1 0 0 0 0]",
        "nuTilda": "[0 2 -1 0 0 0 0]",
        "muTilda": "[1 -1 -1 0 0 0 0]",
        "omega": "[0 0 -1 0 0 0 0]",
        "T": "[0 0 0 1 0 0 0]",
        "alphat": "[1 -1 -1 0 0 0 0]",
        "alpha.water": "[]",
        "alpha.air": "[]",
        "alpha": "[]",
        "pointDisplacement": "[0 1 0 0 0 0 0]",
    }

    FIELD_CLASSES = {
        "U": "volVectorField",
        "pointDisplacement": "pointVectorField",
    }

    DEFAULT_INTERNAL_FIELDS = {
        "U": "uniform (0 0 0)",
        "p": "uniform 0",
        "p_rgh": "uniform 0",
        "k": "uniform 0.375",
        "epsilon": "uniform 0.125",
        "omega": "uniform 1.0",
        "nut": "uniform 0",
        "alpha.water": "uniform 0",
        "alpha.air": "uniform 1",
        "alpha": "uniform 0",
        "T": "uniform 0",
        "pointDisplacement": "uniform (0 0 0)",
    }

    def __init__(self, field, boundaries, internal_field=None, dimensions=None,
                 include_etc=True, compressible=False, base_path=None):
        """Initialize BoundaryDict.

        Args:
            field: Field name (e.g., "U", "p", "k")
            boundaries: Dict of patch_name -> params dict
            internal_field: Optional internalField value string
            dimensions: Optional dimensions string
            include_etc: Whether to include #includeEtc directive
            compressible: Whether this is a compressible flow
            base_path: Optional base path for writing
        """
        self.field = field
        self.boundaries = boundaries
        self.internal_field = internal_field
        self.dimensions = dimensions or self.FIELD_DIMENSIONS.get(field, "[0 0 0 0 0 0 0]")
        self.include_etc = include_etc
        self.compressible = compressible
        self._base_path = base_path

        if compressible and field == "p":
            self.dimensions = "[1 -1 -2 0 0 0 0]"

        data = {}
        if self.dimensions:
            data["dimensions"] = self.dimensions
        if self.internal_field is None:
            self.internal_field = self.DEFAULT_INTERNAL_FIELDS.get(field)
        if self.internal_field:
            data["internalField"] = self.internal_field
        if self.boundaries is not None:
            bf = {}
            if include_etc:
                bf["_include_etc"] = '#includeEtc "caseDicts/setConstraintTypes"'
            for patch, params in self.boundaries.items():
                if params:
                    bf[patch] = params
            data["boundaryField"] = bf

        field_class = self.FIELD_CLASSES.get(field, "volScalarField")
        self._foam_dict = FoamDict(field, base_path=base_path, default_data=data)
        self._foam_dict.set_header(**{"class": field_class})

    def write(self, path=None, footer=False):
        """Write the boundary condition file.

        Args:
            path: Optional path to write to
            footer: Whether to add the OpenFOAM footer
        """
        self._foam_dict.write(path=path, footer=footer)
