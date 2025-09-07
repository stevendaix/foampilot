import os
from pathlib import Path

class OpenFOAMFile:
    """
    A base class for OpenFOAM configuration files.

    Attributes:
        header (dict): The header information for the OpenFOAM file.
        attributes (dict): The specific attributes for the file, including nested dictionaries.
    """

    def __init__(self, object_name, **attributes):
        """
        Initializes the OpenFOAMFile with a header and specific attributes.

        Args:
            object_name (str): The name of the object for the header.
            **attributes: Arbitrary keyword arguments representing the file's attributes.
        """
        # Define the header
        self.header = {
            "version": "2.0",
            "format": "ascii",
            "class": "dictionary",
            "object": object_name
        }
        # Specific attributes for the file, including nested dictionaries
        self.attributes = attributes

        self.object_name = object_name
        for key, value in attributes.items():  # Correction ici
            setattr(self, key, value)
            

    def _write_attributes(self, file, attributes, indent_level=0):
        """
        Writes the attributes to the file with proper indentation.

        Args:
            file (file object): The file object to write to.
            attributes (dict): The attributes to write.
            indent_level (int): The current level of indentation.
        """
        indent = "    " * indent_level

        # Only write if there are attributes to write
        for key, value in attributes.items():
            if isinstance(value, dict):
                if value:  # Write only if the dictionary is not empty
                    file.write(f"{indent}{key}\n{indent}{{\n")
                    self._write_attributes(file, value, indent_level + 1)
                    file.write(f"{indent}}}\n")
            else:
                # Convert Python bool to OpenFOAM true/false
                if isinstance(value, bool):
                    value = "true" if value else "false"
                # Write simple attributes
                file.write(f"{indent}{key} {value};\n")

    def write(self, filepath):
        """
        Writes the OpenFOAM file to the specified filepath.

        Args:
            filepath (str): The path to the file to write.
        """
        try:
            filepath = Path(filepath)
            with open(filepath, 'w') as file:
                # Automatic writing of the FoamFile header
                file.write("FoamFile\n{\n")
                for key, value in self.header.items():
                    file.write(f"    {key}     {value};\n")
                file.write("}\n\n")

                # Writing specific attributes
                self._write_attributes(file, self.attributes)
        except IOError as e:
            print(f"Error writing file {filepath}: {e}")
