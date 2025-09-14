import os
from pathlib import Path
from foampilot.utilities.manageunits import Quantity

class OpenFOAMFile:
    """
    A base class for OpenFOAM configuration files.

    Attributes:
        header (dict): The header information for the OpenFOAM file.
        attributes (dict): The specific attributes for the file, including nested dictionaries.
    """
    DEFAULT_UNITS = {
        # transportProperties
        "nu": "m^2/s",           # kinematic viscosity
        "mu": "Pa.s",            # dynamic viscosity
        "rho": "kg/m^3",         # density

        # turbulenceProperties
        "k": "m^2/s^2",          # turbulent kinetic energy
        "epsilon": "m^2/s^3",    # turbulent dissipation rate
        "omega": "1/s",           # specific dissipation rate
        "nut": "m^2/s",           # turbulent viscosity
        "mut": "Pa.s",            # turbulent dynamic viscosity

        # field variables
        "U": "m/s",               # velocity vector
        "p": "Pa",                # pressure
        "T": "K",                 # temperature
        "alpha": "m^2/s",         # thermal diffusivity
        "phi": "m^3/s",           # volumetric flow rate

        # derived quantities
        "Re": None,               # Reynolds number, dimensionless
        "Pr": None,               # Prandtl number, dimensionless
        "Ma": None,               # Mach number, dimensionless
        "Fo": None,               # Fourier number, dimensionless
        "yPlus": None,            # dimensionless wall distance

        # porous media / multiphase
        "porosity": None,          # fraction, dimensionless
        "alpha.water": None,       # volume fraction, dimensionless
        "alpha.air": None,         # volume fraction, dimensionless
    }


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
 # Global switches
        self.compressible: bool = False     # True → physicalProperties, False → transportProperties
        self.energy: bool = False           # Add energy term in physicalProperties
        self.boussinesq: bool = False       # Use Boussinesq EOS if True, PerfectGas otherwise
        self.with_gravity: bool = True           # Control if g file is written

        # Store attributes dictionary
        self.attributes = dict(attributes)
        self.object_name = object_name

    def __getattr__(self, item):
        """
        Dynamically access attributes inside self.attributes.
        """
        if item in self.attributes:
            return self.attributes[item]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        """
        Dynamically update attributes inside self.attributes.
        """
        # To avoid recursion, bypass __setattr__ for core attributes
        if key in ("header", "attributes", "object_name"):
            super().__setattr__(key, value)
        elif "attributes" in self.__dict__ and key in self.attributes:
            self.attributes[key] = value
        else:
            super().__setattr__(key, value)

    def _format_value(self, key, value):
        """
        Convert value to a string for OpenFOAM.
        Uses DEFAULT_UNITS if value is a Quantity.
        """
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return format(value, ".15g")
        if isinstance(value, Quantity):
            unit = self.DEFAULT_UNITS.get(key, None)
            if unit:
                return format(value.get_in(unit), ".15g")
            else:
                return format(value.magnitude, ".15g")
        return str(value)



    def _write_attributes(self, file, attributes, indent_level=0):
        indent = "    " * indent_level
        for key, value in attributes.items():
            if isinstance(value, dict):
                if value:
                    file.write(f"{indent}{key}\n{indent}{{\n")
                    self._write_attributes(file, value, indent_level + 1)
                    file.write(f"{indent}}}\n")
            else:
                file.write(f"{indent}{key} {self._format_value(key, value)};\n")

    def write(self, filepath):
        """
        Writes the OpenFOAM file to the specified filepath.
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
