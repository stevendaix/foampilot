import os
from pathlib import Path

def _fmt_vec(vec):
    """Format a coordinate tuple as ``(a b c)``."""
    return "(" + " ".join(str(v) for v in vec) + ")"

def _write_dict_inline(file, d, indent=1):
    """Write a flat dict as OpenFOAM key-value pairs."""
    pad = "        " * indent
    for key, value in d.items():
        if isinstance(value, (list, tuple)):
            formatted = ' '.join(str(v) for v in value)
            file.write(f"{pad}{key}         ({formatted});\n")
        else:
            file.write(f"{pad}{key}         {value};\n")


class Functions:
    """A utility class for generating and writing OpenFOAM function dictionary files.
    
    This class provides static methods to create configuration dictionaries for various
    OpenFOAM function objects (fieldAverage, reference pressure, runTimeControl) and
    write them to appropriate files. It also includes utility methods for directory
    handling and controlDict modification.
    """

    @staticmethod
    def check_directory(path):
        """Check if directory exists and create it if necessary.
        
        Args:
            path (str or Path): Path to the directory to check/create
            
        Returns:
            Path: The input path as a Path object
        """
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def is_numeric(s: str) -> bool:
        """Return True if a string can be parsed as a float."""
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def restore_includetec_boundary(case_path: str | Path, field: str) -> None:
        """Re-insert ``#includeEtc "caseDicts/setConstraintTypes"`` into the
        ``boundaryField`` block of a 0/ field file.

        OpenFOAM's ``setFields`` utility rewrites field files and strips the
        ``#includeEtc`` directive.  Call this after ``setFields`` to
        restore it so that parallel-processor patches are correctly typed.
        """
        path = Path(case_path) / "0" / field
        if not path.exists():
            return
        content = path.read_text()
        if "#includeEtc" in content:
            return
        marker = "boundaryField\n{"
        idx = content.find(marker)
        if idx == -1:
            return
        replacement = (
            'boundaryField\n{\n'
            '    #includeEtc "caseDicts/setConstraintTypes"\n\n'
        )
        content = content[:idx] + replacement + content[idx + len(marker):]
        path.write_text(content)

    @staticmethod
    def field_average(
        name_field,
        field="U",
        base="iteration",
        mean="on",
        prime2Mean="off",
        trigger_start="1",
        time_start="500",
        control_mode="timeOrTrigger",
        write_control="writeTime"
    ):
        """Generate a filename and dictionary for an OpenFOAM fieldAverage function.
        
        Args:
            name_field (str): Name for the field average function
            field (str): Field to average (default: "U")
            base (str): Base type for averaging (default: "iteration")
            mean (str): Whether to compute mean (default: "on")
            prime2Mean (str): Whether to compute prime squared mean (default: "off")
            trigger_start (str): When to start averaging (default: "1")
            time_start (str): Time to start averaging (default: "500")
            control_mode (str): Control mode (default: "timeOrTrigger")
            write_control (str): Write control method (default: "writeTime")
            
        Returns:
            tuple: (name_field, field_average_dict) where field_average_dict contains
                   the configuration for the fieldAverage function
        """
        field_average_dict = {
            "type": "fieldAverage",
            "libs": '("libfieldFunctionObjects.so");',
            "triggerStart": trigger_start,
            "timeStart": time_start,
            "controlMode": control_mode,
            "writeControl": write_control,
            "fields": {
                field: {
                    "base": base,
                    "mean": mean,
                    "prime2Mean": prime2Mean
                }
            }
        }
        return name_field, field_average_dict

    @classmethod
    def write_function_field_average(cls, name_field, field_average_dict, base_path, folder='system'):
        """Write the field average function to a specified file.
        
        Args:
            name_field (str): Name of the function/file
            field_average_dict (dict): Dictionary containing the configuration
            base_path (str or Path): Base path of the OpenFOAM case
            folder (str): Subfolder to write to (default: "system")
        """
        path = Path(base_path) / folder / f"{name_field}"

        cls.check_directory(path.parent)  # Ensure the parent directory exists

        with open(path, 'w') as file:
            file.write(f"{name_field}\n{{\n")
            file.write(f"    type {field_average_dict['type']};\n")
            file.write(f"    libs {field_average_dict['libs']}\n")

            # Writing other attributes
            file.write(f"    triggerStart {field_average_dict['triggerStart']};\n")
            file.write(f"    timeStart {field_average_dict['timeStart']};\n")
            file.write(f"    controlMode {field_average_dict['controlMode']};\n")
            file.write(f"    writeControl {field_average_dict['writeControl']};\n")

            # Writing fields
            file.write("    fields\n    (\n")
            for field, attrs in field_average_dict['fields'].items():
                file.write(f"        {field}\n        {{\n")
                for key, value in attrs.items():
                    file.write(f"            {key} {value};\n")
                file.write("        }\n")
            file.write("    );\n")
            file.write("}\n")

    @staticmethod
    def reference_pressure(
        name_field,
        ref_value="functionObjectValue",
        field="p",
        function_object="average1",
        function_object_result="average(p)Mean",
        write_control="writeTime",
        position=None
    ):
        """Generate a filename and dictionary for an OpenFOAM reference pressure function.
        
        Args:
            name_field (str): Name for the reference function
            ref_value (str): Reference value type (default: "functionObjectValue")
            field (str): Field to reference (default: "p")
            function_object (str): Function object to use (default: "average1")
            function_object_result (str): Result to use from function object (default: "average(p)Mean")
            write_control (str): Write control method (default: "writeTime")
            position (list, optional): Position coordinates if needed (default: None)
            
        Returns:
            tuple: (name_field, reference_dict) where reference_dict contains
                   the configuration for the reference function
        """
        reference_dict = {
            "type": "reference",
            "libs": '("libfieldFunctionObjects.so");',
            "writeControl": write_control,
            "field": field,
            "refValue": ref_value,
            "functionObject": function_object,
            "functionObjectResult": function_object_result,
            "position": position
        }
        return name_field, reference_dict


    @staticmethod
    def force_coeffs_and_binfield(
        name_force="forceCoeffs1",
        name_binfield="binField1",
        patches=("body",),
        p="p",
        U="U",
        rho="rhoInf",
        rhoInf=1.0,
        CofR=(3.5, 0, 0),
        liftDir=(0, 1, 0),
        dragDir=(1, 0, 0),
        pitchAxis=(0, 0, 1),
        magUInf=10.0,
        lRef=4.0,
        Aref=1.0,
        porosity=True,
        nBin=20,
        direction=(1, 0, 0),
        cellZones=("porousZone",),
        write_control="writeTime"
    ):
        """Crée les dictionnaires Python pour forceCoeffs (avec binData intégré) et binField (séparé)."""

        force_coeffs_dict = {
            "type": "forceCoeffs",
            "libs": '("libforces.so")',
            "writeControl": write_control,
            "writeFields": "true",
            "patches": f"({' '.join(patches)})",
            "p": p,
            "U": U,
            "rho": rho,
            "log": "true",
            "rhoInf": rhoInf,
            "liftDir": f"({ ' '.join(map(str, liftDir)) })",
            "dragDir": f"({ ' '.join(map(str, dragDir)) })",
            "CofR": f"({ ' '.join(map(str, CofR)) })",
            "pitchAxis": f"({ ' '.join(map(str, pitchAxis)) })",
            "magUInf": magUInf,
            "lRef": lRef,
            "Aref": Aref,
            "porosity": "on" if porosity else "off",
            "binData": {
                "nBin": nBin,
                "direction": f"({ ' '.join(map(str, direction)) })",
                "cumulative": "yes"
            }
        }



        return (name_force, force_coeffs_dict)

    @staticmethod
    def check_directory(path):
        Path(path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def write_refine_mesh_dict(
        cls, name, base_path, folder='system',
        coordinate_type="global",
        e1=(1, 0, 0),
        e2=(0, 1, 0),
        directions=(1, 1, 1),
        zones=None,
        append=False
    ):
        """Write a refineMeshDict configuration file.

        Args:
            name (str): Name of the file (default: "refineMeshDict")
            base_path (str or Path): Base path of the OpenFOAM case
            folder (str): Subfolder to write to (default: "system")
            coordinate_type (str): Type of coordinate system (default: "global")
            e1 (tuple): First direction vector (default: (1, 0, 0))
            e2 (tuple): Second direction vector (default: (0, 1, 0))
            directions (tuple): Directions to refine (default: (1, 1, 1))
            zones (list, optional): List of zone dictionaries, each with keys:
                'name', 'type', and zone-specific fields (default: None)
            append (bool): Whether to append to existing file (default: False)
        """
        path = Path(base_path) / folder / name
        cls.check_directory(path.parent)

        with open(path, 'a' if append else 'w') as file:
            file.write("FoamFile\n{\n")
            file.write("    format      ascii;\n")
            file.write("    class       dictionary;\n")
            file.write(f"    location    \"{folder}\";\n")
            file.write(f"    object      {name};\n")
            file.write("}\n")
            file.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")

            file.write("coordinates\n{\n")
            file.write(f"    type        {coordinate_type};\n")
            e1_str = ' '.join(str(v) for v in e1)
            e2_str = ' '.join(str(v) for v in e2)
            file.write(f"    e1          ({e1_str});\n")
            file.write(f"    e2          ({e2_str});\n")
            if directions == (1, 1, 1):
                file.write("    directions  (e1 e2 e3);\n")
            else:
                dirs = []
                labels = ['e1', 'e2', 'e3']
                for i, d in enumerate(directions):
                    if d:
                        dirs.append(labels[i])
                file.write(f"    directions  ({' '.join(dirs)});\n")
            file.write("}\n\n")

            if zones:
                file.write("zones\n{\n")
                for zone in zones:
                    zone_name = zone.pop('name', '')
                    zone_type = zone.pop('type', '')
                    if zone_name:
                        file.write(f"    {zone_name}\n    {{\n")
                    else:
                        file.write("    {\n")
                    file.write(f"        type        {zone_type};\n")
                    for key, value in zone.items():
                        if isinstance(value, (list, tuple)):
                            formatted = ' '.join(str(v) for v in value)
                            file.write(f"        {key}         ({formatted});\n")
                        else:
                            file.write(f"        {key}         {value};\n")
                    file.write("    }\n\n")
                file.write("}\n")
            file.write("\n// ************************************************************************* //\n")

    @classmethod
    def write_dynamic_mesh_dict(cls, name, base_path, folder='constant',
                                 refinement_regions=None, refine_interval=1,
                                 n_buffer_layers=1, max_cells=1000000,
                                 dump_level=True, append=False):
        """Write a dynamicMeshDict for dynamic mesh refinement.

        Args:
            refinement_regions (list): Each dict:
                ``{"cellZone": str, "field": str, "lowerRefineLevel": float,
                   "upperRefineLevel": float, "maxRefinement": int}``
        """
        path = Path(base_path) / folder / name
        cls.check_directory(path.parent)

        with open(path, 'a' if append else 'w') as file:
            file.write("FoamFile\n{\n")
            file.write("    format      ascii;\n")
            file.write("    class       dictionary;\n")
            file.write("    location    \"constant\";\n")
            file.write(f"    object      {name};\n")
            file.write("}\n")
            file.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")

            file.write("topoChanger\n{\n")
            file.write("    type            refiner;\n")
            file.write("    libs            (\"libfvMeshTopoChangers.so\");\n")
            file.write("    mover           none;\n")
            file.write(f"    refineInterval  {refine_interval};\n")
            if refinement_regions:
                file.write("    refinementRegions\n")
                file.write("    {\n")
                for region in refinement_regions:
                    name_r = region.pop('name', '')
                    file.write(f"        {name_r}\n        {{\n")
                    for k, v in region.items():
                        file.write(f"            {k:<20} {v};\n")
                    file.write("        }\n\n")
                file.write("    }\n")
            file.write(f"    nBufferLayers   {n_buffer_layers};\n")
            file.write(f"    maxCells        {max_cells};\n")
            if dump_level:
                file.write("    dumpLevel       true;\n")
            file.write("}\n")
            file.write("\n// ************************************************************************* //\n")

    @classmethod
    def write_create_zones_dict(cls, name, base_path, folder='system', zones=None, append=False):
        """Write a createZonesDict configuration file.

        Args:
            name (str): Name of the file (e.g. "createZonesDict")
            base_path (str or Path): Base path of the OpenFOAM case
            folder (str): Subfolder to write to (default: "system")
            zones (list): List of zone dicts, each with a single top-level key:
                ``"zoneName" -> {"type": ..., "zoneType": ..., ...}``
            append (bool): Whether to append to existing file
        """
        path = Path(base_path) / folder / name
        cls.check_directory(path.parent)

        with open(path, 'a' if append else 'w') as file:
            file.write("FoamFile\n{\n")
            file.write("    format      ascii;\n")
            file.write("    class       dictionary;\n")
            file.write("    location    \"system\";\n")
            file.write(f"    object      {name};\n")
            file.write("}\n")
            file.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")

            if zones:
                for zone_dict in zones:
                    for zone_name, zone_config in zone_dict.items():
                        ztype = zone_config.pop("type", "")
                        zone_config.pop("name", None)
                        file.write(f"{zone_name}\n{{\n")
                        file.write(f"    type        {ztype};\n")
                        for key, value in zone_config.items():
                            if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, (list, tuple)) for v in value):
                                formatted = f"({_fmt_vec(value[0])} {_fmt_vec(value[1])})"
                                file.write(f"    {key:<16} {formatted};\n")
                            elif isinstance(value, (list, tuple)):
                                formatted = ' '.join(str(v) for v in value)
                                file.write(f"    {key:<16} ({formatted});\n")
                            else:
                                file.write(f"    {key:<16} {value};\n")
                        file.write("}\n\n")
            file.write("// ************************************************************************* //\n")

    @classmethod
    def write_create_patch_dict(cls, name, base_path, folder='system',
                                 patches=None, append=False):
        """Write a createPatchDict for creating new boundary patches (e.g. annulus inlet).

        Args:
            patches (list): Each dict: {"name": str, "patchInfo": dict,
                "constructFrom": str, "zone": dict}
        """
        path = Path(base_path) / folder / name
        cls.check_directory(path.parent)

        with open(path, 'a' if append else 'w') as file:
            file.write("FoamFile\n{\n")
            file.write("    format      ascii;\n")
            file.write("    class       dictionary;\n")
            file.write(f"    location    \"{folder}\";\n")
            file.write(f"    object      {name};\n")
            file.write("}\n")
            file.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")

            if patches:
                file.write("patches\n{\n")
                for patch in patches:
                    pname = patch.pop('name', '')
                    file.write(f"    {pname}\n    {{\n")
                    if 'patchInfo' in patch:
                        file.write("        patchInfo\n        {\n")
                        _write_dict_inline(file, patch.pop('patchInfo'), indent=2)
                        file.write("        }\n")
                    if 'constructFrom' in patch:
                        file.write(f"        constructFrom {patch.pop('constructFrom')};\n")
                    if 'zone' in patch:
                        zone = patch.pop('zone')
                        file.write("        zone\n        {\n")
                        _write_dict_inline(file, zone, indent=2)
                        file.write("        }\n")
                    for k, v in patch.items():
                        file.write(f"        {k} {v};\n")
                    file.write("    }\n\n")
                file.write("}\n")
            file.write("// ************************************************************************* //\n")

    @classmethod
    def write_set_fields_dict(
        cls,
        name,
        base_path,
        folder='system',
        default_values=None,
        zones=None,
        append=False
    ):
        """Write a setFieldsDict configuration file.

        Args:
            name (str): Name of the file (default: "setFieldsDict")
            base_path (str or Path): Base path of the OpenFOAM case
            folder (str): Subfolder to write to (default: "system")
            default_values (dict, optional): Default field values, e.g.
                {"alpha.water": "0"} (default: None)
            zones (list, optional): List of zone dictionaries, each with keys:
                'name', 'type', and zone-specific fields (default: None)
            append (bool): Whether to append to existing file (default: False)
        """
        path = Path(base_path) / folder / name
        cls.check_directory(path.parent)

        with open(path, 'a' if append else 'w') as file:
            file.write("FoamFile\n{\n")
            file.write("    format      ascii;\n")
            file.write("    class       dictionary;\n")
            file.write(f"    location    \"{folder}\";\n")
            file.write(f"    object      {name};\n")
            file.write("}\n")
            file.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")

            file.write("defaultValues\n{\n")
            if default_values:
                for field, value in default_values.items():
                    file.write(f"    {field} {value};\n")
            file.write("}\n\n")

            if zones:
                file.write("zones\n{\n")
                for zone in zones:
                    zone_name = zone.pop('name', '')
                    zone_type = zone.pop('type', '')
                    if zone_name:
                        file.write(f"    {zone_name}\n    {{\n")
                    else:
                        file.write("    {\n")
                    file.write(f"        type        {zone_type};\n")
                    for key, value in zone.items():
                        if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, (list, tuple)) for v in value):
                            formatted = f"{_fmt_vec(value[0])} {_fmt_vec(value[1])}"
                            file.write(f"        {key}         {formatted};\n")
                        elif isinstance(value, (list, tuple)):
                            formatted = ' '.join(str(v) for v in value)
                            file.write(f"        {key}         ({formatted});\n")
                        elif isinstance(value, dict):
                            file.write(f"        {key}\n        {{\n")
                            for k, v in value.items():
                                file.write(f"            {k} {v};\n")
                            file.write("        }\n")
                        else:
                            file.write(f"        {key}         {value};\n")
                    file.write("    }\n\n")
                file.write("}\n")
            file.write("\n// ************************************************************************* //\n")

    @classmethod
    def write_force_coeffs_and_binfield(cls, force_tuple,  base_path, folder="system", append=False):
        def write_dict(file, name, dic):
            file.write(f"{name}\n{{\n")
            for k, v in dic.items():
                if isinstance(v, dict):
                    file.write(f"    {k}\n    {{\n")
                    for sub_k, sub_v in v.items():
                        file.write(f"        {sub_k} {sub_v};\n")
                    file.write("    }\n")
                else:
                    file.write(f"    {k} {v};\n")
            file.write("}\n\n")

        path = Path(base_path) / folder / force_tuple[0]
        cls.check_directory(path.parent)

        with open(path, "a" if append else "w") as f:
            write_dict(f, force_tuple[0], force_tuple[1])


    @classmethod
    def write_function_reference_pressure(cls, name_field, function_dict, base_path, folder='system', append=False):
        """Write the reference pressure function to a specified file.
        
        Args:
            name_field (str): Name of the function/file
            function_dict (dict): Dictionary containing the configuration
            base_path (str or Path): Base path of the OpenFOAM case
            folder (str): Subfolder to write to (default: "system")
            append (bool): Whether to append to existing file (default: False)
        """
        path = Path(base_path) / folder / f"{name_field}"

        cls.check_directory(path.parent)  # Ensure the parent directory exists

        with open(path, 'a' if append else 'w') as file:
            if not append:  # Write the header only if creating a new file
                file.write("// Auto-generated function dictionary\n\n")
            file.write(f"{name_field}\n{{\n")
            file.write(f"    type {function_dict['type']};\n")
            file.write(f"    libs {function_dict['libs']}\n")

            # Writing other attributes
            for key in ['writeControl', 'field', 'refValue', 'functionObject', 'functionObjectResult']:
                if key in function_dict:
                    file.write(f"    {key} {function_dict[key]};\n")

            # Writing position if provided
            if 'position' in function_dict and function_dict['position'] is not None:
                file.write(f"    position ({' '.join(map(str, function_dict['position']))});\n")

            file.write("}\n")

    @staticmethod
    def run_time_control(
        name_field,
        control_mode=None,
        trigger_start="1",
        satisfied_action="setTrigger",
        conditions=None
    ):
        """Generate a filename and dictionary for an OpenFOAM runTimeControl function.
        
        Args:
            name_field (str): Name for the runTimeControl function
            control_mode (str, optional): Control mode (default: None)
            trigger_start (str): When to start control (default: "1")
            satisfied_action (str): Action when condition is satisfied (default: "setTrigger")
            conditions (dict, optional): Dictionary of conditions (default: None)
            
        Returns:
            tuple: (name_field, run_time_control_dict) where run_time_control_dict contains
                   the configuration for the runTimeControl function
        """
        if conditions is None:
            conditions = {}

        run_time_control_dict = {
            "type": "time",
            "libs": '("libutilityFunctionObjects.so");',
            "controlMode": control_mode,
            "triggerStart": trigger_start,
            "satisfiedAction": satisfied_action,
            "conditions": conditions
        }
        return name_field, run_time_control_dict

    @classmethod
    def write_function_run_time_control(cls, name_field, name_condition, function_dict, base_path, folder='system', append=False):
        """Write the runTimeControl function to a specified file.
        
        Args:
            name_field (str): Name of the function/file
            name_condition (str): Name of the condition
            function_dict (dict): Dictionary containing the configuration
            base_path (str or Path): Base path of the OpenFOAM case
            folder (str): Subfolder to write to (default: "system")
            append (bool): Whether to append to existing file (default: False)
        """
        path = Path(base_path) / folder / f"{name_field}"

        cls.check_directory(path.parent)  # Ensure the parent directory exists

        with open(path, 'a' if append else 'w') as file:
            if not append:  # Write the header only if creating a new file
                file.write("// Auto-generated function dictionary\n\n")

            file.write(f"{name_condition}\n{{\n")
            file.write(f"    type {function_dict['type']};\n")
            file.write(f"    libs {function_dict['libs']}\n")

            # Writing other attributes
            for key in ['controlMode', 'triggerStart', 'satisfiedAction']:
                if key in function_dict and function_dict[key] is not None:
                    file.write(f"    {key} {function_dict[key]};\n")

            # Writing conditions if provided
            if 'conditions' in function_dict and function_dict['conditions']:
                file.write("    conditions\n    {\n")
                for condition_name, condition_attrs in function_dict['conditions'].items():
                    file.write(f"        {condition_name}\n        {{\n")
                    for key, value in condition_attrs.items():
                        file.write(f"            {key} {value};\n")
                    file.write("        }\n")
                file.write("    }\n")

            file.write("}\n")

    @classmethod
    def write_functions_in_controlDict(cls, base_path, folder='system', 
                                    control_dict_filename='controlDict', 
                                    functions_files=None):
        """Add or replace the 'functions' section in controlDict with includes.
        
        Modifies the controlDict file to include the specified function files
        in the functions section. If the section exists, it is replaced. If not,
        it is added at the end of the file.
        
        Args:
            base_path (str or Path): Path to the OpenFOAM case directory
            folder (str): Subfolder containing controlDict (default: "system")
            control_dict_filename (str): Name of controlDict file (default: "controlDict")
            functions_files (list): List of function files to include (without extension)
                                   Default: ["fieldAverage", "referencePressure", "runTimeControls"]
                                   
        Raises:
            FileNotFoundError: If controlDict file is not found
        """
        if functions_files is None:
            functions_files = ["fieldAverage", "referencePressure", "runTimeControls"]

        control_dict_path = Path(base_path) / folder / control_dict_filename
        if not control_dict_path.exists():
            raise FileNotFoundError(f"{control_dict_path} not found.")

        with open(control_dict_path, "r") as f:
            lines = f.readlines()

        # Build new functions section
        functions_section = ["functions\n", "{\n"]
        for file in functions_files:
            functions_section.append(f'    #include "{file}"\n')
        functions_section.append("}\n")

        # Find and replace existing functions section
        in_functions_block = False
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if "functions" in line.strip() and (i + 1 < len(lines)) and "{" in lines[i + 1]:
                # Enter functions block
                in_functions_block = True
                i += 2  # skip "functions" and "{"
                brace_count = 1
                while i < len(lines) and brace_count > 0:
                    if "{" in lines[i]:
                        brace_count += 1
                    if "}" in lines[i]:
                        brace_count -= 1
                    i += 1
                # Insert new section
                new_lines.extend(functions_section)
                continue
            else:
                new_lines.append(line)
                i += 1

        # If no functions section found, add it at the end
        if not in_functions_block:
            if not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"
            new_lines.append("\n")
            new_lines.extend(functions_section)

        # Rewrite controlDict
        with open(control_dict_path, "w") as f:
            f.writelines(new_lines)

        print(f"Section 'functions' added/modified in {control_dict_path}")