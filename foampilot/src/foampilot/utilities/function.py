import os
from pathlib import Path

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