import re
from pathlib import Path

class BoundaryFileHandler:
    """
    Handles OpenFOAM `boundary` files located in `constant/polyMesh/`.

    This class provides utilities to parse, update, and write
    the `boundary` dictionary that defines mesh patches in OpenFOAM.

    Attributes
    ----------
    base_path : Path
        Root case directory path.
    all_data : bool
        If True, all attributes of patches are extracted (nFaces, startFace, etc.).
    file_path : Path
        Path to the `boundary` file.
    data : dict
        Parsed boundary data stored as a dictionary.
    """

    def __init__(self, base_path, all_data: bool = False):
        """
        Initialize the boundary file handler.

        Parameters
        ----------
        base_path : str or Path
            Path to the root OpenFOAM case directory (must contain `constant/polyMesh/boundary`).
        all_data : bool, optional
            If True, extract all attributes of patches (default: False).

        Raises
        ------
        FileNotFoundError
            If the `boundary` file does not exist in the given path.
        """
        self.base_path = Path(base_path)
        self.all_data = all_data
        self.file_path = self.base_path / "constant" / "polyMesh" / "boundary"
        if not self.file_path.exists():
            raise FileNotFoundError(f"The file '{self.file_path}' does not exist.")
        self.data = self.parse_boundary_file(self.all_data)

    def comment_remover(self, text: str) -> str:
        """
        Remove line (`//`) and block (`/* */`) comments from a text
        while preserving string literals.

        Parameters
        ----------
        text : str
            Input text from which comments should be removed.

        Returns
        -------
        str
            The text with comments removed.
        """
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):  # Comment
                return " "  # Replace comments with a space
            else:
                return s  # Preserve string literals

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"', 
            re.DOTALL | re.MULTILINE
        )
        return re.sub(pattern, replacer, text)

    def parse_boundary_file(self, all_data: bool) -> dict:
        """
        Parse an OpenFOAM `boundary` file and extract patches.

        Parameters
        ----------
        all_data : bool
            If True, extract detailed patch attributes.

        Returns
        -------
        dict
            Dictionary containing patches and their attributes.
            Example:
            {
                "inlet": {"type": "patch", "nFaces": 100, "startFace": 0},
                "outlet": {"type": "patch"}
            }
        """
        with self.file_path.open('r') as file:
            file_content = file.read()

        # Remove comments
        cleaned_content = self.comment_remover(file_content)

        # Regex to capture patches
        patch_pattern = r"(\w+)\s*{\s*type\s*(\w+);(.*?)\}"

        patches = re.findall(patch_pattern, cleaned_content, re.DOTALL)
        dict_all_data = {}

        for patch in patches:
            patch_name = patch[0]
            patch_type = patch[1]
            patch_attributes = patch[2].strip().splitlines()
            patch_data = {'type': patch_type}
            if all_data:
                for attr in patch_attributes:
                    attr = attr.strip()
                    if 'inGroups' in attr:
                        patch_data['inGroups'] = attr.strip()
                    elif 'nFaces' in attr:
                        patch_data['nFaces'] = int(attr.split()[1].strip(';'))
                    elif 'startFace' in attr:
                        patch_data['startFace'] = int(attr.split()[1].strip(';'))

            dict_all_data[patch_name] = patch_data

        return dict_all_data

    def write_boundary_file(self, data: dict = None):
        """
        Write patch data to the OpenFOAM `boundary` file.

        Parameters
        ----------
        data : dict, optional
            Dictionary containing patch data to write. If None, uses `self.data`.
        """
        if data is None:
            data = self.data

        with self.file_path.open('w') as file:
            file.write('(\n')
            for patch, attributes in data.items():
                file.write(f'    {patch}\n')
                file.write('    {{\n')
                for key, value in attributes.items():
                    if isinstance(value, int):
                        file.write(f'        {key}            {value};\n')
                    else:
                        file.write(f'        {key}            {value}\n')
                file.write('    }}\n')
            file.write(')\n')

    def update_patch(self, patch_name: str, patch_data: dict):
        """
        Update or add a patch in the boundary data.

        Parameters
        ----------
        patch_name : str
            Name of the patch to update or add.
        patch_data : dict
            Dictionary of attributes for the patch.
        """
        self.data[patch_name] = patch_data