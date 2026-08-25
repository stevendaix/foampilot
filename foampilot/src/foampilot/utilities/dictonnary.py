import os
from pathlib import Path

class OpenFOAMDictAddFile:
    """
    A base class for OpenFOAM configuration files.
    """

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
        self.attributes = attributes

    def _write_attributes(self, file, attributes, indent_level=0):
        indent = "    " * indent_level

        for key, value in attributes.items():
            if key == "box" and isinstance(value, list) and len(value) == 2 and all(
                isinstance(p, (list, tuple)) and len(p) == 3 for p in value
            ):
                # Cas particulier pour box
                flat = ''.join(f"({p[0]} {p[1]} {p[2]})" for p in value)
                file.write(f"{indent}{key}     {flat};\n")
            elif isinstance(value, dict):
                if value:
                    file.write(f"{indent}{key}\n{indent}{{\n")
                    self._write_attributes(file, value, indent_level + 1)
                    file.write(f"{indent}}}\n")
            elif isinstance(value, list):
                file.write(f"{indent}{key}\n{indent}(\n")
                for item in value:
                    if isinstance(item, dict):
                        file.write(f"{indent}    {{\n")
                        self._write_attributes(file, item, indent_level + 2)
                        file.write(f"{indent}    }}\n")
                    else:
                        file.write(f"{indent}    {item};\n")
                file.write(f"{indent});\n")
            else:
                file.write(f"{indent}{key} {str(value)};\n")


    def write_raw(self, name_dict, base_path, content, folder="system"):
        """Write an OpenFOAM dictionary body without serializing its syntax.

        Tutorial dictionaries often contain constructs that cannot be safely
        represented by the generic attribute serializer (for example
        ``#include`` directives, function objects, coded entries or nested
        lists with OpenFOAM-specific grammar).  If the supplied content already
        contains a ``FoamFile`` header it is preserved verbatim; otherwise the
        writer adds the standard header owned by this instance.

        Returns:
            Path: The path of the written dictionary.
        """
        path = Path(base_path) / folder / name_dict
        path.parent.mkdir(parents=True, exist_ok=True)
        text = str(content)
        has_header = "FoamFile" in text
        if not has_header:
            header = "FoamFile\n{\n"
            header += "    version     2.0;\n"
            header += "    format      ascii;\n"
            header += "    class       dictionary;\n"
            header += f"    object      {self.header['object']};\n"
            header += "}\n\n"
            text = header + text
        path.write_text(text, encoding="utf-8")
        return path

    def write(self, name_dict, base_path, folder='system'):
        """
        Writes the OpenFOAM file to the specified filepath.
        """
        try:
            path = Path(base_path) / folder / name_dict
            path.parent.mkdir(parents=True, exist_ok=True)
            filepath = path
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write("FoamFile\n{\n")
                for key, value in self.header.items():
                    file.write(f"    {key}     {value};\n")
                file.write("}\n\n")
                self._write_attributes(file, self.attributes)
        except IOError as e:
            print(f"Error writing file {filepath}: {e}")

class dict_tools:

    @staticmethod
    def create_patches_dict(patch_names, construct_from="set", point_sync=False):
        """Crée un dictionnaire pour plusieurs patches dans OpenFOAM.

        Args:
            patch_names (list): Liste des noms des patches à créer.
            construct_from (str): Méthode de construction ('patches' ou 'set').
            point_sync (bool): Indique si la synchronisation des points est activée.

        Returns:
            dict: Dictionnaire représentant la configuration des patches.
        """
        patches_list = []
        unique_patch_names = set(patch_names)

        for name in unique_patch_names:
            patch_dict = {
                "name": name,
                "patchInfo": {
                    "type": "patch"
                },
                "constructFrom": construct_from,
                "patches": "()",
                "set": name  # Vous pouvez personnaliser cela si nécessaire
            }
            patches_list.append(patch_dict)

        return {
            "pointSync": point_sync,
            "patches": patches_list
        }

    @staticmethod
    def create_action(name, action_type, action, source, **kwargs):
        """Crée une action pour topoSetDict.

        Args:
            name (str): Nom de l'action.
            action_type (str): Type de l'action (e.g., 'cellSet', 'faceSet').
            action (str): Type d'action (e.g., 'new', 'subset').
            source (str): Source de l'action (e.g., 'boxToCell', 'patchToFace').
            **kwargs: Autres attributs optionnels pour l'action.

        Returns:
            dict: Dictionnaire représentant l'action.
        """
        valid_types = ['cellSet', 'cellZoneSet', 'faceSet']
        valid_actions = ['new', 'subset', 'delete']

        if action_type not in valid_types:
            raise ValueError(f"Invalid action type: {action_type}. Must be one of {valid_types}.")
        if action not in valid_actions:
            raise ValueError(f"Invalid action: {action}. Must be one of {valid_actions}.")
        
        # Crée le dictionnaire de l'action avec les attributs fournis
        action_dict = {
            "name": name,
            "type": action_type,
            "action": action,
            "source": source
        }
        action_dict.update(kwargs)  # Ajoute d'autres attributs

        return action_dict

    @staticmethod
    def create_actions_dict(actions):
        """Crée un dictionnaire pour les actions dans topoSetDict.

        Args:
            actions (list): Liste des dictionnaires représentant chaque action.

        Returns:
            dict: Dictionnaire représentant les actions.
        """
        return {
            "actions": actions
        }
