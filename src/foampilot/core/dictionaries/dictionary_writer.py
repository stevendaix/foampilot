from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Union
import logging

from foampilot.core.dictionaries.foam_dict import FoamDict

logger = logging.getLogger(__name__)


class DictionaryWriter:
    """Coordinates writing multiple FoamDict instances to a directory.

    This class manages a collection of FoamDict instances and provides
    batch writing capabilities for OpenFOAM case directories.

    Attributes:
        directory: Target directory for writing
    """

    def __init__(self, directory: Path):
        """Initialize DictionaryWriter.

        Args:
            directory: Target directory for writing
        """
        self.directory = Path(directory)
        self._dictionaries: Dict[str, FoamDict] = {}

    def register(self, name: str, foam_dict: FoamDict) -> None:
        """Register a FoamDict with a given name.

        Args:
            name: Name to register under (used as filename)
            foam_dict: FoamDict instance to register
        """
        self._dictionaries[name] = foam_dict
        if foam_dict._base_path is None:
            foam_dict._base_path = self.directory / name

    def get(self, name: str) -> Optional[FoamDict]:
        """Get a registered FoamDict by name.

        Args:
            name: Name of the FoamDict

        Returns:
            The FoamDict if registered, None otherwise
        """
        return self._dictionaries.get(name)

    def unregister(self, name: str) -> None:
        """Unregister a FoamDict by name.

        Args:
            name: Name of the FoamDict to unregister
        """
        if name in self._dictionaries:
            del self._dictionaries[name]

    def write_all(self) -> None:
        """Write all registered dictionaries to their respective files."""
        self.directory.mkdir(parents=True, exist_ok=True)

        for name, foam_dict in self._dictionaries.items():
            try:
                foam_dict.write(self.directory / name)
                logger.debug(f"Wrote dictionary: {name}")
            except Exception as e:
                logger.error(f"Error writing dictionary '{name}': {e}")
                raise

    def write(self, name: str) -> None:
        """Write a specific registered dictionary.

        Args:
            name: Name of the dictionary to write

        Raises:
            KeyError: If name is not registered
        """
        if name not in self._dictionaries:
            raise KeyError(f"Dictionary '{name}' is not registered")

        foam_dict = self._dictionaries[name]
        foam_dict.write(self.directory / name)
        logger.debug(f"Wrote dictionary: {name}")

    def names(self) -> list:
        """Return list of registered dictionary names."""
        return list(self._dictionaries.keys())

    def __len__(self) -> int:
        return len(self._dictionaries)

    def __contains__(self, name: str) -> bool:
        return name in self._dictionaries

    def clear(self) -> None:
        """Clear all registered dictionaries."""
        self._dictionaries.clear()

    def create_foam_dict(self, name: str, object_name: Optional[str] = None,
                         **kwargs) -> FoamDict:
        """Create and register a new FoamDict.

        Args:
            name: Name to register under (used as filename)
            object_name: Optional object_name override (defaults to name)
            **kwargs: Initial data for the FoamDict

        Returns:
            The created FoamDict instance
        """
        obj_name = object_name if object_name is not None else name
        foam_dict = FoamDict(object_name=obj_name, base_path=self.directory / name, default_data=kwargs)
        self.register(name, foam_dict)
        return foam_dict
