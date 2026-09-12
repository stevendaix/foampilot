from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any


class MeshGenerator(ABC):
    """Abstract base class for mesh generators.

    All mesh generators must implement ``generate()`` and ``write()``.
    """

    def __init__(self, case_path: Path, **options: Any):
        self.case_path = Path(case_path)
        self.options = options

    @abstractmethod
    def generate(self) -> None:
        """Generate or update the mesh description in memory."""

    @abstractmethod
    def write(self, destination: Optional[Path] = None) -> Path:
        """Write the mesh configuration to disk.

        Args:
            destination: Optional explicit destination path.

        Returns:
            The path that was written.
        """
