from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class CaseLayout:
    """Manages OpenFOAM case directory structure.

    This class provides methods to create and manage the standard
    OpenFOAM case directory structure including time directories,
    constant directory, system directory, and auxiliary directories.

    Attributes:
        case_path: Path to the case root directory
    """

    STANDARD_DIRS = ["0", "constant", "system"]

    STANDARD_FILES = {
        "system": ["controlDict", "fvSchemes", "fvSolution"],
        "constant": ["polyMesh"],
    }

    def __init__(self, case_path: Union[str, Path]):
        """Initialize CaseLayout.

        Args:
            case_path: Path to the case root directory
        """
        self.case_path = Path(case_path)

    def ensure(self, subdirs: Optional[List[str]] = None,
               extra_directories: Optional[List[str]] = None) -> "CaseLayout":
        """Create standard directories and optionally extra directories.

        Args:
            subdirs: Optional list of time directories to create (e.g., ["0", "1", "2"])
            extra_directories: Optional list of extra directories (e.g., ["triSurface", "batch"])

        Returns:
            self for chaining
        """
        self.case_path.mkdir(parents=True, exist_ok=True)

        for dirname in self.STANDARD_DIRS:
            (self.case_path / dirname).mkdir(parents=True, exist_ok=True)

        if subdirs:
            for dirname in subdirs:
                (self.case_path / str(dirname)).mkdir(parents=True, exist_ok=True)

        if extra_directories:
            for dirname in extra_directories:
                (self.case_path / dirname).mkdir(parents=True, exist_ok=True)

        logger.debug(f"Ensured case directory structure at {self.case_path}")
        return self

    def subdir(self, name: str) -> Path:
        """Get path to a subdirectory.

        Args:
            name: Name of the subdirectory

        Returns:
            Path to the subdirectory
        """
        return self.case_path / name

    def constant(self) -> Path:
        """Get path to the constant directory."""
        return self.case_path / "constant"

    def system(self) -> Path:
        """Get path to the system directory."""
        return self.case_path / "system"

    def time_dir(self, time: Union[str, int, float]) -> Path:
        """Get path to a time directory.

        Args:
            time: Time value (e.g., "0", "1e-5", "end")

        Returns:
            Path to the time directory
        """
        return self.case_path / str(time)

    def exists(self) -> bool:
        """Check if the case path exists."""
        return self.case_path.exists()

    def is_case(self) -> bool:
        """Check if this looks like an OpenFOAM case (has constant and system dirs)."""
        if not self.exists():
            return False
        has_constant = (self.case_path / "constant").is_dir()
        has_system = (self.case_path / "system").is_dir()
        return has_constant and has_system

    def get_times(self) -> List[str]:
        """Get list of time directories (excluding constant and system).

        Returns:
            List of time directory names
        """
        if not self.exists():
            return []

        times = []
        for item in self.case_path.iterdir():
            if item.is_dir():
                name = item.name
                if name not in self.STANDARD_DIRS:
                    try:
                        float(name)
                        times.append(name)
                    except ValueError:
                        pass
        return sorted(times, key=lambda x: float(x) if x not in ("start", "end", "0") else (0 if x == "0" else float('inf')))

    def clean_times(self, keep_first: bool = True) -> List[Path]:
        """Remove all time directories except the first.

        Args:
            keep_first: If True, keeps the first (initial) time directory

        Returns:
            List of removed directories
        """
        removed = []
        times = self.get_times()

        if keep_first and times:
            times = times[1:]

        for time_str in times:
            time_path = self.case_path / time_str
            if time_path.exists() and time_path.is_dir():
                import shutil
                shutil.rmtree(time_path)
                removed.append(time_path)
                logger.info(f"Removed time directory: {time_path}")

        return removed
