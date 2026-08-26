from pathlib import Path
from typing import Optional, Any
from foampilot.base.openFOAMFile import OpenFOAMFile


class PhasePropertiesFile(OpenFOAMFile):
    """
    OpenFOAM ``constant/phaseProperties`` file for VoF (incompressibleVoF /
    compressibleVoF) solvers.

    Unlike the standard ``transportProperties`` (single-phase), VoF solvers
    read phase-fraction data from ``phaseProperties`` which declares the
    list of phases and the surface tension coefficient ``sigma``.
    """

    def __init__(
        self,
        parent: Optional[Any] = None,
        phases=None,
        sigma: float = 0.0728,
    ):
        """
        Args:
            parent: Parent solver object (used for case_path lookups).
            phases: List of phase names, e.g. ``["water", "air"]``.
            sigma: Surface tension coefficient (N/m).
        """
        self.parent = parent
        self.phases = list(phases) if phases else ["water", "air"]
        self.sigma = float(sigma) if isinstance(sigma, (int, float)) else sigma

        super().__init__(object_name="phaseProperties")
        self.attributes = {
            "phases": f"({ ' '.join(self.phases) })",
            "sigma": self.sigma,
        }

    def write(self, filepath: Path):
        self.write_file(filepath)
