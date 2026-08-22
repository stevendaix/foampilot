"""
FoamPilot adapter for the Wolf Dynamics SandiaD flame tutorial.
"""

from pathlib import Path
from foampilot.tutorials import WolfDynamicsTutorialBase


class SandiaDFlameTutorial(WolfDynamicsTutorialBase):
    """Adapter for the SandiaD_LTS-GRI30Small_EDC case."""

    def __init__(
        self,
        source_case_path: str | Path,
        target_case_path: str | Path,
        end_time: float | int | None = 10,
        write_interval: float | int | None = 5,
    ):
        super().__init__(
            source_case_path=source_case_path,
            target_case_path=target_case_path,
            foamrun_module="multicomponentFluid",
            compressible=True,
            end_time=end_time,
            write_interval=write_interval,
        )

    def write_case(self) -> None:
        """Apply specific modifications for this tutorial."""
        super().write_case()
