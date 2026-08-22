"""
FoamPilot adapter for the Wolf Dynamics DamBreak VOF tutorial.
"""

from pathlib import Path
from foampilot.tutorials import WolfDynamicsTutorialBase


class DamBreakVOFTutorial(WolfDynamicsTutorialBase):
    """Adapter for the damBreak VOF case."""

    def __init__(
        self,
        source_case_path: str | Path,
        target_case_path: str | Path,
        end_time: float | int | None = 0.05,
        write_interval: float | int | None = 0.01,
    ):
        super().__init__(
            source_case_path=source_case_path,
            target_case_path=target_case_path,
            foamrun_module="incompressibleVoF",
            compressible=False,
            end_time=end_time,
            write_interval=write_interval,
        )

    def write_case(self) -> None:
        """Apply specific modifications for this tutorial."""
        super().write_case()
