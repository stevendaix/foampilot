"""
FoamPilot adapter for the Wolf Dynamics CounterFlow Flame tutorial.
"""

from pathlib import Path
from foampilot.tutorials import WolfDynamicsTutorialBase


class CounterFlowFlameTutorial(WolfDynamicsTutorialBase):
    """Adapter for the counterFlowFlame2DLTS-OF13Units case."""

    def __init__(
        self,
        source_case_path: str | Path,
        target_case_path: str | Path,
        end_time: float | int | None = 20,
        write_interval: float | int | None = 10,
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
        """Generate and record all CounterFlow Flame input data via FoamPilot."""
        # `WolfDynamicsTutorialBase` rewrites every text input through
        # OpenFOAMDictAddFile and updates controlDict from these attributes.
        super().write_case()
