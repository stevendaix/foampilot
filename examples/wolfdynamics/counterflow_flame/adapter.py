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
        """Apply specific modifications for this tutorial."""
        # Execute base smoke-test controls
        super().write_case()
        
        # Here we could inject specific boundary conditions or 
        # missing properties if the source case was incomplete.
        # For this specific OF13 clean case, the dictionaries are complete.
