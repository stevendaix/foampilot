"""Backward-compatible Wolf Dynamics adapter entry point.

`WolfDynamicsTutorial` now directly inherits the fully FoamPilot-managed
`WolfDynamicsTutorialBase`; it does not keep a separate copy-and-run path.
"""

from foampilot.tutorials.wolfdynamics_base import WolfDynamicsTutorialBase


class WolfDynamicsTutorial(WolfDynamicsTutorialBase):
    """Compatibility alias for the FoamPilot-managed Wolf tutorial adapter."""

    pass
