"""Urban domain extensions.

.. deprecated::
    The urban module has been reorganized.
    Use ``foampilot.workflows.urban`` for workflows.
"""
import warnings
warnings.warn(
    "foampilot.urban is deprecated, "
    "use foampilot.workflows.urban for workflows",
    DeprecationWarning,
    stacklevel=2
)

import foampilot.urban as _urban
globals().update(vars(_urban))