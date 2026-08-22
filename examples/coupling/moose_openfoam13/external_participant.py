#!/usr/bin/env python3
"""Validation participant for the OpenFOAM 13 externalCoupled tutorial.

In production, the section computing ``next_temperature`` is replaced by the
MOOSE transfer/solve step. The file protocol remains identical.
"""

from __future__ import annotations

import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = REPO_ROOT / "foampilot" / "src" / "foampilot" / "coupling" / "external_coupled.py"
_SPEC = spec_from_file_location("foampilot_external_coupled", _MODULE_PATH)
_MODULE = module_from_spec(_SPEC)
sys.modules["foampilot_external_coupled"] = _MODULE
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
ExternalCoupledTemperature = _MODULE.ExternalCoupledTemperature


CASE_DIR = Path(__file__).resolve().parent
COMMS = CASE_DIR / "comms"
COUPLING = ExternalCoupledTemperature(COMMS, file_name="data", wait_interval=0.05, timeout=30)
STEPS = int(os.environ.get("FOAMPILOT_COUPLING_STEPS", "4"))


def write_initial_values() -> None:
    # The official OpenFOAM tutorial mesh has 2250 faces on each patch.
    # MOOSE would normally provide these values during its initial transfer.
    COUPLING.send_temperature_mixed_values(
        [(303.0 if i < 2250 else 283.0, 0.0, 1.0) for i in range(4500)]
    )


def main() -> int:
    write_initial_values()
    for step in range(1, STEPS + 1):
        records = COUPLING.wait_for_openfoam()
        next_temperature = [
            (record.temperature + 1.0, 0.0, 1.0) for record in records
        ]
        COUPLING.send_temperature_mixed_values(next_temperature)
        print(f"Foampilot participant: exchanged step {step} ({len(records)} faces)")
        time.sleep(0.01)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
