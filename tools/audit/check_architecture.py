#!/usr/bin/env python3
"""Architecture audit for FoamPilot v3.

Verifies that ``core/`` never imports from forbidden packages such as
``workflows``, ``examples``, ``tutorials``, ``validation``, ``patches``,
or version-specific OpenFOAM directories.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FORBIDDEN_IMPORT_PREFIXES = (
    "foampilot.workflows",
    "foampilot.examples",
    "foampilot.tutorials",
    "foampilot.validation",
    "foampilot.patches",
    "foampilot.openfoam13",
    "foampilot.openfoam14",
    "foampilot.openfoam.com",
    "foampilot.openfoam.foundation",
    "foampilot.openfoam.esi",
)

FORBIDDEN_IMPORT_PATTERN = re.compile(
    r"^\s*(?:from|import)\s+(" + "|".join(re.escape(p) for p in FORBIDDEN_IMPORT_PREFIXES) + r")",
    re.IGNORECASE,
)


def scan_core_directory(root: Path) -> list[str]:
    """Return list of files in ``src/foampilot/core`` with forbidden imports."""
    core_root = root / "src" / "foampilot" / "core"
    violations: list[str] = []

    if not core_root.exists():
        return violations

    for path in sorted(core_root.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for line in text.splitlines():
            if FORBIDDEN_IMPORT_PATTERN.match(line):
                violations.append(f"{path}: {line.strip()}")

    return violations


def main() -> int:
    root = Path(__file__).resolve().parent.parent.parent
    violations = scan_core_directory(root)

    if violations:
        print("Architecture violation: core/ must not import from forbidden packages")
        for violation in violations:
            print(f"  - {violation}")
        return 1

    print("Architecture check passed: core/ imports are clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
