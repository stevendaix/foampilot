#!/usr/bin/env python3
"""Negative checks for the multi-cloud audit contract."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDITOR = HERE / "analyze_multi_cloud_species.py"
EXPECTED = {
    "species": ["H2O", "C2H5OH"],
    "fragments": {
        "waterCloud:0": {
            "species": ["H2O", "C2H5OH"],
            "speciesMass": {"H2O": 0.7, "C2H5OH": 0.3},
        }
    },
}


def audit(log_text: str) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "log").write_text(log_text)
        (root / "expected.json").write_text(json.dumps(EXPECTED))
        result = subprocess.run(
            ["python3", str(AUDITOR), "--log", str(root / "log"),
             "--expected", str(root / "expected.json"), "--json"],
            check=False, capture_output=True, text=True,
        )
        assert result.returncode == 1, result.stdout + result.stderr
        return json.loads(result.stdout)


def main() -> None:
    duplicate = audit("\n".join([
        "End",
        "VOF direct commit cloud=waterCloud fragmentId=0 success=true mass=1",
        "VOF direct commit cloud=waterCloud fragmentId=0 success=true mass=1",
        "VOF confirmation cloud=waterCloud alphaField=alpha.water fragmentId=0 success=true mass=1 speciesMass=2(0.7 0.3)",
    ]))
    assert not duplicate["checks"]["allExpectedCommittedExactlyOnce"]

    orphan = audit("\n".join([
        "End",
        "VOF direct commit cloud=waterCloud fragmentId=0 success=true mass=1",
        "VOF confirmation cloud=waterCloud alphaField=alpha.water fragmentId=99 success=true mass=1 speciesMass=2(0.7 0.3)",
    ]))
    assert not orphan["checks"]["allExpectedConfirmedExactlyOnce"]

    print("negative_audit_checks=pass")


if __name__ == "__main__":
    main()
