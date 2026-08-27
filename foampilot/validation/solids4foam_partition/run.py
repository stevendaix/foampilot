"""Validate the generated solids4foam regional partition."""

from pathlib import Path

from foampilot.solids4foam import build_partition_validation


if __name__ == "__main__":
    case_path = Path(__file__).resolve().parent / "case"
    case, workflow = build_partition_validation(case_path)
    required = (
        case_path / "constant/fluid/polyMesh/points",
        case_path / "constant/solid/polyMesh/points",
        case_path / "constant/fsiProperties",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing generated files: " + ", ".join(missing))
    print("solids4foam partition validation: OK")
    print(workflow.preview())
