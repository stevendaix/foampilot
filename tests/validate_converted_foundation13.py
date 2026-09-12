"""Validate converted Foundation 13 cases without machine-local paths."""
from argparse import ArgumentParser
from pathlib import Path

from foampilot.tutorials.openfoam13 import validate_generated_case


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_case(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError("case must use NAME=PATH syntax")
    return name, Path(path).expanduser()


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="case to validate; may be repeated",
    )
    args = parser.parse_args()
    cases = {
        "propellerFoundation13": REPO_ROOT / "openfoam13/FoamPilotCases/propellerFoundation13",
    }
    for specification in args.case:
        name, path = parse_case(specification)
        cases[name] = path if path.is_absolute() else REPO_ROOT / path

    missing_cases = [str(path) for path in cases.values() if not path.is_dir()]
    if missing_cases:
        parser.error("case directory does not exist: " + ", ".join(missing_cases))

    for name, path in cases.items():
        result = validate_generated_case(path, is_vof=True)
        print(name)
        print(f"valid={result.valid}")
        print(f"missing={result.missing_files}")
        print(f"warnings={result.warnings}")
        if not result.valid or result.missing_files or result.warnings:
            raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
