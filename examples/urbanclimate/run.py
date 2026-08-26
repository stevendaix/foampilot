"""Generate and run urbanMicroclimateFoam Foundation 13 cases with FoamPilot."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from foampilot.openfoam13.urbanclimate import PROFILES, UrbanClimateCase

ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT / "templates"
CASES = ROOT / "cases"


def write_case(name: str, *, overwrite: bool = False) -> Path:
    """Generate one named case through the specialized FoamPilot API."""
    case = UrbanClimateCase.from_name(name, TEMPLATES)
    path = case.write_case(CASES / name, overwrite=overwrite)
    errors = case.validate(path)
    if errors:
        raise RuntimeError(f"{name}: generated case failed preflight: {'; '.join(errors)}")
    print(f"Generated FoamPilot case: {path}")
    return path


def run_case(name: str, *, overwrite: bool = False, regenerate: bool = True) -> Path:
    path = write_case(name, overwrite=overwrite) if regenerate else CASES / name
    allrun = path / "Allrun"
    if not allrun.is_file():
        raise FileNotFoundError(f"{name}: missing Allrun in generated case")
    subprocess.run(["sh", str(allrun)], cwd=path, check=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case", choices=tuple(PROFILES))
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--list", action="store_true")
    parser.add_argument("--generate", action="store_true", help="only generate files")
    parser.add_argument("--no-regenerate", action="store_true", help="run existing output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name, profile in PROFILES.items():
            print(f"{name}: {profile.description}")
        return 0
    names = list(PROFILES) if args.all else [args.case]
    for name in names:
        if args.generate:
            write_case(name, overwrite=args.overwrite)
        else:
            run_case(name, overwrite=args.overwrite, regenerate=not args.no_regenerate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
