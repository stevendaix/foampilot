#!/usr/bin/env python3
"""
Run the Wolf Dynamics SandiaD flame OpenFOAM 13 tutorial via FoamPilot.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from foampilot.tutorials import OpenFOAM13Environment
from adapter import SandiaDFlameTutorial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-case", type=Path, required=True,
        help="Path to the extracted SandiaD_LTS-GRI30Small_EDC case.",
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path(".runs/sandia_d"),
        help="Disposable directory where the generated run case is written.",
    )
    args = parser.parse_args()

    target = args.run_root / "SandiaD_LTS-GRI30Small_EDC"
    tutorial = SandiaDFlameTutorial(
        source_case_path=args.source_case,
        target_case_path=target,
    )
    environment = OpenFOAM13Environment()

    print(f"Preparing case at {target}...")
    tutorial.setup_case()
    tutorial.write_case()
    
    validation = tutorial.validate()
    if not validation.valid:
        raise SystemExit(
            "FoamPilot validation failed: "
            + "; ".join((*validation.missing_files, *validation.warnings))
        )
        
    print("Checking mesh...")
    tutorial.check_mesh(environment)
    
    print("Running solver (smoke test)...")
    tutorial.run(environment)
    
    print(f"Tutorial run completed successfully at {target}")


if __name__ == "__main__":
    main()
