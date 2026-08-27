from pathlib import Path
from foampilot.tutorials.openfoam13 import validate_generated_case

cases = {
    "propellerFoundation13": Path("/home/ubuntu/foampilot-audit/openfoam13/FoamPilotCases/propellerFoundation13"),
}
for name, path in cases.items():
    result = validate_generated_case(path, is_vof=True)
    print(name)
    print(f"valid={result.valid}")
    print(f"missing={result.missing_files}")
    print(f"warnings={result.warnings}")
    if not result.valid:
        raise SystemExit(1)
