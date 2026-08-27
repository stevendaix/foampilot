from pathlib import Path
from foampilot.tutorials.openfoam13 import validate_generated_case

ROOT = Path('/home/ubuntu/foampilot-audit')
CASES = {
    'dtc_background': ROOT / 'openfoam13/DTCMoving_Overset_Foundation13/background',
    'mrf_smoke': Path('/tmp/propeller_mrf_smoke'),
    'foundation_propeller': Path('/tmp/of13_propeller_tutorial'),
}

for name, case in CASES.items():
    result = validate_generated_case(case, is_vof=True)
    print(f'{name}: valid={result.valid}')
    print(f'  missing={result.missing_files}')
    print(f'  warnings={result.warnings}')
    if result.missing_files or result.warnings:
        raise SystemExit(f'validation failed for {name}')
