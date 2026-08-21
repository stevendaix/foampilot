from pathlib import Path
import math
import sys

case_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(case_dir.parents[2] / "src" / "foampilot" / "utilities"))
from vof_to_dpm import VofToDpmConverter

converter = VofToDpmConverter(alpha_threshold=0.5)
fragments = converter.extract_case(case_dir, velocity_name="U")
assert len(fragments) == 1, fragments
fragment = fragments[0]
assert abs(fragment.volume - 1.0) < 1e-12, fragment.volume
assert max(abs(value - 0.5) for value in fragment.centroid) < 1e-12, fragment.centroid
assert fragment.velocity == (2.0, 0.0, 0.0), fragment.velocity
expected_diameter = (6.0 / math.pi) ** (1.0 / 3.0)
assert abs(fragment.equivalent_diameter - expected_diameter) < 1e-12
outputs = converter.write_openfoam_outputs(fragments, case_dir / "constant")
for path in outputs.values():
    assert path.exists() and path.stat().st_size > 0, path
print("PASS: OpenFOAM 13 mesh/fields -> VOF fragments -> DPM outputs")
