from pathlib import Path
import json
from shapely.geometry import Polygon
from foampilot.urban.generation import UrbGENConfig, generate_urbgen

site = Polygon([(10, 10), (190, 10), (190, 140), (10, 140)])
report = []
for mode in range(4):
    result = generate_urbgen(site, UrbGENConfig(bcr=0.08, far=1.5, setback=5.0, seed=42, podium_floors=0, tower_size_mode=mode))
    report.append({"tower_size_mode": mode, "tower_count": result.diagnostics["tower_count"], "actual_bcr": result.actual_bcr, "actual_far": result.actual_far})
Path(__file__).with_name("urbgen_population_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
