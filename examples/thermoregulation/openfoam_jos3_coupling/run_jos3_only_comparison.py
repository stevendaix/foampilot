from __future__ import annotations

import csv
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

case = Path(sys.argv[1]).resolve()
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 584
dtime = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
root = Path(__file__).resolve().parents[3]
src = root / "foampilot" / "src"
jos_src = Path("/home/ubuntu/JOS-3/src")
sys.path.insert(0, str(jos_src))
import jos3

pkg = types.ModuleType("foampilot")
pkg.__path__ = [str(src / "foampilot")]
sys.modules.setdefault("foampilot", pkg)
phys = types.ModuleType("foampilot.physiology")
phys.__path__ = [str(src / "foampilot" / "physiology")]
sys.modules.setdefault("foampilot.physiology", phys)
jpkg = types.ModuleType("foampilot.physiology.jos3")
jpkg.__path__ = [str(src / "foampilot" / "physiology" / "jos3")]
sys.modules.setdefault("foampilot.physiology.jos3", jpkg)

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

jos_mod = load("foampilot.physiology.jos3.jos3", src / "foampilot/physiology/jos3/jos3.py")
jpkg.JOS3 = jos_mod.JOS3
jpkg.BODY_NAMES = jos_mod.BODY_NAMES
coupling = load("foampilot.physiology.coupling", src / "foampilot/physiology/coupling.py")

rows = list(csv.DictReader((case / "zone_mapping_openfoam.csv").open(newline="", encoding="utf-8")))
zone_ids = np.asarray([int(row["zone_id"]) for row in sorted(rows, key=lambda r: int(r["face_id"]))], dtype=int)
area_map = np.asarray([float(row["area_m2"]) for row in sorted(rows, key=lambda r: int(r["face_id"]))], dtype=float)
data_out = case / "comms/data.out"
table = []
for line in data_out.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("//"):
        continue
    tokens = line.replace("(", " ").replace(")", " ").split()
    if len(tokens) >= 4:
        table.append([float(token) for token in tokens[:4]])
table = np.asarray(table)
if table.shape[0] != len(rows):
    raise ValueError(f"data.out faces={table.shape[0]}, mapping={len(rows)}")
areas = table[:, 0]
h = table[:, 3]
surface_temperature = table[:, 1] - 273.15
mapping = coupling.SurfaceMapping(zone_ids=zone_ids, areas=areas)
model = jos_mod.JOS3(ex_output="all")
network = coupling.DistributedSurfaceNetwork(model, mapping, surface_temperature=surface_temperature)

out = case / "jos3_only.csv"
with out.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["time_s", "Tsk_mean_C", "Tsk_min_C", "Tsk_max_C", "power_body_W", "power_environment_W"])
    for step in range(steps):
        exchange = network.step(h, np.full(h.size, 20.0), radiative_temperature=np.full(h.size, 20.0), dtime=dtime, hr=4.5)
        tsk = np.asarray(model.Tsk, dtype=float)
        writer.writerow([f"{(step + 1) * dtime:.9g}", f"{tsk.mean():.9g}", f"{tsk.min():.9g}", f"{tsk.max():.9g}", f"{exchange.zone_body_power.sum():.9g}", f"{exchange.environment_power.sum():.9g}"])
        if step == 0 or (step + 1) % 100 == 0:
            print(f"step={step + 1} time={((step + 1) * dtime):.6g}s Tsk=[{tsk.min():.6g},{tsk.max():.6g}]", flush=True)
print(f"JOS3_only terminé: steps={steps} duration={steps*dtime:.6g}s output={out}")
