from __future__ import annotations

import csv
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

CASE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent / "openfoam_case"
COMMS = CASE / "comms"
ROOT = Path(__file__).resolve().parents[3]
SRC_CANDIDATES = [ROOT / "foampilot" / "src", Path("/home/ubuntu/foampilot/foampilot/src")]
SRC = next(path for path in SRC_CANDIDATES if (path / "foampilot").exists())
JOS_SRC = Path("/home/ubuntu/JOS-3/src")
sys.path.insert(0, str(JOS_SRC))

import jos3

# Chargement ciblé des modules physiologiques, sans importer le reste de FoamPilot.
_pkg = types.ModuleType("foampilot")
_pkg.__path__ = [str(SRC / "foampilot")]
sys.modules.setdefault("foampilot", _pkg)
_phys = types.ModuleType("foampilot.physiology")
_phys.__path__ = [str(SRC / "foampilot" / "physiology")]
sys.modules.setdefault("foampilot.physiology", _phys)
_jpkg = types.ModuleType("foampilot.physiology.jos3")
_jpkg.__path__ = [str(SRC / "foampilot" / "physiology" / "jos3")]
sys.modules.setdefault("foampilot.physiology.jos3", _jpkg)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


jos3_embedded = load("foampilot.physiology.jos3.jos3", SRC / "foampilot" / "physiology" / "jos3" / "jos3.py")
_jpkg.JOS3 = jos3_embedded.JOS3
_jpkg.BODY_NAMES = jos3_embedded.BODY_NAMES
coupling = load("foampilot.physiology.coupling", SRC / "foampilot" / "physiology" / "coupling.py")
provider_module = load("foampilot.postprocess.openfoam_external_coupled", SRC / "foampilot" / "postprocess" / "openfoam_external_coupled.py")

model = jos3_embedded.JOS3(ex_output="all")
network = None
provider = provider_module.OpenFOAM13TemperatureProvider(
    COMMS,
    file="data",
    timeout=10.0,
    air_temperature=20.0,
    radiative_temperature=20.0,
)

# externalCoupledTemperature attend une valeur initiale avant de produire data.out.
COMMS.mkdir(parents=True, exist_ok=True)
patch_faces = COMMS / "patchFaces"
n_initial = int(patch_faces.read_text().splitlines()[1]) if patch_faces.exists() else 4500
initial_rows = np.column_stack((np.full(n_initial, 307.15), np.zeros(n_initial), np.ones(n_initial)))
(COMMS / "data.in").write_text(
    "\n".join(" ".join(f"{value:.16g}" for value in row) for row in initial_rows) + "\n",
    encoding="utf-8",
)
(COMMS / "OpenFOAM.lock").touch()

while True:
    try:
        fields = provider.read_nodal_fields()
    except TimeoutError:
        break

    n = fields["areas"].size
    if network is None:
        mapping_file = CASE / "zone_mapping_openfoam.csv"
        if mapping_file.exists():
            with mapping_file.open(newline="", encoding="utf-8") as stream:
                mapping_rows = sorted(csv.DictReader(stream), key=lambda row: int(row["face_id"]))
            if len(mapping_rows) != n:
                raise ValueError(f"Mapping {len(mapping_rows)} faces, OpenFOAM en a fourni {n}")
            zone_ids = np.asarray([int(row["zone_id"]) for row in mapping_rows], dtype=int)
        else:
            zone_ids = np.arange(n, dtype=int) % 17
        mapping = coupling.SurfaceMapping(zone_ids=zone_ids, areas=fields["areas"])
        network = coupling.DistributedSurfaceNetwork(
            model,
            mapping,
            surface_temperature=fields["surface_temperature"],
        )

    exchange = network.step(
        fields["h"],
        fields["air_temperature"],
        radiative_temperature=fields["radiative_temperature"],
        dtime=1.0,
        hr=4.5,
    )
    provider.write_surface_temperature(exchange.surface_temperature)
    print(
        f"step: faces={n} h=[{fields['h'].min():.4g},{fields['h'].max():.4g}] "
        f"Tsurface=[{exchange.surface_temperature.min():.4g},{exchange.surface_temperature.max():.4g}]",
        flush=True,
    )

print("FoamPilot JOS-3 driver terminé.")
