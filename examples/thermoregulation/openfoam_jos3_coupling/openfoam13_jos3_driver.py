from __future__ import annotations

import csv
import importlib.util
import re
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
surface_relaxation = 0.1
control_dict = (CASE / "system" / "controlDict").read_text(encoding="utf-8")
dtime_match = re.search(r"\bdeltaT\s+([0-9.eE+-]+)\s*;", control_dict)
if dtime_match is None:
    raise ValueError(f"deltaT absent de {CASE / 'system' / 'controlDict'}")
cfd_dtime = float(dtime_match.group(1))
if not np.isfinite(cfd_dtime) or cfd_dtime <= 0:
    raise ValueError(f"deltaT invalide: {cfd_dtime}")
provider = provider_module.OpenFOAM13TemperatureProvider(
    COMMS,
    file="data",
    timeout=10.0,
    air_temperature=20.0,
    radiative_temperature=20.0,
)

# Les fichiers natifs externalCoupled restent strictement inchangés. La
# traçabilité détaillée est écrite dans deux fichiers latéraux CSV.
TRACE = CASE / "coupling_trace.csv"
ZONE_TRACE = CASE / "coupling_zone_trace.csv"
TRACE.parent.mkdir(parents=True, exist_ok=True)
with TRACE.open("w", newline="", encoding="utf-8") as stream:
    csv.writer(stream).writerow([
        "exchange_id", "time_cfd_s", "time_jos3_s", "deltaT_cfd_s", "dtime_jos3_s",
        "n_faces", "area_total_m2", "h_area_mean_W_m2_K", "h_min_W_m2_K",
        "h_max_W_m2_K", "Ta_area_mean_C", "Tsurf_cfd_area_mean_C",
        "Tsurf_cfd_min_C", "Tsurf_cfd_max_C", "qDot_area_mean_W_m2",
        "qDot_min_W_m2", "qDot_max_W_m2", "qDot_integral_W",
        "Ttarget_area_mean_C", "Ttarget_min_C", "Ttarget_max_C",
        "Treturn_area_mean_C", "Treturn_min_C", "Treturn_max_C",
        "environment_power_W", "body_power_W", "time_error_s",
    ])
with ZONE_TRACE.open("w", newline="", encoding="utf-8") as stream:
    csv.writer(stream).writerow([
        "exchange_id", "time_cfd_s", "time_jos3_s", "zone_id", "zone_name",
        "zone_area_m2", "zone_air_temperature_C", "zone_h_mean_W_m2_K",
        "zone_body_power_W", "zone_surface_temperature_area_mean_C",
    ])

exchange_id = 0
cfd_time_s = 0.0
jos3_time_s = 0.0

def _area_mean(values, areas):
    values = np.asarray(values, dtype=float)
    areas = np.asarray(areas, dtype=float)
    return float(np.sum(values * areas) / np.sum(areas))

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

    exchange_id += 1
    cfd_time_s = exchange_id * cfd_dtime
    exchange = network.step(
        fields["h"],
        fields["air_temperature"],
        radiative_temperature=fields["radiative_temperature"],
        dtime=cfd_dtime,
        hr=4.5,
    )
    # Sous-relaxation du couplage CFD-physiologie : elle amortit la condition
    # mixte externalCoupledTemperature sans modifier les unités (K en retour).
    target_temperature = exchange.surface_temperature.copy()
    relaxed_temperature = (
        (1.0 - surface_relaxation) * fields["surface_temperature"]
        + surface_relaxation * target_temperature
    )
    network.surface_temperature = relaxed_temperature.copy()
    provider.write_surface_temperature(relaxed_temperature)
    jos3_time_s += cfd_dtime
    area_total = float(np.sum(fields["areas"]))
    q_dot = np.asarray(fields["q_dot"], dtype=float)
    environment_power = float(np.sum(exchange.environment_power))
    body_power = float(np.sum(exchange.body_power))
    time_error = cfd_time_s - jos3_time_s
    with TRACE.open("a", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow([
            exchange_id, f"{cfd_time_s:.16g}", f"{jos3_time_s:.16g}",
            f"{cfd_dtime:.16g}", f"{cfd_dtime:.16g}", n, f"{area_total:.16g}",
            f"{_area_mean(fields['h'], fields['areas']):.16g}",
            f"{fields['h'].min():.16g}", f"{fields['h'].max():.16g}",
            f"{_area_mean(fields['air_temperature'], fields['areas']):.16g}",
            f"{_area_mean(fields['surface_temperature'], fields['areas']):.16g}",
            f"{fields['surface_temperature'].min():.16g}",
            f"{fields['surface_temperature'].max():.16g}",
            f"{_area_mean(q_dot, fields['areas']):.16g}", f"{q_dot.min():.16g}",
            f"{q_dot.max():.16g}", f"{np.sum(q_dot * fields['areas']):.16g}",
            f"{_area_mean(target_temperature, fields['areas']):.16g}",
            f"{target_temperature.min():.16g}", f"{target_temperature.max():.16g}",
            f"{_area_mean(relaxed_temperature, fields['areas']):.16g}",
            f"{relaxed_temperature.min():.16g}", f"{relaxed_temperature.max():.16g}",
            f"{environment_power:.16g}", f"{body_power:.16g}", f"{time_error:.16g}",
        ])
    zone_area = network.zone_area
    zone_ta = np.zeros(17)
    zone_h = np.zeros(17)
    zone_surface = np.zeros(17)
    np.add.at(zone_ta, network.mapping.zone_ids, fields["air_temperature"] * fields["areas"])
    np.add.at(zone_h, network.mapping.zone_ids, fields["h"] * fields["areas"])
    np.add.at(zone_surface, network.mapping.zone_ids, relaxed_temperature * fields["areas"])
    zone_ta = np.divide(zone_ta, zone_area, out=np.zeros(17), where=zone_area > 0)
    zone_h = np.divide(zone_h, zone_area, out=np.zeros(17), where=zone_area > 0)
    zone_surface = np.divide(zone_surface, zone_area, out=np.zeros(17), where=zone_area > 0)
    with ZONE_TRACE.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for zone_id, zone_name in enumerate(jos3_embedded.BODY_NAMES):
            writer.writerow([
                exchange_id, f"{cfd_time_s:.16g}", f"{jos3_time_s:.16g}", zone_id,
                zone_name, f"{zone_area[zone_id]:.16g}", f"{zone_ta[zone_id]:.16g}",
                f"{zone_h[zone_id]:.16g}", f"{exchange.zone_body_power[zone_id]:.16g}",
                f"{zone_surface[zone_id]:.16g}",
            ])
    print(
        f"step: faces={n} dt={cfd_dtime:.6g} alpha={surface_relaxation:.3g} h=[{fields['h'].min():.4g},{fields['h'].max():.4g}] "
        f"Ttarget=[{target_temperature.min():.4g},{target_temperature.max():.4g}] "
        f"Treturn=[{relaxed_temperature.min():.4g},{relaxed_temperature.max():.4g}]",
        flush=True,
    )

print("FoamPilot JOS-3 driver terminé.")
