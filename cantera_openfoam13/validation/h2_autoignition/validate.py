#!/usr/bin/env python3
"""Generate a Cantera reference and optionally validate an OpenFOAM run."""
from __future__ import annotations
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

try:
    import cantera as ct
except ImportError as exc:
    raise SystemExit("Cantera est requis: sudo pip3 install cantera") from exc

gas = ct.Solution("gri30.yaml")
gas.TPX = 1000.0, ct.one_atm, "H2:2,O2:1,N2:3.76"
reactor = ct.IdealGasReactor(gas, energy="on")
network = ct.ReactorNet([reactor])
records = []
for time in [i * 2.0e-5 for i in range(101)]:
    network.advance(time)
    records.append((time, reactor.T, reactor.thermo["OH"].Y[0]))
with (RESULTS / "cantera_reference.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["time_s", "temperature_K", "OH_mass_fraction"])
    writer.writerows(records)

peak = max(records, key=lambda row: row[2])
print(f"Cantera reference: peak OH at t={peak[0]:.8g} s, T={peak[1]:.8g} K")

block_mesh = shutil.which("blockMesh")
ico_foam = shutil.which("icoFoam")
case = ROOT / "openfoam_case"
if not block_mesh or not ico_foam or not (case / "system" / "controlDict").exists():
    print("OpenFOAM validation skipped: source /opt/openfoam13/etc/bashrc then retry.")
    sys.exit(0)

for executable, name in ((block_mesh, "blockMesh"), (ico_foam, "icoFoam")):
    log = RESULTS / f"{name}.log"
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run([executable, "-case", str(case)], stdout=stream, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"{name} failed with return code {completed.returncode}; see {log}")
    text = log.read_text(encoding="utf-8", errors="replace")
    if "FOAM FATAL ERROR" in text:
        raise SystemExit(f"{name} reported FOAM FATAL ERROR; see {log}")
print("OpenFOAM validation passed: blockMesh and icoFoam")
