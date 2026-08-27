#!/usr/bin/env python3
"""Strict validation of the Cantera/OpenFOAM 13 bridge."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import shutil
import subprocess
import sys
import cantera as ct

ROOT = Path(__file__).resolve().parent
CASE = ROOT / "openfoam_case"
RESULTS = ROOT / "results"
REFERENCE = ROOT / "reference" / "cantera_reference.csv"
RESULTS.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--skip-openfoam", action="store_true", help="only check the frozen Cantera reference")
args = parser.parse_args()
if not REFERENCE.exists():
    raise SystemExit(f"Missing frozen reference: {REFERENCE}")

with REFERENCE.open(newline="", encoding="utf-8") as stream:
    reference = list(csv.DictReader(stream))
if len(reference) != 101:
    raise SystemExit(f"Unexpected reference length: {len(reference)}")
peak = max(reference, key=lambda row: float(row["OH_mass_fraction"]))
print(f"Frozen Cantera reference: peak OH at t={float(peak['time_s']):.8g} s, T={float(peak['temperature_K']):.8g} K")

if args.skip_openfoam:
    raise SystemExit(0)
block_mesh = shutil.which("blockMesh")
ico_foam = shutil.which("icoFoam")
cantera_foam = shutil.which("canteraFoam")
if not block_mesh or not ico_foam or not cantera_foam:
    raise SystemExit("OpenFOAM validation unavailable: blockMesh, icoFoam and canteraFoam are required")
for executable, name in ((block_mesh, "blockMesh"), (cantera_foam, "canteraFoam"), (ico_foam, "icoFoam")):
    log = RESULTS / f"{name}.log"
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run([executable, "-case", str(CASE)], stdout=stream, stderr=subprocess.STDOUT, check=False)
    text = log.read_text(encoding="utf-8", errors="replace")
    if completed.returncode != 0 or "FOAM FATAL ERROR" in text:
        raise SystemExit(f"{name} failed; see {log}")

out = CASE / "canteraThermo.csv"
if not out.exists():
    raise SystemExit(f"Missing bridge output: {out}")
with out.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
required = {"cell", "T_eq", "p_eq", "rho", "cp_mass", "thermal_conductivity", "OH_mass_fraction"}
if len(rows) != 1000 or not required <= set(rows[0]):
    raise SystemExit(f"Invalid canteraFoam output: {len(rows)} rows / columns={set(rows[0]) if rows else set()}")

gas = ct.Solution("gri30.yaml")
gas.TPX = 1000.0, ct.one_atm, "H2:2,O2:1,N2:3.76"
gas.equilibrate("HP")
expected = {"T_eq": gas.T, "p_eq": gas.P, "OH_mass_fraction": gas["OH"].Y[0]}
for key, value in expected.items():
    values = [float(row[key]) for row in rows]
    spread = max(values) - min(values)
    if spread > 1e-9 * max(1.0, abs(value)):
        raise SystemExit(f"Non-uniform {key}: spread={spread}")
    if abs(values[0] - value) > 1e-6 * max(1.0, abs(value)):
        raise SystemExit(f"{key} mismatch: got={values[0]}, expected={value}")
print(f"Bridge comparison passed: {len(rows)} cells, T_eq={expected['T_eq']:.8g} K, OH_eq={expected['OH_mass_fraction']:.8g}")
print("OpenFOAM validation passed: blockMesh, canteraFoam and icoFoam")
