#!/usr/bin/env python3
"""Audit des commits Direct Commit par cloud et par espèce."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
COMMIT = re.compile(
    r"VOF direct commit cloud=(\w+) fragmentId=(\w+) "
    r"success=(true|false) mass=(%s)" % NUM
)
CONF = re.compile(
    r"VOF confirmation cloud=(\w+) alphaField=([A-Za-z0-9_.]+) "
    r"fragmentId=(\w+) success=(true|false) mass=(%s) "
    r"speciesMass=\d+\(([^)]*)\)" % NUM
)

def floats(text: str) -> list[float]:
    return [float(x) for x in re.findall(NUM, text)]

def close(a: float, b: float) -> bool:
    # OpenFOAM prints scalarList values with limited precision in Info output.
    return abs(a - b) <= 1e-6 * max(abs(a), abs(b), 1.0)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--expected", type=Path, required=True)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    text = a.log.read_text(errors="replace")
    expected = json.loads(a.expected.read_text())
    commits = COMMIT.findall(text)
    confirmations = CONF.findall(text)

    commit_keys = [(c, f) for c, f, ok, mass in commits if ok == "true"]
    confirm_keys = [(c, f) for c, alpha, f, ok, mass, species in confirmations if ok == "true"]
    expected_keys = [tuple(key.split(":", 1)) for key in expected["fragments"]]

    species_ok = True
    species_audit = []
    for cloud, alpha, fragment, ok, mass, species_text in confirmations:
        key = f"{cloud}:{fragment}"
        expected_fragment = expected["fragments"].get(key)
        actual_species = floats(species_text)
        expected_species = []
        if expected_fragment is not None:
            expected_species = [
                float(expected_fragment["speciesMass"][name])
                for name in expected["species"]
            ] if "species" in expected_fragment else [
                float(expected_fragment["speciesMass"][name])
                for name in expected.get("species", [])
            ]
        # Accept the top-level species ordering used by this case.
        if expected_fragment is not None and not expected_species:
            expected_species = [
                float(expected_fragment["speciesMass"][name])
                for name in expected.get("species", [])
            ]
        match = ok == "true" and len(actual_species) == len(expected_species)
        if match:
            match = all(close(x, y) for x, y in zip(actual_species, expected_species))
        species_ok = species_ok and match
        species_audit.append({"key": key, "actual": actual_species, "expected": expected_species, "pass": match})

    result = {
        "expectedKeys": expected_keys,
        "commitKeys": commit_keys,
        "confirmationKeys": confirm_keys,
        "speciesAudit": species_audit,
        "checks": {
            "solverEnd": bool(re.search(r"^End$", text, re.M)),
            "noFatalOrMPI": not bool(re.search(r"FOAM FATAL|Floating point exception|MPI_ERR", text)),
            "allExpectedCommittedExactlyOnce": sorted(commit_keys) == sorted(expected_keys) and len(commit_keys) == len(set(commit_keys)),
            "allExpectedConfirmedExactlyOnce": sorted(confirm_keys) == sorted(expected_keys) and len(confirm_keys) == len(set(confirm_keys)),
            "speciesMassesConserved": species_ok,
            "noDefaultCloudFallback": not bool(re.search(r"default single parcel cloud|requested .* with 2 clouds", text, re.I)),
            "twoSpeciesConfigured": expected.get("species") == ["H2O", "C2H5OH"],
        },
    }
    result["pass"] = all(result["checks"].values())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
