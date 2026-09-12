#!/usr/bin/env python3
"""Audit conservatif du transfert VOF -> thermoCloud sur un log OpenFOAM."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def values(text: str, patterns: list[str]) -> list[float]:
    result: list[float] = []
    for pattern in patterns:
        result.extend(float(value) for value in re.findall(pattern, text))
    return result


def count_any(text: str, patterns: list[str]) -> int:
    return sum(len(re.findall(pattern, text, flags=re.MULTILINE)) for pattern in patterns)


def last(values_: list[float]) -> float | None:
    return values_[-1] if values_ else None


def relative_error(actual: float | None, expected: float | None) -> float | None:
    if actual is None or expected is None:
        return None
    return abs(actual - expected) / max(abs(expected), 1.0e-30)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--case", type=Path, default=Path("."))
    parser.add_argument("--mass-tol", type=float, default=1.0e-8)
    parser.add_argument("--energy-tol", type=float, default=1.0e-8)
    parser.add_argument("--rho-liquid", type=float, default=1000.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.log.is_file():
        print(f"ERROR: log absent: {args.log}", file=sys.stderr)
        return 2

    text = args.log.read_text(errors="replace")

    detected_volumes = values(text, [
        rf"local convertible volume\s*:\s*({NUMBER})",
        rf"detectedVolume\s*[=:]\s*({NUMBER})",
    ])
    derived_mass_detected = last(detected_volumes)
    if derived_mass_detected is not None:
        derived_mass_detected *= args.rho_liquid

    mass_detected_values = values(text, [
        rf"(?:detectedMass|mass detected|massDetected)\s*[=:]\s*({NUMBER})",
    ])
    mass_prepared_values = values(text, [
        rf"(?:preparedMass|mass prepared|massPrepared)\s*[=:]\s*({NUMBER})",
    ])
    mass_created_values = values(text, [
        rf"(?:createdMass|mass created|massCreated|mass introduced)\s*[=:]\s*({NUMBER})",
    ])
    mass_confirmed_values = values(text, [
        rf"(?:confirmedMass|mass confirmed|massConfirmed)\s*[=:]\s*({NUMBER})",
    ])

    energy_detected_values = values(text, [
        rf"(?:enthalpyDetected|detectedEnthalpy|enthalpy detected|energyDetected)\s*[=:]\s*({NUMBER})",
    ])
    energy_created_values = values(text, [
        rf"(?:enthalpyCreated|createdEnthalpy|enthalpy created|energyCreated)\s*[=:]\s*({NUMBER})",
    ])
    energy_confirmed_values = values(text, [
        rf"(?:enthalpyConfirmed|confirmedEnthalpy|enthalpy confirmed|energyConfirmed)\s*[=:]\s*({NUMBER})",
    ])

    mass_detected = last(mass_detected_values)
    mass_prepared = last(mass_prepared_values)
    mass_created = last(mass_created_values)
    mass_confirmed = last(mass_confirmed_values)
    energy_detected = last(energy_detected_values)
    energy_created = last(energy_created_values)
    energy_confirmed = last(energy_confirmed_values)

    parcel_created = count_any(text, [r"parcelCreated", r"Added\s+1\s+new parcels", r"VOF\s+direct\s+commit.*?success\s*=\s*(?:1|true)"])
    enthalpy_sources = count_any(text, [r"Applied compressible enthalpy transfer"])
    alpha_sources = count_any(text, [r"Applied compressible alphaRho transfer"])
    solver_end = bool(re.search(r"^End$", text, re.MULTILINE))
    fatal = bool(re.search(r"FOAM FATAL|Floating point exception|MPI_ERR", text))

    checks: dict[str, bool] = {
        "solver_end": solver_end,
        "no_fatal_or_mpi_error": not fatal,
        "parcel_created_seen": parcel_created > 0,
        "enthalpy_source_seen": enthalpy_sources > 0,
    }

    if mass_detected is None and derived_mass_detected is not None:
        mass_detected = derived_mass_detected

    if mass_detected is not None and mass_prepared is not None:
        checks["detected_prepared_mass_balance"] = (
            relative_error(mass_prepared, mass_detected) <= args.mass_tol
        )
    if mass_created is not None and mass_confirmed is not None:
        checks["created_confirmed_mass_balance"] = (
            relative_error(mass_created, mass_confirmed) <= args.mass_tol
        )
    if energy_created is not None and energy_confirmed is not None:
        checks["created_confirmed_energy_balance"] = (
            relative_error(energy_created, energy_confirmed) <= args.energy_tol
        )

    # Les bilans physiques sont déclarés contrôlables uniquement si les
    # compteurs correspondants sont effectivement présents dans le log.
    required_metric_series = [
        mass_detected_values,
        mass_prepared_values,
        mass_created_values,
        mass_confirmed_values,
        energy_detected_values,
        energy_created_values,
        energy_confirmed_values,
    ]
    checks["explicit_mass_energy_metrics_present"] = all(
        len(series) > 0 for series in required_metric_series
    )
    checks["nonzero_mass_transfer_seen"] = any(
        value > 1.0e-30 for value in mass_confirmed_values
    )
    checks["nonzero_enthalpy_transfer_seen"] = any(
        value > 1.0e-30 for value in energy_confirmed_values
    )

    result = {
        "case": str(args.case),
        "log": str(args.log),
        "metrics": {
            "massDetected": mass_detected,
            "derivedMassDetectedFromVolume": derived_mass_detected,
            "massPrepared": mass_prepared,
            "massCreated": mass_created,
            "massConfirmed": mass_confirmed,
            "enthalpyDetected": energy_detected,
            "enthalpyCreated": energy_created,
            "enthalpyConfirmed": energy_confirmed,
            "parcelCreatedCount": parcel_created,
            "alphaRhoSourceApplications": alpha_sources,
            "enthalpySourceApplications": enthalpy_sources,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(result, indent=2))

    if not result["pass"]:
        missing = [name for name, ok in checks.items() if not ok]
        print("FAILED checks: " + ", ".join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
