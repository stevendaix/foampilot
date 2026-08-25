#!/usr/bin/env python3
"""Audit post-traité du bilan VOF-to-DPM pour le cas sprayCrossFlow."""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


def read_volume_series(path: Path) -> list[tuple[float, float]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 2:
            rows.append((float(fields[0]), float(fields[1])))
    if not rows:
        raise RuntimeError(f"Série volumique vide: {path}")
    return rows


def read_rho_liquid(cloud_path: Path) -> float:
    text = cloud_path.read_text()
    match = re.search(r"rhoLiquid\s+([0-9.eE+-]+)", text)
    if match:
        return float(match.group(1))
    match = re.search(r"rho0\s+([0-9.eE+-]+)", text)
    if match:
        return float(match.group(1))
    raise RuntimeError(f"Densité liquide absente de {cloud_path}")


def read_cloud_stats(log_path: Path) -> dict:
    text = log_path.read_text()
    times = [float(x) for x in re.findall(r"^Time = ([0-9.eE+-]+)s?$", text, re.MULTILINE)]
    fragment_volumes = [float(x) for x in re.findall(r"fragment 0 id \d+ volume ([0-9.eE+-]+)", text)]
    masses = [float(x) for x in re.findall(r"mass introduced\s+= ([0-9.eE+-]+)", text)]
    parcel_counts = [int(x) for x in re.findall(r"Current number of parcels\s+= (\d+)", text)]
    nonzero_mass = next((x for x in masses if x > 0), 0.0)
    if not fragment_volumes:
        raise RuntimeError("Aucun fragment VOF n’a été trouvé dans le journal")
    if nonzero_mass <= 0:
        raise RuntimeError("Aucune masse de parcel non nulle n’a été trouvée dans le journal")
    return {
        "times": times,
        "fragment_volume_first_m3": fragment_volumes[0],
        "fragment_volume_last_m3": fragment_volumes[-1],
        "parcel_mass_kg": nonzero_mass,
        "parcel_count_final": parcel_counts[-1] if parcel_counts else None,
    }


def main(case_dir: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    volume_path = output_dir / "liquidVolume" / "0" / "volFieldValue.dat"
    volume = read_volume_series(volume_path)
    cloud = read_cloud_stats(case_dir / "log.foamRun")
    rho_liquid = read_rho_liquid(case_dir / "constant" / "cloudProperties")
    expected_mass = cloud["fragment_volume_first_m3"] * rho_liquid
    error = abs(cloud["parcel_mass_kg"] - expected_mass) / expected_mass
    report = {
        "case": str(case_dir),
        "rho_liquid_kg_m3": rho_liquid,
        "vof_volume_initial_m3": volume[0][1],
        "vof_volume_final_m3": volume[-1][1],
        "vof_volume_series_points": len(volume),
        **cloud,
        "expected_parcel_mass_from_first_fragment_kg": expected_mass,
        "relative_conversion_mass_error": error,
        "conversion_mass_balance_pass": error < 1e-10,
        "solver_end_pass": "\nEnd\n" in (case_dir / "log.foamRun").read_text(),
    }
    (output_dir / "spray_balance.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output_dir / "spray_balance.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "vof_liquid_volume_m3"])
        writer.writerows(volume)

    try:
        import matplotlib.pyplot as plt
        times = [row[0] for row in volume]
        volumes = [row[1] for row in volume]
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        ax.plot(times, volumes, color="#1769aa", linewidth=1.8, label="∫ alpha.water dV")
        ax.axvline(times[-1], color="#777777", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Temps [s]")
        ax.set_ylabel("Volume liquide VOF [m³]")
        ax.set_title("Spray cross-flow : volume liquide résolu")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.savefig(output_dir / "spray_liquid_volume.png", dpi=160)
        plt.close(fig)
    except ImportError:
        report["plot_generated"] = False
        (output_dir / "spray_balance.json").write_text(json.dumps(report, indent=2) + "\n")
    else:
        report["plot_generated"] = True
        (output_dir / "spray_balance.json").write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))
    return 0 if report["conversion_mass_balance_pass"] and report["solver_end_pass"] else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: postprocess.py CASE_DIR OUTPUT_DIR")
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
