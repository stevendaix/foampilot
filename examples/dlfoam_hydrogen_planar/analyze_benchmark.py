#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "benchmark_results"
records = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
serial = next(r for r in records if r["nproc"] == 1)
base = float(serial["foam_execution_seconds"])
base_wall = float(serial["wall_seconds_solver_command"])
for row in records:
    row["speedup_foam"] = base / float(row["foam_execution_seconds"])
    row["efficiency_foam"] = row["speedup_foam"] / int(row["nproc"])
    row["speedup_wall"] = base_wall / float(row["wall_seconds_solver_command"])
    row["efficiency_wall"] = row["speedup_wall"] / int(row["nproc"])

plt.figure(figsize=(7, 4.2))
plt.plot([r["nproc"] for r in records], [r["speedup_foam"] for r in records], "o-", label="ExecutionTime OpenFOAM")
plt.plot([r["nproc"] for r in records], [r["speedup_wall"] for r in records], "s--", label="Temps mur commande")
plt.plot([r["nproc"] for r in records], [r["nproc"] for r in records], ":", label="Idéal")
plt.xlabel("Nombre de rangs MPI")
plt.ylabel("Speedup relatif à np=1")
plt.xticks([r["nproc"] for r in records])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "mpi_speedup.png", dpi=160)

lines = [
    "# Benchmark MPI de multicomponentFluid sous OpenFOAM 13",
    "",
    "Le benchmark utilise le cas Foampilot DLBFoam hydrogène avec une résolution contrôlée de 2×2 cellules et une seule itération (`endTime = 1.1e-6`). Cette taille est destinée à vérifier le chemin MPI et ne permet pas d’extrapoler les performances du cas scientifique 2000×2000. Les bibliothèques DLBFoam, FickianTransportFoam et PyJac sont compilées avant chaque run.",
    "",
    "## Résultats",
    "",
    "| Rangs MPI | Cellules | Temps OpenFOAM (s) | Temps mur (s) | Speedup OpenFOAM | Efficacité OpenFOAM | Terminé |",
    "|---:|---:|---:|---:|---:|---:|:---:|",
]
for r in records:
    lines.append(f"| {r['nproc']} | {r['cells']} | {float(r['foam_execution_seconds']):.6f} | {float(r['wall_seconds_solver_command']):.3f} | {r['speedup_foam']:.3f} | {100*r['efficiency_foam']:.1f}% | {'oui' if r['completed'] else 'non'} |")
lines += [
    "",
    "![Speedup MPI](mpi_speedup.png)",
    "",
    "## Interprétation",
    "",
    f"Le temps interne OpenFOAM passe de {base:.6f} s en série à {float(records[-1]['foam_execution_seconds']):.6f} s avec {records[-1]['nproc']} rangs, soit un speedup de {records[-1]['speedup_foam']:.2f} et une efficacité de {100*records[-1]['efficiency_foam']:.1f} %. Le temps mur de la commande reste dominé par la compilation dynamique des codeStreams et l’initialisation répétée du mécanisme ; il ne doit donc pas être utilisé comme mesure principale de scaling.",
    "",
    "La mesure est fonctionnelle : chaque run se termine avec un log `End` et les résultats sont produits par le même solveur `foamRun -solver multicomponentFluid`. Toutefois, le maillage de 4 cellules est beaucoup trop petit pour une conclusion de performance physique. Pour une étude de scaling exploitable, il faut conserver les paramètres de référence ou au minimum utiliser un maillage suffisamment grand pour que la chimie et les communications MPI dominent les coûts fixes, puis exécuter plusieurs répétitions par nombre de rangs.",
    "",
    "## Reproduction",
    "",
    "```bash",
    "export FOAM_BASHRC=/chemin/vers/OpenFOAM-13/etc/bashrc",
    "source \"$FOAM_BASHRC\"",
    "cd examples/dlfoam_hydrogen_planar",
    "python3 benchmark_mpi.py",
    "python3 analyze_benchmark.py",
    "```",
    "",
    "## Références",
    "",
    "[1]: https://openfoam.org/download/13-ubuntu/ OpenFOAM 13 — installation Ubuntu",
    "[2]: https://github.com/Aalto-CFD/DLBFoam DLBFoam — branche OpenFOAM 13",
]
(OUT / "mpi_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(OUT / "results_analyzed.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
print((OUT / "mpi_report.md"))
