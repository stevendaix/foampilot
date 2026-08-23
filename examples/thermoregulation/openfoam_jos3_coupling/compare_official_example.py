#!/usr/bin/env python3
"""Comparaison numérique avec l'exemple officiel JOS-3 example_v2.py."""
from pathlib import Path
import importlib.util
import sys
import types

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
SRC = ROOT / "foampilot" / "src"
JOS3_SRC = HERE.parents[4] / "JOS-3" / "src"
sys.path[:0] = [str(SRC), str(JOS3_SRC)]

import jos3 as reference_jos3

# Chargement ciblé de la copie FoamPilot sans importer les extensions CAD globales.
sys.modules.setdefault("foampilot", types.ModuleType("foampilot"))
physiology_spec = importlib.util.spec_from_file_location(
    "foampilot.physiology",
    SRC / "foampilot" / "physiology" / "__init__.py",
    submodule_search_locations=[str(SRC / "foampilot" / "physiology")],
)
physiology = importlib.util.module_from_spec(physiology_spec)
sys.modules[physiology_spec.name] = physiology
physiology_spec.loader.exec_module(physiology)
EmbeddedJOS3 = physiology.JOS3
JOS3NodeCoupler = physiology.JOS3NodeCoupler
SurfaceMapping = physiology.SurfaceMapping


def configure(model):
    model.Ta = 28
    model.Tr = 30
    model.RH = 40
    model.Va = 0.2
    model.PAR = 1.2
    model.posture = "sitting"
    model.Icl = np.array([
        0.00, 0.00, 1.14, 0.84, 1.04, 0.84, 0.42, 0.00,
        0.84, 0.42, 0.00, 0.58, 0.62, 0.82, 0.58, 0.62, 0.82,
    ])


def run_official(Model):
    model = Model(
        height=1.7, weight=60, fat=20, age=30, sex="male",
        bmr_equation="japanese", bsa_equation="fujimoto", ex_output="all",
    )
    configure(model)
    model.simulate(times=30, dtime=60)
    model.To = 20
    model.Va = np.array([
        0.2, 0.4, 0.4, 0.1, 0.1, 0.4, 0.4, 0.4, 0.4,
        0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    ])
    model.simulate(times=60, dtime=60)
    model.Ta = 30
    model.Tr = 35
    model.simulate(times=30, dtime=60)
    return model


def history_frame(model, label):
    data = model.dict_results()
    return pd.DataFrame({
        "time_min": np.asarray(data["CycleTime"]) * 60.0 / 60.0,
        f"{label}_TskMean": np.asarray(data["TskMean"], dtype=float),
        f"{label}_TskHead": np.asarray(data["TskHead"], dtype=float),
        f"{label}_TskChest": np.asarray(data["TskChest"], dtype=float),
        f"{label}_TskLFoot": np.asarray(data["TskLFoot"], dtype=float),
        f"{label}_TcrChest": np.asarray(data["TcrChest"], dtype=float),
    })


def compare_columns(left, right):
    diff = left - right
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "final_abs": float(abs(diff[-1])),
    }


def main():
    original = run_official(reference_jos3.JOS3)
    embedded = run_official(EmbeddedJOS3)
    ref = history_frame(original, "original")
    emb = history_frame(embedded, "embedded")
    merged = ref.merge(emb, on="time_min")

    metrics = {}
    for name in ["TskMean", "TskHead", "TskChest", "TskLFoot", "TcrChest"]:
        metrics[name] = compare_columns(
            merged[f"original_{name}"].to_numpy(),
            merged[f"embedded_{name}"].to_numpy(),
        )

    # Validation supplémentaire du chemin couplé : un point CFD par zone,
    # flux nul, mais les températures d'air sont bien transmises à JOS-3.
    coupled = EmbeddedJOS3(
        height=1.7, weight=60, fat=20, age=30, sex="male",
        bmr_equation="japanese", bsa_equation="fujimoto", ex_output="all",
    )
    configure(coupled)
    mapping = SurfaceMapping(zone_ids=np.arange(17), areas=np.ones(17))
    coupler = JOS3NodeCoupler(coupled, mapping)
    zero_h = np.full(17, 10.0)
    zero_ta = np.full(17, 28.0)
    coupler.run_steady(
        {"h": zero_h, "surface_temperature": zero_ta, "air_temperature": zero_ta},
        dtime=60.0, steps=30,
    )
    coupled.To = 20
    coupled.Va = np.array([
        0.2, 0.4, 0.4, 0.1, 0.1, 0.4, 0.4, 0.4, 0.4,
        0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    ])
    phase2_ta = np.full(17, 20.0)
    coupler.run_steady(
        {"h": zero_h, "surface_temperature": phase2_ta, "air_temperature": phase2_ta},
        dtime=60.0, steps=60,
    )
    coupled.Ta = 30
    coupled.Tr = 35
    phase3_ta = np.full(17, 30.0)
    coupler.run_steady(
        {"h": zero_h, "surface_temperature": phase3_ta, "air_temperature": phase3_ta},
        dtime=60.0, steps=30,
    )
    coupled_df = history_frame(coupled, "coupled_zero_flux")
    merged = merged.merge(coupled_df, on="time_min", how="left")
    valid = merged[["embedded_TskMean", "coupled_zero_flux_TskMean"]].dropna()
    metrics["coupled_zero_flux_TskMean"] = compare_columns(
        valid["embedded_TskMean"].to_numpy(),
        valid["coupled_zero_flux_TskMean"].to_numpy(),
    )

    out = HERE.parent / "comparison_results"
    out.mkdir(exist_ok=True)
    merged.to_csv(out / "official_example_comparison.csv", index=False)
    pd.DataFrame(metrics).T.to_csv(out / "comparison_metrics.csv")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(merged.time_min, merged.original_TskMean, label="JOS-3 original", lw=2)
    axes[0].plot(merged.time_min, merged.embedded_TskMean, "--", label="JOS-3 embarqué")
    axes[0].plot(merged.time_min, merged.coupled_zero_flux_TskMean, ":", label="Couplage mémoire, flux nul")
    axes[0].set_ylabel("TskMean [°C]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(merged.time_min, merged.original_TskHead, label="Head")
    axes[1].plot(merged.time_min, merged.original_TskChest, label="Chest")
    axes[1].plot(merged.time_min, merged.original_TskLFoot, label="LFoot")
    axes[1].set_xlabel("Temps [min]")
    axes[1].set_ylabel("Température cutanée [°C]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "official_example_comparison.png", dpi=160)
    plt.close(fig)

    report = [
        "# Rapport de comparaison JOS-3",
        "",
        "L'exemple officiel `JOS-3/example/example_v2.py` a été repris sans modifier ses paramètres : morphologie, conditions 28/30 °C, phase à 20 °C, ventilation variable, vêtements, posture et pas de 60 s.",
        "",
        "## Critères",
        "",
        "Les métriques sont calculées sur les séries temporelles de température moyenne cutanée, tête, thorax, pied gauche et température centrale du thorax. L'écart est évalué sur les mêmes instants : maximum absolu, moyenne absolue, RMSE et écart final.",
        "",
        "| Sortie | max absolu [°C] | moyenne absolue [°C] | RMSE [°C] | final [°C] |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in metrics.items():
        report.append(f"| {name} | {values['max_abs']:.3e} | {values['mean_abs']:.3e} | {values['rmse']:.3e} | {values['final_abs']:.3e} |")
    report += [
        "",
        "## Validation du chemin couplé",
        "",
        f"Pour le cas à 17 points, avec `h=10 W/m²/K` et `T_surface=T_air` dans chaque phase, le flux CFD est nul. L'écart RMSE entre la copie embarquée et le chemin couplé est de {metrics['coupled_zero_flux_TskMean']['rmse']:.3e} °C sur les 121 instants comparables.",
        "",
        "Les données complètes sont dans `official_example_comparison.csv`, les métriques dans `comparison_metrics.csv` et la figure dans `official_example_comparison.png`.",
    ]
    (out / "comparison_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Comparaison officielle terminée.")
    print(pd.DataFrame(metrics).T.to_string(float_format=lambda x: f"{x:.3e}"))


if __name__ == "__main__":
    main()
