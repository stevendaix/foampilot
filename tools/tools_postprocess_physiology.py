from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "examples/thermoregulation/openfoam_jos3_coupling/comparison_results/official_example_comparison.csv"
OUT = ROOT / "examples/thermoregulation/postprocess_results"
OPENFOAM_RUNS = ROOT / "openfoam_runs"

SIGNALS = ["TskMean", "TskHead", "TskChest", "TskLFoot", "TcrChest"]


def error_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    result = {}
    for signal in SIGNALS:
        error = frame[f"embedded_{signal}"] - frame[f"original_{signal}"]
        result[signal] = {
            "max_abs_C": float(np.max(np.abs(error))),
            "mean_abs_C": float(np.mean(np.abs(error))),
            "rmse_C": float(np.sqrt(np.mean(error**2))),
            "final_abs_C": float(abs(error.iloc[-1])),
        }
    zero_flux = frame["coupled_zero_flux_TskMean"] - frame["embedded_TskMean"]
    result["coupled_zero_flux_TskMean"] = {
        "max_abs_C": float(np.max(np.abs(zero_flux))),
        "mean_abs_C": float(np.mean(np.abs(zero_flux))),
        "rmse_C": float(np.sqrt(np.mean(zero_flux**2))),
        "final_abs_C": float(abs(zero_flux.iloc[-1])),
    }
    return result


def openfoam_summary() -> dict[str, dict[str, object]]:
    result = {}
    for case in ("buoyantCavity_validation", "coolingSphere_validation"):
        case_dir = OPENFOAM_RUNS / case
        times = []
        if case_dir.exists():
            for child in case_dir.iterdir():
                if child.is_dir():
                    try:
                        times.append(float(child.name))
                    except ValueError:
                        pass
        result[case] = {
            "exists": case_dir.exists(),
            "time_directories": len(times),
            "final_time": max(times) if times else None,
            "has_log": (case_dir / "log.foamRun").exists() or any(case_dir.glob("log.*")),
        }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA)
    metrics = error_metrics(frame)
    summary = {
        "source": str(DATA.relative_to(ROOT)),
        "samples": int(len(frame)),
        "time_min": {
            "start": float(frame.time_min.iloc[0]),
            "end": float(frame.time_min.iloc[-1]),
        },
        "jos3_metrics": metrics,
        "openfoam": openfoam_summary(),
    }
    (OUT / "postprocess_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    time = frame["time_min"]
    for signal in SIGNALS:
        axes[0].plot(time, frame[f"original_{signal}"], label=signal)
    axes[0].set_ylabel("Température (°C)")
    axes[0].set_title("JOS3 officiel : températures suivies")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8)

    for signal in SIGNALS:
        axes[1].plot(time, frame[f"embedded_{signal}"] - frame[f"original_{signal}"], label=signal)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Erreur (°C)")
    axes[1].set_title("Écart copie FoamPilot – référence officielle")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time, frame["embedded_TskMean"], label="Embedded")
    axes[2].plot(time, frame["coupled_zero_flux_TskMean"], "--", label="Couplage, flux nul")
    axes[2].set_xlabel("Temps (min)")
    axes[2].set_ylabel("TskMean (°C)")
    axes[2].set_title("Contrôle du couplage à flux nul")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(OUT / "jos3_comparison_postprocess.png", dpi=180)
    plt.close(fig)

    metrics_frame = pd.DataFrame(metrics).T
    metrics_frame.to_csv(OUT / "jos3_metrics_postprocess.csv", index_label="signal")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
