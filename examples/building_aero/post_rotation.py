#!/usr/bin/env python
from pathlib import Path
import json
import pyvista as pv

from foampilot import postprocess
from lawson import (
    WindRose,
    WindCaseResult,
    WindEnsemble,
    LawsonProcessor,
    LawsonVisualizer,
)

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

CASE_ROOT = Path.cwd()
POST_DIR = CASE_ROOT / "post"
POST_DIR.mkdir(exist_ok=True)

REFERENCE_HEIGHT_SPEED = 10.0  # m/s (vitesse inlet CFD normalisée)
SECTOR_HALF_WIDTH = 15.0       # ±15° → secteurs de 30°

# -------------------------------------------------
# 1. CHARGEMENT ROSE DES VENTS
# -------------------------------------------------

def load_wind_rose(fp: Path) -> WindRose:
    """
    JSON attendu :
    {
        "0": [{"speed": 5, "frequency": 0.12}],
        "30": [{"speed": 8, "frequency": 0.08}]
    }
    """
    with open(fp, "r") as f:
        data = json.load(f)

    # directions en float
    data = {float(k): v for k, v in data.items()}
    return WindRose(data)


wind_rose = load_wind_rose(CASE_ROOT / "wind_rose.json")

# -------------------------------------------------
# 2. CONSTRUCTION ENSEMBLE CFD
# -------------------------------------------------

ensemble = WindEnsemble()
cases_dir = CASE_ROOT / "cases"

for case_dir in cases_dir.iterdir():
    if not case_dir.is_dir():
        continue

    # Convention : wind_30deg
    direction = float(
        case_dir.name
        .replace("wind_", "")
        .replace("deg", "")
    )

    foam_post = postprocess.FoamPostProcessing(case_dir)

    case = WindCaseResult(
        post=foam_post,
        direction_deg=direction,
        u_ref=REFERENCE_HEIGHT_SPEED,
        field_name="U",
    )

    ensemble.add_case(direction, case)

# Lancement post-traitement CFD
ensemble.run_all()

# -------------------------------------------------
# 3. TRAITEMENT LAWSON (AVEC REGROUPEMENT SECTORIEL)
# -------------------------------------------------

lawson = LawsonProcessor(
    ensemble=ensemble,
    wind_rose=wind_rose,
    sector_half_width=SECTOR_HALF_WIDTH,  # <<< MODIFICATION ICI
)

lawson_maps = lawson.compute_lawson_maps()

# -------------------------------------------------
# 4. VISUALISATION
# -------------------------------------------------

reference_case = next(iter(ensemble.cases.values()))
reference_mesh = reference_case.mesh

viz = LawsonVisualizer(reference_mesh)

for label, data in lawson_maps.items():
    viz.add_probability_field(f"lawson_{label}", data)

    viz.plot(
        field=f"lawson_{label}",
        filename=POST_DIR / f"lawson_{label}.png",
    )

reference_mesh.save(POST_DIR / "lawson_results.vtp")

print("Post-traitement Lawson terminé.")