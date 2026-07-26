#!/usr/bin/env python3
"""Tutoriel 2 : Écoulement turbulent autour d'un véhicule simplifié (simpleFoam).

Référence OpenFOAM-14 : tutorials/incompressible/simpleFoam/simpleCar
https://develop.openfoam.com/Development/openfoam/-/tree/master/tutorials/incompressible/simpleFoam/simpleCar

Cet exemple illustre la configuration d'une simulation turbulente en régime
stationnaire avec le solveur simpleFoam et le modèle k-omega SST.
"""

from pathlib import Path
from foampilot.solver import Solver
from foampilot import ValueWithUnit


def main():
    case_path = Path.cwd()

    # --- 1. Initialiser le solveur ---
    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = False
    solver.turbulence_model = "kOmegaSST"

    # --- 2. Écrire les dictionnaires système et constantes ---
    solver.system.write()
    solver.constant.write()

    # --- 3. Conditions aux limites ---
    solver.boundary.initialize_boundary()
    solver.boundary.apply_condition_with_wildcard(
        pattern="inlet",
        condition_type="velocityInlet",
        velocity=(ValueWithUnit(30, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
        turbulence_intensity=0.05,
    )
    solver.boundary.apply_condition_with_wildcard(
        pattern="outlet",
        condition_type="pressureOutlet",
    )
    solver.boundary.apply_condition_with_wildcard(
        pattern="walls",
        condition_type="wall",
    )
    solver.boundary.apply_condition_with_wildcard(
        pattern="farfield",
        condition_type="freestream",
        velocity=(ValueWithUnit(30, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
        turbulence_intensity=0.01,
    )
    solver.boundary.write_boundary_conditions()

    # --- 4. Lancer la simulation ---
    solver.run_simulation(nb_proc=1)

    # --- 5. Post-traitement ---
    # Visualiser les lignes de courant, le champ de pression et Cp


if __name__ == "__main__":
    main()