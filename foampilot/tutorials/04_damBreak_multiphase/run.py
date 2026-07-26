#!/usr/bin/env python3
"""Tutoriel 4 : Cas de référence damBreak — écoulement multifluide VOF (interFoam).

Référence OpenFOAM-14 : tutorials/multiphase/interFoam/damBreak
https://develop.openfoam.com/Development/openfoam/-/tree/master/tutorials/multiphase/interFoam/damBreak

Cet exemple montre la modélisation d'un écoulement à deux phases (eau/air)
avec le modèle VOF et le solveur interFoam.
"""

from pathlib import Path
from foampilot.solver import Solver
from foampilot import ValueWithUnit


def main():
    case_path = Path.cwd()

    # --- 1. Initialiser le solveur VOF incompressible ---
    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = False
    solver.is_vof = True
    solver.turbulence_model = "laminar"

    # --- 2. Écrire les dictionnaires ---
    solver.system.write()
    solver.constant.write()

    # --- 3. Conditions aux limites ---
    solver.boundary.initialize_boundary()
    solver.boundary.apply_condition_with_wildcard(
        pattern="inlet",
        condition_type="velocityInlet",
        velocity=(ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
    )
    solver.boundary.apply_condition_with_wildcard(
        pattern="outlet",
        condition_type="pressureOutlet",
    )
    solver.boundary.apply_condition_with_wildcard(
        pattern="walls",
        condition_type="wall",
    )
    solver.boundary.write_boundary_conditions()

    # --- 4. Lancer ---
    solver.run_simulation(nb_proc=1)


if __name__ == "__main__":
    main()