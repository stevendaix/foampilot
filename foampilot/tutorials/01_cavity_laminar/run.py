#!/usr/bin/env python3
"""Tutoriel 1 : Écoulement laminaire dans une cavité entraînée (icoFoam).

Référence OpenFOAM-14 : tutorials/incompressible/icoFoam/cavity
https://develop.openfoam.com/Development/openfoam/-/tree/master/tutorials/incompressible/icoFoam/cavity

Cet exemple montre la mise en place complète d'un cas laminaire incompressible
avec foampilot : géométrie blockMesh, conditions aux limites, solveur et
post-traitement des résidus.
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
    solver.turbulence_model = "laminar"

    # --- 2. Écrire les dictionnaires système et constantes ---
    solver.system.write()
    solver.constant.write()

    # --- 3. Conditions aux limites ---
    solver.boundary.initialize_boundary()
    solver.boundary.apply_condition_with_wildcard(
        pattern="movingWall",
        condition_type="velocityInlet",
        velocity=(ValueWithUnit(1, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
    )
    solver.boundary.apply_condition_with_wildcard(
        pattern="fixedWalls",
        condition_type="wall",
    )
    solver.boundary.apply_condition_with_wildcard(
        pattern="frontAndBack",
        condition_type="symmetry",
    )
    solver.boundary.write_boundary_conditions()

    # --- 4. Lancer la simulation ---
    solver.run_simulation(nb_proc=1)

    # --- 5. Post-traitement ---
    # Les résultats sont dans le dossier VTK/ du cas


if __name__ == "__main__":
    main()