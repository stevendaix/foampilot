#!/usr/bin/env python3
"""Tutoriel 6 : Aérodynamique des bâtiments — quartier (simpleFoam).

Référence OpenFOAM-14 : tutorials/incompressible/simpleFoam/building
https://develop.openfoam.com/Development/openfoam/-/tree/master/tutorials/incompressible/simpleFoam/buildingAirFlow

Écoulement turbulent extérieur autour d'un quartier de bâtiments.
Utilise blockMesh pour créer le domaine et topoSet/createPatch pour les patchs.
"""

from pathlib import Path
from foampilot.solver import Solver
from foampilot import ValueWithUnit


def main():
    case_path = Path.cwd()

    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = False
    solver.turbulence_model = "kOmegaSST"

    solver.system.write()
    solver.constant.write()

    # topoSet pour définir les régions de bâtiments
    solver.system.run_topoSet()

    # createPatch pour renommer les patches
    solver.system.run_createPatch()

    solver.boundary.initialize_boundary()
    solver.boundary.apply_condition_with_wildcard(
        pattern="inlet",
        condition_type="velocityInlet",
        velocity=(ValueWithUnit(10, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
        turbulence_intensity=0.15,
    )
    solver.boundary.apply_condition_with_wildcard(
        pattern="outlet",
        condition_type="pressureOutlet",
    )
    solver.boundary.apply_condition_with_wildcard(
        pattern=".*building.*",
        condition_type="wall",
    )
    solver.boundary.write_boundary_conditions()

    solver.run_simulation(nb_proc=1)


if __name__ == "__main__":
    main()