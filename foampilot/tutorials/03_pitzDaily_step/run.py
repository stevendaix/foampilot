#!/usr/bin/env python3
"""Tutoriel 3 : Écoulement sur marche descendante (pitzDaily, simpleFoam).

Référence OpenFOAM-14 : tutorials/incompressible/simpleFoam/pitzDaily
https://develop.openfoam.com/Development/openfoam/-/tree/master/tutorials/incompressible/simpleFoam/pitzDaily

Écoulement turbulent autour d'une marche descendante (backward-facing step).
Permet de valider le recouvrement de la recirculation.
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

    solver.boundary.initialize_boundary()
    solver.boundary.apply_condition_with_wildcard(
        pattern="inlet",
        condition_type="velocityInlet",
        velocity=(ValueWithUnit(1, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
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
    solver.boundary.write_boundary_conditions()

    solver.run_simulation(nb_proc=1)


if __name__ == "__main__":
    main()