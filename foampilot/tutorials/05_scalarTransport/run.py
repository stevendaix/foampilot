#!/usr/bin/env python3
"""Tutoriel 5 : Transport de scalaire passif (scalarTransportFoam).

Référence OpenFOAM-14 : tutorials/scalarTransportFoam/scalarTransport
https://develop.openfoam.com/Development/openfoam/-/tree/master/tutorials/scalarTransportFoam/scalarTransport

Transport d'un champ scalaire passif (température, concentration) dans un
écoulement précalculé ou simultané.
"""

from pathlib import Path
from foampilot.solver import Solver
from foampilot import ValueWithUnit


def main():
    case_path = Path.cwd()

    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = False
    solver.turbulence_model = "laminar"
    solver.energy_activated = True

    solver.system.write()
    solver.constant.write()

    solver.boundary.initialize_boundary()
    solver.boundary.apply_condition_with_wildcard(
        pattern="inlet",
        condition_type="velocityInlet",
        velocity=(ValueWithUnit(1, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
    )
    solver.boundary.apply_condition_with_wildcard(
        pattern="walls",
        condition_type="wall",
    )
    solver.boundary.write_boundary_conditions()

    solver.run_simulation(nb_proc=1)


if __name__ == "__main__":
    main()