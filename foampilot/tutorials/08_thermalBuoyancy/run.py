#!/usr/bin/env python3
"""Tutoriel 8 : Convection thermique naturelle (buoyantSimpleFoam).

Référence OpenFOAM-14 : tutorials/incompressible/buoyantSimpleFoam/room
https://develop.openfoam.com/Development/openfoam/-/tree/master/tutorials/incompressible/buoyantSimpleFoam/room

Écoulement thermo-buoyant dans une pièce chauffée.
La gravité active le couplage thermique-fluide via Boussinesq.
"""

from pathlib import Path
from foampilot.solver import Solver
from foampilot import ValueWithUnit


def main():
    case_path = Path.cwd()

    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = True
    solver.turbulence_model = "kEpsilon"

    solver.system.write()
    solver.constant.write()

    solver.boundary.initialize_boundary()
    solver.boundary.apply_condition_with_wildcard(
        pattern="inlet",
        condition_type="velocityInlet",
        velocity=(ValueWithUnit(0.1, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
    )
    solver.boundary.apply_condition_with_wildcard(
        pattern="walls",
        condition_type="wall",
    )
    # Hot wall at 350 K (isothermal patch)
    solver.boundary.set_raw_condition("hotWall", "T", {"type": "fixedValue", "value": "350"})
    # Cold wall at 300 K
    solver.boundary.set_raw_condition("coldWall", "T", {"type": "fixedValue", "value": "300"})
    solver.boundary.write_boundary_conditions()

    solver.run_simulation(nb_proc=1)


if __name__ == "__main__":
    main()