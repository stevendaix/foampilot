#!/usr/bin/env python3
"""Tutorial CHT : Shell-and-Tube Heat Exchanger (chtMultiRegionFoam)

Reproduit le cas ``shellAndTubeHeatExchanger`` d'OpenFOAM-14 avec foampilot.

Ce tutoriel montre comment :
1. Configurer un échangeur thermique à tubes avec régions solides et fluides
2. Utiliser ``chtMultiRegionFoam`` (transitoire) pour le couplage fluide-solide
3. Post-traiter le flux de chaleur à l'interface

Usage ::

    cd examples/cht/shellAndTube
    python run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "foampilot" / "src"))

from foampilot.cht import (
    ChtSolver,
    FluidRegion,
    SolidRegion,
    CoupledInterface,
    ExternalTemperatureBC,
    TotalTemperatureBC,
    FixedTemperatureBC,
    calc_interface_heat_flux,
    calc_nusselt_number,
    calc_thermal_resistance,
)

# ---------------------------------------------------------------------------
# 1. Case path
# ---------------------------------------------------------------------------
case_path = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 2. Define regions
# ---------------------------------------------------------------------------
tube_fluid = FluidRegion(
    name="tube",
    temperature=300.0,
    velocity=(2.0, 0.0, 0.0),
    turbulence_model="kOmegaSST",
)

shell_fluid = FluidRegion(
    name="shell",
    temperature=350.0,
    velocity=(0.5, 0.0, 0.0),
    turbulence_model="kOmegaSST",
)

tube_solid = SolidRegion(
    name="tubeWall",
    temperature=320.0,
    thermal_conductivity=15.0,     # W/(m·K) — PVC / fibre verre
    density=1200.0,                # kg/m³
    specific_heat=2000.0,          # J/(kg·K)
)

regions = [tube_fluid, shell_fluid, tube_solid]

# ---------------------------------------------------------------------------
# 3. Define interfaces
# ---------------------------------------------------------------------------
interfaces = [
    CoupledInterface(
        name="tubeWall_inner",
        fluid_region="tube",
        solid_region="tubeWall",
        heat_transfer_coefficient=500.0,
    ),
    CoupledInterface(
        name="tubeWall_outer",
        fluid_region="shell",
        solid_region="tubeWall",
        heat_transfer_coefficient=200.0,
    ),
]

# ---------------------------------------------------------------------------
# 4. Create CHT solver (transient)
# ---------------------------------------------------------------------------
solver = ChtSolver(
    case_path=case_path,
    solver_name="chtMultiRegionFoam",
    regions=regions,
    interfaces=interfaces,
    region_solvers={
        "tube": "fluid",
        "shell": "fluid",
        "tubeWall": "solid",
    },
)

# ---------------------------------------------------------------------------
# 5. Set up and run
# ---------------------------------------------------------------------------
solver.setup_case()
solver.write_case()

print(f"CHT case created at: {case_path}")
print(f"Regions: {[r.name for r in regions]}")
print(f"Region solvers: {solver._region_solvers}")
print(f"Interfaces: {[i.name for i in interfaces]}")

# Uncomment to run the simulation (requires OpenFOAM installed)
# solver.run_simulation(nb_proc=4, log_filename="log.chtMultiRegionFoam")

print("Done — case files generated. Run with: chtMultiRegionFoam")
