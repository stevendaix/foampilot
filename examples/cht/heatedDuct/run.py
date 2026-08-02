#!/usr/bin/env python3
"""Tutorial CHT : Heated Duct (chtMultiRegionFoam / chtMultiRegionSimpleFoam)

Reproduit le cas ``heatedDuct`` d'OpenFOAM-14 avec foampilot.

NOTE : Ce cas utilise la syntaxe OpenFOAM-14 (units [mm] dans blockMeshDict,
snappyHexMesh). Pour OpenFOAM 13, utilisez le cas ``simple_heated_duct``
qui est pleinement fonctionnel.

Ce tutoriel montre comment :
1. Définir des régions fluides et solides
2. Configurer des interfaces CHT
3. Générer les fichiers de champ par région
4. Lancer ``chtMultiRegionFoam``

Usage ::

    cd examples/cht/heatedDuct
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
    CoupledTemperatureBC,
    ExternalTemperatureBC,
    FixedTemperatureBC,
    InletOutletTemperatureBC,
    SymmetryBC,
    TotalTemperatureBC,
    RadiationCoupledTemperatureBC,
    calc_nusselt_number,
    calc_heat_transfer_coefficient,
)

# ---------------------------------------------------------------------------
# 1. Case path
# ---------------------------------------------------------------------------
case_path = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 2. Define regions
# ---------------------------------------------------------------------------
fluid = FluidRegion(
    name="fluid",
    temperature=300.0,
    velocity=(1.0, 0.0, 0.0),
    turbulence_model="kOmegaSST",
)

heater = SolidRegion(
    name="heater",
    temperature=350.0,
    thermal_conductivity=50.0,    # W/(m·K)
    density=7800.0,               # kg/m³
    specific_heat=460.0,          # J/(kg·K)
)

metal = SolidRegion(
    name="metal",
    temperature=300.0,
    thermal_conductivity=45.0,    # W/(m·K) — aluminium
    density=2700.0,               # kg/m³
    specific_heat=900.0,          # J/(kg·K)
)

regions = [fluid, heater, metal]

# ---------------------------------------------------------------------------
# 3. Define interfaces
# ---------------------------------------------------------------------------
interfaces = [
    CoupledInterface(
        name="fluid_heater",
        fluid_region="fluid",
        solid_region="heater",
        heat_transfer_coefficient=50.0,
    ),
    CoupledInterface(
        name="fluid_metal",
        fluid_region="fluid",
        solid_region="metal",
        heat_transfer_coefficient=50.0,
    ),
]

# ---------------------------------------------------------------------------
# 4. Create CHT solver
# ---------------------------------------------------------------------------
solver = ChtSolver(
    case_path=case_path,
    solver_name="chtMultiRegionFoam",
    regions=regions,
    interfaces=interfaces,
    region_solvers={
        "fluid": "fluid",
        "heater": "solid",
        "metal": "solid",
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
# solver.run_simulation(nb_proc=1)

print("Done — case files generated.")
