#!/usr/bin/env python3
"""Tutoriel 9 : Transfert de chaleur conjugué (chtMultiRegionFoam)

Démontre la configuration et l'exécution d'un cas CHT multi-régions
avec foampilot et OpenFOAM 13 (chtMultiRegionFoam).

Workflow :
  1. ChtSolver — configuration des régions, interfaces, et controlDict
  2. blockMesh — création du maillage de base (via Meshing + JSON)
  3. createZones — définition des zones cellulaires solide/fluide
  4. splitMeshRegions -cellZones — découpage en maillages par région
  5. foamSetupCHT — génération automatique des champs et propriétés
  6. foamDictionary — réglage des conditions initiales
  7. chtMultiRegionFoam — exécution de la simulation
  8. foamToVTK — conversion pour post-traitement
  9. pyvista — analyse et graphismes

Note : OpenFOAM 13 utilise `chtMultiRegionFoam` (binaire autonome)
et `foamSetupCHT` pour le setup automatisé. Le solveur
`chtMultiRegionSimpleFoam` n'est pas disponible dans OF-13.
"""

import subprocess
from pathlib import Path

from foampilot import Meshing
from foampilot.utilities import OpenFOAMDictAddFile
from foampilot.cht import (
    ChtSolver,
    FluidRegion,
    SolidRegion,
    CoupledInterface,
)


def write_create_zones_dict(case_path: Path):
    """Write createZonesDict for cell zone definition (solid region)
    using the foampilot OpenFOAMDictAddFile API."""
    zones_dict = OpenFOAMDictAddFile(
        object_name="createZonesDict",
        metal={
            "type": "box",
            "zoneType": "cell",
            "box": "(0.03 0.005 -1) (0.07 0.015 1)",
        },
        heater={
            "type": "box",
            "zoneType": "cell",
            "box": "(0.07 0.005 -1) (0.09 0.015 1)",
        },
    )
    system_path = case_path / "system"
    system_path.mkdir(parents=True, exist_ok=True)
    zones_dict.write("createZonesDict", case_path)


def main():
    case_path = Path.cwd()

    # --- 1. Configurer les régions CHT ------------------------------------
    # Fluide : air (compressible, heRhoThermo)
    fluid_region = FluidRegion(
        name="fluid",
        temperature=300.0,
        velocity=(1.0, 0.0, 0.0),
        turbulence_model="kOmegaSST",
        thermophysical_model="heRhoThermo",
        thermo_model="hConst",
        equation_of_state="rhoConst",
    )

    # Solide : cuivre (mur chauffé, conductivité élevée)
    metal_region = SolidRegion(
        name="metal",
        temperature=350.0,
        thermal_conductivity=380.0,
        density=8960.0,
        specific_heat=385.0,
    )
    heater_region = SolidRegion(
        name="heater",
        temperature=350.0,
        thermal_conductivity=0.5,
        density=1000.0,
        specific_heat=1000.0,
    )

    # Interfaces fluid-metal and metal-heater from the reference topology.
    fluid_metal = CoupledInterface(
        name="fluid_to_metal", fluid_region="fluid", solid_region="metal",
        heat_transfer_coefficient=10.0,
    )
    metal_fluid = CoupledInterface(
        name="metal_to_fluid", fluid_region="fluid", solid_region="metal",
        heat_transfer_coefficient=10.0,
    )
    metal_heater = CoupledInterface(
        name="metal_to_heater", fluid_region="heater", solid_region="metal",
        heat_transfer_coefficient=10.0,
    )
    heater_metal = CoupledInterface(
        name="heater_to_metal", fluid_region="heater", solid_region="metal",
        heat_transfer_coefficient=10.0,
    )

    # --- 2. Initialiser le solveur CHT ------------------------------------
    solver = ChtSolver(
        case_path=case_path,
        solver_name="chtMultiRegionFoam",
        regions=[fluid_region, metal_region, heater_region],
        interfaces=[fluid_metal, metal_fluid, metal_heater, heater_metal],
    )

    # --- 3. Configurer le controlDict ------------------------------------
    solver.system.controlDict.startTime = 0
    solver.system.controlDict.endTime = 20
    solver.system.controlDict.deltaT = 1e-3
    solver.system.controlDict.writeControl = "adjustableRunTime"
    solver.system.controlDict.writeInterval = 1
    solver.system.controlDict.application = "foamMultiRun"

    # --- 4. Setup de base : fichiers système et constant -----------------
    solver.setup_case()
    solver.write_case()
    solver.write_region_system_files()
    solver.set_region_gravity("fluid", "(0 0 0)")
    solver.set_region_momentum_transport("fluid", "laminar")

    # --- 5. Maillage (blockMesh via JSON config) -------------------------
    print("1. Generating mesh (blockMesh) ...")
    data_path = case_path / "block_mesh.json"
    mesh = Meshing(case_path, mesher="blockMesh")
    mesh.mesher.load_from_json(data_path)
    mesh.mesher.write(file_path=case_path / "system" / "blockMeshDict")
    solver.run_command(["blockMesh"], log_filename="log.blockMesh")

    # --- 6. Zones de cellules et découpage en régions -------------------
    print("2. Creating cell zones (createZones) ...")
    write_create_zones_dict(case_path)
    solver.run_command(["createZones", "-dict", "system/createZonesDict"],
                       log_filename="log.createZones")

    print("3. Splitting mesh into regions (splitMeshRegions) ...")
    solver.run_command(
        ["splitMeshRegions", "-cellZones", "-defaultRegionName", "fluid"],
        log_filename="log.splitMeshRegions",
    )

    # --- 7. Conditions limites régionales via l’API ChtSolver ------------
    fluid_default = {".*": {"type": "zeroGradient"}}
    solver.set_region_boundary_conditions("fluid", "T", {
        "fluidInlet": {"type": "fixedValue", "value": "$internalField"},
        "fluidOutlet": {"type": "inletOutlet", "value": "$internalField", "inletValue": "$internalField"},
        "frontAndBack": {"type": "empty"},
        "fluid_to_metal": {"type": "coupledTemperature", "value": "$internalField"},
        "fluid_to_heater": {"type": "coupledTemperature", "value": "$internalField"},
        ".*": {"type": "zeroGradient"},
    })
    solver.set_region_boundary_conditions("fluid", "U", {
        "fluidInlet": {"type": "fixedValue", "value": "uniform (0 0 1e-3)"},
        "fluidOutlet": {"type": "pressureInletOutletVelocity", "value": "$internalField"},
        "frontAndBack": {"type": "empty"},
        "fluid_to_metal": {"type": "noSlip"},
        "fluid_to_heater": {"type": "noSlip"},
        ".*": {"type": "noSlip"},
    })
    solver.set_region_boundary_conditions("fluid", "p", {".*": {"type": "calculated", "value": "$internalField"}})
    solver.set_region_boundary_conditions("fluid", "p_rgh", {
        ".*": {"type": "fixedFluxPressure", "value": "$internalField"},
        "fluidOutlet": {"type": "fixedValue", "value": "$internalField"},
    })
    for field in ("k", "omega", "nut"):
        solver.set_region_boundary_conditions("fluid", field, fluid_default)
    for region in ("metal", "heater"):
        solver.set_region_boundary_conditions(region, "T", {
            ".*": {"type": "zeroGradient"},
            ("heater_to_fluid" if region == "heater" else "metal_to_fluid"): {
                "type": "coupledTemperature", "value": "$internalField"
            },
        })

    # --- 8. Conditions initiales via l’API ChtSolver ----------------------
    print("4. Setting initial conditions ...")
    solver.set_region_internal_field("heater", "T", "uniform 350")
    solver.set_region_internal_field("metal", "T", "uniform 300")
    solver.set_region_internal_field("fluid", "T", "uniform 300")
    solver.set_region_internal_field("fluid", "p", "uniform 0")

    # --- 8. Lancer la simulation ------------------------------------------
    print("6. Running chtMultiRegionFoam ...")
    solver.run_simulation(nb_proc=1)

    # --- 10. Conversion VTK ----------------------------------------------
    print("7. Converting to VTK (foamToVTK) ...")
    solver.run_command(
        ["foamToVTK", "-region", "fluid", "-latestTime",
         "-fields", "(T U p k omega)"],
        log_filename="log.foamToVTK_fluid",
    )
    for solid_region in ("metal", "heater"):
        solver.run_command(
            ["foamToVTK", "-region", solid_region, "-latestTime",
             "-fields", "(T)"],
            log_filename=f"log.foamToVTK_{solid_region}",
        )

    # --- 11. Post-traitement ---------------------------------------------
    print("8. Post-processing ...")
    solver.run_command(
        ["python", "run_post.py"],
        log_filename="log.run_post",
    )

    # --- 12. Validation --------------------------------------------------
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    print("Expected results (from reference heatedDuct case):")
    print("  - Interface T: 350.00 K (continuous)")
    print("  - h: 3.38 W/(m²·K)")
    print("  - Nu: 0.2597")
    print("  - R_th: 0.2963 K/W")
    print("=" * 60)
    print(f"\nTutoriel terminé !")
    print(f"  Cas      : {case_path}")
    print(f"  Résultats: {case_path / 'postProcessing'}")
    print(f"  Graphismes: {case_path / 'postProcessing' / '*.png'}")


if __name__ == "__main__":
    main()
