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
        solid={
            "type": "box",
            "zoneType": "cell",
            "box": "(0 -1 -1) (0.1 0.002 1)",
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
        equation_of_state="perfectGas",
    )

    # Solide : cuivre (mur chauffé, conductivité élevée)
    solid_region = SolidRegion(
        name="solid",
        temperature=350.0,
        thermal_conductivity=380.0,  # W/(m·K)
        density=8960.0,              # kg/m³
        specific_heat=385.0,         # J/(kg·K)
    )

    # Interfaces fluide-solide
    interface = CoupledInterface(
        name="fluid_to_solid",
        fluid_region="fluid",
        solid_region="solid",
        heat_transfer_coefficient=10.0,
    )
    solid_interface = CoupledInterface(
        name="solid_to_fluid",
        fluid_region="fluid",
        solid_region="solid",
        heat_transfer_coefficient=10.0,
    )

    # --- 2. Initialiser le solveur CHT ------------------------------------
    solver = ChtSolver(
        case_path=case_path,
        solver_name="chtMultiRegionFoam",
        regions=[fluid_region, solid_region],
        interfaces=[interface, solid_interface],
    )

    # --- 3. Configurer le controlDict ------------------------------------
    solver.system.controlDict.start_time = 0
    solver.system.controlDict.end_time = 1.0
    solver.system.controlDict.delta_t = 5e-4
    solver.system.controlDict.write_interval = 0.1
    solver.system.controlDict.application = "chtMultiRegionFoam"

    # --- 4. Setup de base : fichiers système et constant -----------------
    solver.setup_case()
    solver.write_case()

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

    # --- 7. foamSetupCHT — setup automatisé ------------------------------
    print("4. Setting up CHT case (foamSetupCHT) ...")
    solver.run_command(["foamSetupCHT"], log_filename="log.foamSetupCHT")

    # --- 8. Conditions initiales -----------------------------------------
    print("5. Setting initial conditions ...")
    solver.run_command(
        ["foamDictionary", "-entry", "internalField",
         "-set", "uniform 350", "0/solid/T"],
        log_filename="log.dict_solid_T",
    )
    solver.run_command(
        ["foamDictionary", "-entry", "internalField",
         "-set", "uniform 300", "0/fluid/T"],
        log_filename="log.dict_fluid_T",
    )
    solver.run_command(
        ["foamDictionary", "-entry", "internalField",
         "-set", "uniform 1e5", "0/fluid/p"],
        log_filename="log.dict_fluid_p",
    )

    # --- 9. Lancer la simulation ------------------------------------------
    print("6. Running chtMultiRegionFoam ...")
    solver.run_simulation(nb_proc=1)

    # --- 10. Conversion VTK ----------------------------------------------
    print("7. Converting to VTK (foamToVTK) ...")
    solver.run_command(
        ["foamToVTK", "-region", "fluid", "-latestTime",
         "-fields", "(T U p k omega)"],
        log_filename="log.foamToVTK_fluid",
    )
    solver.run_command(
        ["foamToVTK", "-region", "solid", "-latestTime",
         "-fields", "(T)"],
        log_filename="log.foamToVTK_solid",
    )

    # --- 11. Post-traitement ---------------------------------------------
    print("8. Post-processing ...")
    run_post = subprocess.run(
        ["python", "run_post.py"],
        cwd=case_path, capture_output=True, text=True,
    )
    print(run_post.stdout)
    if run_post.returncode != 0:
        print(f"Post-processing warning: {run_post.stderr[-500:]}")

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
