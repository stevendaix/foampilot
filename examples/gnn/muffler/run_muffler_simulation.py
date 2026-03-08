#!/usr/bin/env python
"""
run_muffler_simulation.py - Simulation pour le silencieux basée sur run_simu.py
Usage: python run_muffler_simulation.py --case-dir ./cas_test --params params.json
"""

import argparse
import json
from pathlib import Path
import numpy as np
from foampilot.solver import Solver
from foampilot import Meshing, utilities, postprocess, FluidMechanics, ValueWithUnit
import classy_blocks as cb
import sys


def run_simulation(case_dir: Path, params: dict) -> dict:
    """
    Lance une simulation avec les paramètresdonnés.
    
    Args:
        case_dir: Répertoire où créer le cas
        params: Dictionnaire des paramètres
    
    Returns:
        dict: Résultat de la simulation avec métadonnées pour le GNN
    """
    print(f"\n{'='*60}")
    print(f"🚀 Simulation dans: {case_dir}")
    print(f"📊 Paramètres: {json.dumps(params, indent=2)}")
    print('='*60)
    
    # Créer le répertoire
    case_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Configuration fluide ---
    # Get available fluids and use Water
    available_fluids = utilities.FluidMechanics.get_available_fluids()
    fluid_mech = FluidMechanics(
        available_fluids['Water'],
        temperature=ValueWithUnit(params.get('temperature', 293.15), "K"),
        pressure=ValueWithUnit(params.get('pressure', 101325), "Pa")
    )
    properties = fluid_mech.get_fluid_properties()
    kinematic_viscosity = properties['kinematic_viscosity']
    print(f"\nUsing fluid: Water")
    print(f"Kinematic viscosity: {kinematic_viscosity}")
    
    # --- 2. Initialisation solver ---
    solver = Solver(case_dir)
    solver.compressible = False
    solver.with_gravity = False
    
    # Set the kinematic viscosity in the solver's constant directory
    solver.constant.transportProperties.nu = kinematic_viscosity
    
    # Generate system and constant directories with updated OpenFOAM configuration
    solver.system.write()
    solver.constant.write()
    
    # Convert numerical schemes settings to Python dictionary for inspection
    solver.system.fvSchemes.to_dict()
    
    # --- 3. Génération maillage paramétrique ---
    # Paramètres géométriques
    pipe_radius = params.get('pipe_radius', 0.05)
    muffler_radius = params.get('muffler_radius', 0.08)
    ref_length = params.get('ref_length', 0.1)
    cell_size = params.get('cell_size', 0.015)
    
    shapes = []
    
    # 0: Inlet pipe
    shapes.append(cb.Cylinder([0, 0, 0], [3 * ref_length, 0, 0], [0, pipe_radius, 0]))
    shapes[-1].chop_axial(start_size=cell_size)
    shapes[-1].chop_radial(start_size=cell_size)
    shapes[-1].chop_tangential(start_size=cell_size)
    shapes[-1].set_start_patch("inlet")
    
    # 1: Chain cylinder
    shapes.append(cb.Cylinder.chain(shapes[-1], ref_length))
    shapes[-1].chop_axial(start_size=cell_size)
    
    # 2: Extruded ring (start muffler)
    shapes.append(cb.ExtrudedRing.expand(shapes[-1], muffler_radius - pipe_radius))
    shapes[-1].chop_radial(start_size=cell_size)
    
    # 3: Extruded ring (muffler body)
    shapes.append(cb.ExtrudedRing.chain(shapes[-1], ref_length))
    shapes[-1].chop_axial(start_size=cell_size)
    
    # 4: Extruded ring (end muffler)
    shapes.append(cb.ExtrudedRing.chain(shapes[-1], ref_length))
    shapes[-1].chop_axial(start_size=cell_size)
    
    # 5: Fill (return to cylinder)
    shapes.append(cb.Cylinder.fill(shapes[-1]))
    shapes[-1].chop_radial(start_size=cell_size)
    
    # 6: Elbow
    elbow_center = shapes[-1].sketch_2.center + np.array([0, 2 * muffler_radius, 0])
    shapes.append(
        cb.Elbow.chain(shapes[-1], np.pi / 2, elbow_center, [0, 0, 1], pipe_radius)
    )
    shapes[-1].chop_axial(start_size=cell_size)
    shapes[-1].set_end_patch("outlet")
    
    # Assemblage maillage
    mesh = cb.Mesh()
    for shape in shapes:
        mesh.add(shape)
    mesh.set_default_patch("walls", "wall")
    
    # Write output files
    mesh.write(case_dir / "system" / "blockMeshDict", case_dir / "debug.vtk")
    
    print("Successfully generated blockMeshDict and debug.vtk files in the case directory.")
    
    # --- 4. Exécution blockMesh ---
    # Run blockMesh directly instead of using Meshing class to avoid issues
    import subprocess
    result = subprocess.run(
        ["blockMesh"],
        cwd=str(case_dir),
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"blockMesh output: {result.stdout}")
        print(f"blockMesh error: {result.stderr}")
        return {"success": False, "error": f"blockMesh failed: {result.stderr[:500]}"}
    
    # --- 5. Conditions limites ---
    solver.boundary.initialize_boundary()
    
    # Inlet avec vitesse variable
    inlet_velocity = params.get('inlet_velocity', 10.0)
    solver.boundary.apply_condition_with_wildcard(
        pattern="inlet",
        condition_type="velocityInlet",
        velocity=(ValueWithUnit(inlet_velocity, "m/s"), 
                  ValueWithUnit(0, "m/s"), 
                  ValueWithUnit(0, "m/s")),
        turbulence_intensity=params.get('turbulence_intensity', 0.05)
    )
    
    # Outlet avec pression variable
    outlet_pressure = params.get('outlet_pressure', 101325)
    solver.boundary.apply_condition_with_wildcard(
        pattern="outlet",
        condition_type="pressureOutlet",
        pressure=ValueWithUnit(outlet_pressure, "Pa")
    )
    
    solver.boundary.apply_condition_with_wildcard(
        pattern="walls",
        condition_type="wall"
    )
    
    solver.boundary.write_boundary_conditions()
    
    print("Boundary condition files have been generated")
    
    # Sauvegarder les BC pour le GNN
    boundary_info = {
        "inlet": {"U_inlet": [inlet_velocity, 0, 0]},
        "outlet": {"p_outlet": outlet_pressure},
        "walls": {"type": "wall"}
    }
    
    # --- 6. Exécution simulation ---
    try:
        # Run simulation
        solver.run_simulation()
    except Exception as e:
        return {"success": False, "error": f"Solver error: {str(e)}"}
    
    # --- 7. Post-traitement et extraction des données pour le GNN ---
    foam_post = postprocess.FoamPostProcessing(case_path=case_dir)
    
    # Convertir les résultats en VTK (requis pour FoamPostProcessing)
    try:
        print("  🔄 Conversion des résultats en VTK...")
        foam_post.foamToVTK()
    except Exception as e:
        print(f"  ⚠️ foamToVTK a échoué: {e}")
        # Essayer avec latestTime uniquement
        try:
            foam_post.foamToVTK(latest_time=True)
        except Exception as e2:
            return {"success": False, "error": f"foamToVTK failed: {e2}"}
    
    # Récupérer les résiduels (convergence)
    try:
        residuals_post = utilities.ResidualsPost(case_dir / "log.incompressibleFluid")
        residuals_data = residuals_post.process(export_json=True, return_data=True)
    except:
        residuals_data = {"final_residual": 1e-3}
    
    # Charger les résultats
    time_steps = foam_post.get_all_time_steps()
    
    if not time_steps:
        return {"success": False, "error": "No time steps found"}
    
    latest_time = time_steps[-1]
    structure = foam_post.load_time_step(latest_time)
    
    # Extraire les statistiques pour le GNN
    cell_stats = foam_post.get_region_statistics(structure, "cell", "U")
    pressure_stats = foam_post.get_region_statistics(structure, "cell", "p")
    
    # CRITIQUE : Sauvegarder les métadonnées pour le GNN
    metadata = {
        "params": params,
        "convergence": residuals_data,
        "stats": {
            "U": cell_stats,
            "p": pressure_stats
        },
        "boundary_conditions": boundary_info,
        "mesh_info": {
            "n_cells": len(structure["cell"].points),
            "cell_size": cell_size,
            "pipe_radius": pipe_radius,
            "muffler_radius": muffler_radius
        }
    }
    
    with open(case_dir / "gnn_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Exporter les données brutes pour vérification
    foam_post.export_region_data_to_csv(structure, "cell", ["U", "p"], 
                                        case_dir / "cell_data.csv")
    
    return {
        "success": True,
        "time_steps": time_steps,
        "n_cells": len(structure["cell"].points),
        "final_residual": float(residuals_data.get('final_residual', 1e-3)),
        "U_mean": float(cell_stats.get('mean', 0)),
        "p_mean": float(pressure_stats.get('mean', 0))
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lance une simulation CFD pour le silencieux")
    parser.add_argument("--case-dir", type=Path, required=True, help="Répertoire du cas")
    parser.add_argument("--params", type=str, required=True, help="Paramètres en JSON")
    args = parser.parse_args()
    
    try:
        params = json.loads(args.params)
        result = run_simulation(args.case_dir, params)
        
        # Sauvegarder le résultat
        with open(args.case_dir / "result.json", 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        # Code de retour pour succès/échec
        sys.exit(0 if result.get("success") else 1)
        
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        with open(args.case_dir / "result.json", 'w') as f:
            json.dump({"success": False, "error": str(e)}, f)
        sys.exit(1)
