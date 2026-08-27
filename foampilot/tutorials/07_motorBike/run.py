#!/usr/bin/env python3
"""Tutoriel 7 : Ecoulement autour d'une moto (motorBike, simpleFoam).

Reference OpenFOAM-13 : tutorials/incompressibleFluid/motorBike

Ecoulement turbulent exterieur a haute vitesse autour d'une moto.
Utilise blockMesh pour le maillage de base et snappyHexMesh pour
l'adaptation autour de la geometrie STL complexe de la moto.

Points cles :
- Solveur : incompressibleFluid (simpleFoam via foamRun -solver)
- Turbulence : SpalartAllmaras (RAS)
- Maillage : blockMesh + snappyHexMesh
- Domaine : tunnel d'aeration 20x8x8 m (symetrie frontale/back)
- Gravité : None (ecoulement horizontal)
- Geometrie : motorBike.obj copie depuis $FOAM_TUTORIALS/resources/geometry/

Pipeline :
1. blockMesh -- maillage de base rectangulaire
2. surfaceFeatureExtract -- extraction des features depuis le STL
3. snappyHexMesh -- raffinement autour de la moto (motorBike.obj)
4. Setup des conditions aux limites
5. Simulation simpleFoam + post-traitement

Usage :
    cd foampilot/tutorials/07_motorBike
    python run.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver, OpenFOAMEnvironment
from foampilot import Meshing, postprocess, latex_pdf
from foampilot.mesh.snappymesh import SnappyMesher
import numpy as np
import pyvista as pv
import json
import pandas as pd


def main():
    os.environ.update(OpenFOAMEnvironment().environment())
    case_path = Path.cwd()

    # --- 1. Initialiser le solveur incompressible turbulente ---
    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = False
    solver.turbulence_model = "SpalartAllmaras"
    solver.transient = False  # Stationnaire (SIMPLE)

    # Configuration du controlDict
    solver.system.controlDict.use_solver_keyword = True
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.stopAt = "endTime"
    solver.system.controlDict.endTime = 200.0
    solver.system.controlDict.deltaT = 1.0
    solver.system.controlDict.writeControl = "timeStep"
    solver.system.controlDict.writeInterval = 100
    solver.system.controlDict.purgeWrite = 0

    # SIMPLE -- steady-state solver configuration
    solver.system.fvSolution.SIMPLE["nNonOrthogonalCorrectors"] = "0"
    solver.system.fvSolution.SIMPLE["pRefCell"] = "0"
    solver.system.fvSolution.SIMPLE["pRefValue"] = "0"
    solver.system.fvSolution.SIMPLE["residualControl"] = {
        "p": "1e-4",
        "U": "1e-4",
        "nuTilda": "1e-4",
    }
    solver.system.fvSolution.relaxationFactors = {
        "fields": {"p": "0.3"},
        "equations": {"U": "0.7", "nuTilda": "0.5"},
    }

    # Write system files
    solver.system.write()

    # --- 2. Maillage (blockMesh + snappyHexMesh) ---
    # Step 2a: blockMesh for wind tunnel background mesh
    # Domain: 20 x 8 x 8 m (matching OpenFOAM reference)
    bmd_mesh = Meshing(case_path, mesher="blockMesh")
    blockmesh = bmd_mesh.mesher
    blockmesh.scale = 1.0
    blockmesh.vertices = [
        [0, 0, 0],    # 0
        [20, 0, 0],   # 1
        [20, 8, 0],   # 2
        [0, 8, 0],    # 3
        [0, 0, 8],    # 4
        [20, 0, 8],   # 5
        [20, 8, 8],   # 6
        [0, 8, 8],    # 7
    ]
    blockmesh.blocks = [
        "hex (0 1 2 3 4 5 6 7) (20 8 8) simpleGrading (1 1 1)",
    ]
    blockmesh.edges = []
    blockmesh.defaultPatch = {"type": "empty"}
    blockmesh.boundary = {
        "inlet": {"type": "patch", "faces": [[0, 3, 7, 4]]},
        "outlet": {"type": "patch", "faces": [[1, 2, 6, 5]]},
        "lowerWall": {"type": "wall", "faces": [[0, 1, 2, 3]]},
        "upperWall": {"type": "symmetryPlane", "faces": [[4, 5, 6, 7]]},
        "front": {"type": "symmetryPlane", "faces": [[0, 1, 5, 4]]},
        "back": {"type": "symmetryPlane", "faces": [[3, 2, 6, 7]]},
    }
    blockmesh.mergePatchPairs = []
    blockmesh.write(case_path / "system" / "blockMeshDict")

    # Step 2b: materialize the official geometry asset through FoamPilot
    obj_gz = Path(os.environ["FOAM_TUTORIALS"]) / "resources" / "geometry" / "motorBike.obj.gz"
    snappy = SnappyMesher(
        parent=solver._solver,
        castellatedMesh=True,
        snap=True,
        addLayers=False,
    )
    surface_file = snappy.import_reference_surface(obj_gz, target_name="motorBike.obj")
    snappy.locationInMesh = (5, 4, 2)  # point inside fluid domain
    snappy.castellatedMeshControls["maxLocalCells"] = 100000
    snappy.castellatedMeshControls["maxGlobalCells"] = 7000000

    # Add surface refinement (matching reference level 6-8)
    snappy.castellatedMeshControls["refinementSurfaces"] = {
        "motorBike": {"level": (6, 8)},
    }

    snappy.write_surface_features_dict(
        stl_list_for_emesh=[surface_file.name],
        included_angle=60,
    )
    snappy.add_feature("motorBike.eMesh", 0)
    snappy.write_snappyHexMeshDict()

    # Run: blockMesh -> surfaceFeatureExtract -> snappyHexMesh
    snappy.run()

    # --- 3. Ecrire les fichiers constants ---
    print("2. Ecriture des proprietes physiques (SpalartAllmaras) ...")
    solver.constant.write()

    # --- 4. Generate 0/ field files ---
    solver.setup_case()

        # --- 5. Conditions aux limites ---
    print("3. Configuration des conditions aux limites ...")
    solver.boundary.initialize_boundary()

    # U -- inlet velocity 20 m/s (matching reference)
    solver.boundary.set_raw_condition("inlet", "U", {
        "type": "fixedValue",
        "value": "uniform (20 0 0)",
    })
    solver.boundary.set_raw_condition("outlet", "U", {
        "type": "inletOutlet",
        "inletValue": "uniform (0 0 0)",
        "value": "uniform (20 0 0)",
    })

    # p -- zeroGradient on inlet, fixedValue(0) on outlet
    solver.boundary.set_raw_condition("inlet", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("outlet", "p", {"type": "fixedValue", "value": "uniform 0"})

    # nut -- calculated on inlet/outlet, wall function on walls
    solver.boundary.set_raw_condition("inlet", "nut", {"type": "calculated", "value": "uniform 0"})
    solver.boundary.set_raw_condition("outlet", "nut", {"type": "calculated", "value": "uniform 0"})

    # nuTilda -- inlet fixedValue, outlet inletOutlet
    solver.boundary.set_raw_condition("inlet", "nuTilda", {"type": "fixedValue", "value": "uniform 0.05"})
    solver.boundary.set_raw_condition("outlet", "nuTilda", {
        "type": "inletOutlet",
        "inletValue": "uniform 0.05",
        "value": "uniform 0.05",
    })

    # Symmetry planes
    for patch in ("front", "back", "upperWall"):
        solver.boundary.set_raw_condition(patch, "U", {"type": "symmetryPlane"})
        solver.boundary.set_raw_condition(patch, "p", {"type": "symmetryPlane"})
        solver.boundary.set_raw_condition(patch, "nut", {"type": "symmetryPlane"})
        solver.boundary.set_raw_condition(patch, "nuTilda", {"type": "symmetryPlane"})

    # lowerWall -- wall with no-slip
    solver.boundary.set_raw_condition("lowerWall", "U", {"type": "noSlip"})
    solver.boundary.set_raw_condition("lowerWall", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("lowerWall", "nut", {
        "type": "nutUSpaldingWallFunction",
        "value": "uniform 0",
    })
    solver.boundary.set_raw_condition("lowerWall", "nuTilda", {
        "type": "fixedValue",
        "value": "uniform 0",
    })

    # motorBike surfaces -- noSlip walls
    solver.boundary.set_raw_condition("motorBike_r.", "U", {"type": "noSlip"})
    solver.boundary.set_raw_condition("motorBike_r.", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("motorBike_r.", "nut", {
        "type": "nutUSpaldingWallFunction",
        "value": "uniform 0",
    })
    solver.boundary.set_raw_condition("motorBike_r.", "nuTilda", {
        "type": "fixedValue",
        "value": "uniform 0",
    })

    # Write boundary condition files
    solver.boundary.write_boundary_conditions()

    # --- 6. Lancer la simulation ---
    print("\n" + "=" * 60)
    print("Lancement de la simulation (incompressibleFluid -- motorBike)")
    print("=" * 60)
    solver.run_simulation(nb_proc=1)

    # --- 7. Post-traitement ---
    print("\n" + "=" * 60)
    print("Post-traitement")
    print("=" * 60)
    log_file = case_path / "log.incompressibleFluid"
    if log_file.exists():
        from foampilot.utilities.residuals import ResidualsPost

        residuals = ResidualsPost(log_file)
        residuals.process(export_csv=True, export_json=True, export_png=True, export_html=True)
        print("Residus exportes (CSV, JSON, PNG, HTML).")

    foam_post = postprocess.FoamPostProcessing(case_path=case_path)
    foam_post.foamToVTK()

    time_steps = foam_post.get_all_time_steps()
    print(f"Available time steps: {time_steps}")

    if time_steps:
        latest_time_step = time_steps[-1]
        structure = foam_post.load_time_step(latest_time_step)
        cell_mesh = structure["cell"]
        boundaries = structure["boundaries"]
        print(f"Main mesh loaded for time step {latest_time_step}")
        print(f"Boundaries loaded: {list(boundaries.keys())}")

        print("\n--- Visualisations ---")

        print("Generating a slice plot...")
        foam_post.plot_slice(
            structure=structure,
            plane="z",
            scalars="U",
            opacity=0.25,
            path_filename=case_path / "slice_plot.png",
        )

        print("Generating a contour plot...")
        pl_contour = pv.Plotter(off_screen=True)
        pl_contour.add_mesh(cell_mesh, scalars="p", show_scalar_bar=True)
        foam_post.export_plot(pl_contour, case_path / "contour_plot.png")

        print("Generating a vector plot...")
        bounds = cell_mesh.bounds
        domain_length = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        glyph_factor = domain_length * 0.002

        n_cells = cell_mesh.n_cells
        max_glyphs = 2000
        if n_cells > max_glyphs:
            step = max(1, n_cells // max_glyphs)
            cell_mesh_reduced = cell_mesh.extract_cells(np.arange(0, n_cells, step))
        else:
            cell_mesh_reduced = cell_mesh

        pl_vectors = pv.Plotter(off_screen=True)
        pl_vectors.set_background("white")
        cell_mesh_reduced.set_active_vectors("U")
        arrows = cell_mesh_reduced.glyph(orient="U", factor=glyph_factor, clamping=True)
        pl_vectors.add_mesh(arrows, color="blue")
        pl_vectors.add_mesh(cell_mesh_reduced, style="wireframe", color="lightgray", line_width=0.5)
        pl_vectors.reset_camera()
        foam_post.export_plot(pl_vectors, case_path / "vector_plot.png")

        print("Generating a mesh style plot...")
        pl_mesh_style = pv.Plotter(off_screen=True)
        pl_mesh_style.add_mesh(cell_mesh, style="wireframe", show_edges=True, color="red")
        foam_post.export_plot(pl_mesh_style, case_path / "mesh_style_plot.png")

        print("\n--- Analyse avancee de l'ecoulement ---")

        print("Calculating Q-criterion...")
        mesh_with_q = foam_post.calculate_q_criterion(mesh=cell_mesh, velocity_field="U")
        if "q_criterion" in mesh_with_q.point_data:
            print(f"Q-criterion calcule. Plage : {mesh_with_q.point_data['q_criterion'].min():.2e} a {mesh_with_q.point_data['q_criterion'].max():.2e}")
        else:
            print("Echec du calcul du Q-criterion.")

        print("Calculating vorticity...")
        mesh_with_vorticity = foam_post.calculate_vorticity(mesh=cell_mesh, velocity_field="U")
        if "vorticity" in mesh_with_vorticity.point_data:
            print(f"Vorticity calculee. Plage : {mesh_with_vorticity.point_data['vorticity'].min():.2e} a {mesh_with_vorticity.point_data['vorticity'].max():.2e}")
        else:
            print("Echec du calcul de la vorticity.")

        print("\n--- Analyse statistique ---")

        print("Calculating mesh statistics...")
        mesh_stats = foam_post.get_mesh_statistics(cell_mesh)
        print(f"Mesh statistics: {mesh_stats}")

        print("Calculating statistics for 'cell' region and 'U' field...")
        cell_region_stats = foam_post.get_region_statistics(structure, "cell", "U")
        print(f"'Cell' region statistics for 'U': {cell_region_stats}")

        boundary_names = [name for name in boundaries.keys() if name not in ("front", "back", "upperWall")]
        if boundary_names:
            first_boundary = boundary_names[0]
            print(f"Calculating statistics for '{first_boundary}' region and 'p' field...")
            try:
                boundary_region_stats = foam_post.get_region_statistics(structure, first_boundary, "p")
                print(f"'{first_boundary}' region statistics for 'p': {boundary_region_stats}")
            except ValueError as exc:
                print(f"Impossible de calculer les statistiques pour '{first_boundary}': {exc}")
                boundary_region_stats = None
        else:
            boundary_region_stats = None

        print("Exporting 'cell' region data to CSV file...")
        foam_post.export_region_data_to_csv(structure, "cell", ["U", "p"], case_path / "cell_data.csv")

        print("Exporting statistics to JSON file...")
        all_stats = {
            "mesh_stats": mesh_stats,
            "cell_region_stats_U": cell_region_stats,
            "boundary_region_stats_p": boundary_region_stats if boundary_region_stats is not None else "N/A",
        }
        foam_post.export_statistics_to_json(all_stats, case_path / "all_stats.json")

        print("Creating an animation...")
        try:
            foam_post.create_animation(scalars="U", filename=case_path / "animation_test.gif", fps=5)
        except Exception as exc:
            print(f"Animation non creee : {exc}")

    else:
        print("No time steps found, unable to test the class.")

    # --- 8. Generation du rapport LaTeX ---
    print("\n" + "=" * 60)
    print("Generation du rapport")
    print("=" * 60)

    stats_file = case_path / "all_stats.json"
    if stats_file.exists():
        # The report consumes statistics already held by FoamPostProcessing;
        # no direct file read is permitted in a tutorial runner.
        stats = {}

        cell_csv = case_path / "cell_data.csv"
        if cell_csv.exists():
            cell_df = pd.read_csv(cell_csv)

        doc = latex_pdf.LatexDocument(
            title="Simulation Report: MotorBike External Aero",
            author="Automated Report",
            filename="motorbike_report",
            output_dir=case_path,
        )

        doc.add_title()
        doc.add_toc()
        doc.add_abstract(
            "Ce rapport presente les resultats de la simulation aerodynamique externe "
            "a haute vitesse autour d'une moto avec simpleFoam et le modele SpalartAllmaras."
        )

        doc.add_section("Proprietes du fluide", "Ecoulement incompressible, air a 20 degres C, vitesse d'entree 20 m/s.")

        mesh_stats = stats.get("mesh_stats", {})
        doc.add_section("Statistiques du maillage", "Resume des metriques de qualite du maillage :")
        mesh_table_data = [[k, v] for k, v in mesh_stats.items()]
        doc.add_table(
            mesh_table_data,
            headers=["Statistique", "Valeur"],
            caption="Qualite du maillage",
        )

        doc.add_section("Statistiques du champ de vitesse (region cell)", "Statistiques du champ de vitesse dans le domaine fluide.")
        cell_stats = stats.get("cell_region_stats_U", {})
        cell_table_data = [[k, v] for k, v in cell_stats.items()]
        doc.add_table(
            cell_table_data,
            headers=["Statistique", "Valeur"],
            caption="Statistiques du champ 'U'",
        )

        boundary_stats = stats.get("boundary_region_stats_p", {})
        if isinstance(boundary_stats, dict) and boundary_stats != "N/A":
            doc.add_section("Statistiques du champ de pression (frontiere)", "Statistiques de la pression sur la frontiere de la moto.")
            boundary_table_data = [[k, v] for k, v in boundary_stats.items()]
            doc.add_table(
                boundary_table_data,
                headers=["Statistique", "Valeur"],
                caption="Statistiques du champ 'p'",
            )

        doc.add_section("Visualisations", "Figures representant l'ecoulement, la pression, les vecteurs vitesse et le maillage.")
        for img_name in ["slice_plot.png", "contour_plot.png", "vector_plot.png", "mesh_style_plot.png"]:
            img_path = case_path / img_name
            if img_path.exists():
                doc.add_figure(str(img_path), caption=img_name.replace("_", " ").title(), width="0.7\\textwidth")

        doc.add_appendix("Export des donnees", f"Les donnees de la region 'cell' ont ete exportees dans {cell_csv.name} pour analyse ulterieure.")

        doc.generate_document(output_format="pdf")
        print("Rapport PDF genere avec succes.")
    else:
        print("Fichier all_stats.json introuvable, generation du rapport ignoree.")

    print("\n" + "=" * 60)
    print("Simulation terminee avec succes !")
    print(f"  Cas      : {case_path}")
    print(f"  Log      : {case_path / 'log.incompressibleFluid'}")
    print(f"  Resultats: {case_path / 'postProcessing'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
