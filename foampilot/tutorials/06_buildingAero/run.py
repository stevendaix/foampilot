#!/usr/bin/env python3
"""Tutoriel 6 : Aérodynamique des bâtiments — écoulement turbulent (simpleFoam).

Référence OpenFOAM-13 : tutorials/incompressibleFluid/windAroundBuildings
https://develop.openfoam.com/Development/openfoam/-/tree/master/tutorials/incompressibleFluid/windAroundBuildings

Cet exemple simule un écoulement turbulent extérieur autour d'un bâtiment
en utilisant blockMesh pour le maillage de base et snappyHexMesh pour
l'adaptation autour de la géométrie STL des bâtiments.

Le pipeline de maillage :
1. blockMesh — maillage de base rectangulaire (tunnel d'aération)
2. surfaceFeatureExtract — extraction des arêtes features depuis le STL
3. snappyHexMesh — raffinement du maillage autour du bâtiment STL
"""

import sys
from pathlib import Path

import numpy as np
import pyvista as pv

# Add src to path for tutorial execution from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing
from foampilot.mesh.snappymesh import SnappyMesher
from foampilot.utilities.function import Functions
from foampilot.postprocess import FoamPostProcessing


def _box_triangles(cx: float, cy: float, cz: float,
                   lx: float, ly: float, lz: float) -> list:
    """Return the 12 triangles (two per face) for an axis-aligned box."""
    x0, x1 = cx - lx / 2, cx + lx / 2
    y0, y1 = cy - ly / 2, cy + ly / 2
    z0, z1 = cz - lz / 2, cz + lz / 2

    v = {
        "000": (x0, y0, z0), "100": (x1, y0, z0),
        "110": (x1, y1, z0), "010": (x0, y1, z0),
        "001": (x0, y0, z1), "101": (x1, y0, z1),
        "111": (x1, y1, z1), "011": (x0, y1, z1),
    }
    # 6 faces, each split into 2 triangles (outward-facing)
    faces = [
        ("000", "100", "110", "010"),  # -z (bottom)
        ("001", "000", "010", "011"),  # -x (left)
        ("101", "111", "110", "100"),  # +x (right)
        ("000", "001", "101", "100"),  # -y (front)
        ("010", "110", "111", "011"),  # +y (back)
        ("011", "111", "101", "001"),  # +z (top)
    ]
    tris = []
    for a, b, c, d in faces:
        tris.append((v[a], v[b], v[c]))
        tris.append((v[a], v[c], v[d]))
    return tris


def generate_buildings_stl(stl_file: Path) -> None:
    """Generate an STL with multiple buildings clustered at the centre of the
    wind-tunnel domain (x=100, y=50).

    Domain: 200 x 100 x 50 m  (x streamwise, y transverse, z vertical).
    Several buildings of varying height create a realistic urban array that
    produces interesting wake interactions and street-canyon flows.
    """

    buildings = [
        # (x_center, y_center, z_center, x_len, y_len, z_len)
        (100,  50, 15, 25, 25, 30),    # main tall building
        ( 85,  35, 10, 20, 15, 20),    # south-west building
        ( 85,  65, 10, 20, 15, 20),    # north-west building
        (115,  40,  8, 15, 10, 16),    # south-east building
        (115,  60,  8, 15, 10, 16),    # north-east building
    ]

    lines = ["solid buildings"]
    for cx, cy, cz, lx, ly, lz in buildings:
        for tri in _box_triangles(cx, cy, cz, lx, ly, lz):
            (v1, v2, v3) = tri
            nx = (v2[1] - v1[1]) * (v3[2] - v1[2]) - (v2[2] - v1[2]) * (v3[1] - v1[1])
            ny = (v2[2] - v1[2]) * (v3[0] - v1[0]) - (v2[0] - v1[0]) * (v3[2] - v1[0])
            nz = (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v2[1] - v1[1]) * (v3[0] - v1[0])
            norm = (nx * nx + ny * ny + nz * nz) ** 0.5
            if norm > 0:
                nx, ny, nz = nx / norm, ny / norm, nz / norm
            lines.append(f"facet normal {nx:.6f} {ny:.6f} {nz:.6f}")
            lines.append("outer loop")
            for vx, vy, vz in (v1, v2, v3):
                lines.append(f"vertex {vx:.6f} {vy:.6f} {vz:.6f}")
            lines.append("endloop")
            lines.append("endfacet")
    lines.append("endsolid buildings")

    stl_file.write_text("\n".join(lines) + "\n")


def main():
    case_path = Path.cwd()

    # --- 1. Initialiser le solveur incompressible turbulente ---
    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = False
    solver.turbulence_model = "kEpsilon"
    solver.transient = False  # Steady-state (simpleFoam)

    # --- 2. Configuration du controlDict ---
    # OpenFOAM 13 uses 'solver' keyword for foamRun -solver
    solver.system.controlDict.use_solver_keyword = True
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.stopAt = "endTime"
    solver.system.controlDict.endTime = 400.0
    solver.system.controlDict.deltaT = 1.0
    solver.system.controlDict.writeControl = "timeStep"
    solver.system.controlDict.writeInterval = 50
    solver.system.controlDict.purgeWrite = 0

    # SIMPLE — steady-state solver configuration
    solver.system.fvSolution.SIMPLE["nNonOrthogonalCorrectors"] = "0"
    solver.system.fvSolution.SIMPLE["pRefCell"] = "0"
    solver.system.fvSolution.SIMPLE["pRefValue"] = "0"
    solver.system.fvSolution.SIMPLE["residualControl"] = {
        "p": "1e-4",
        "U": "1e-4",
        "(k|epsilon)": "1e-4",
    }

    # Relaxation factors — p=0.3, U/k/epsilon=0.7 (match OpenFOAM reference)
    solver.system.fvSolution.relaxationFactors = {
        "fields": {"p": "0.3"},
        "equations": {"U": "0.7", "(k|epsilon).*": "0.7"},
    }

    # Write system files
    solver.system.write()

    # --- 3. Maillage (blockMesh + snappyHexMesh) ---
    # Step 3a: blockMesh for the wind tunnel background mesh
    # Wind tunnel domain: 200 x 100 x 50 m (x = streamwise, y = transverse, z = vertical)
    bmd_mesh = Meshing(case_path, mesher="blockMesh")
    blockmesh = bmd_mesh.mesher
    blockmesh.scale = 1.0
    blockmesh.vertices = [
        [0, 0, 0],    # 0
        [200, 0, 0],  # 1
        [200, 100, 0],# 2
        [0, 100, 0],  # 3
        [0, 0, 50],   # 4
        [200, 0, 50], # 5
        [200, 100, 50],# 6
        [0, 100, 50], # 7
    ]
    blockmesh.blocks = [
        "hex (0 1 2 3 4 5 6 7) (40 20 10) simpleGrading (1 1 1)",
    ]
    blockmesh.edges = []
    blockmesh.defaultPatch = {"type": "empty"}
    blockmesh.boundary = {
        "inlet": {"type": "patch", "faces": [[0, 3, 7, 4]]},
        "outlet": {"type": "patch", "faces": [[1, 2, 6, 5]]},
        "ground": {"type": "wall", "faces": [[0, 1, 2, 3]]},
        "ceiling": {"type": "wall", "faces": [[4, 5, 6, 7]]},
        "leftWall": {"type": "wall", "faces": [[0, 1, 5, 4]]},
        "rightWall": {"type": "wall", "faces": [[3, 2, 6, 7]]},
    }
    blockmesh.mergePatchPairs = []
    blockmesh.write(case_path / "system" / "blockMeshDict")

    # Step 3b: blockMesh, surfaceFeatures, snappyHexMesh
    import os
    reference_surface = Path(os.environ["FOAM_TUTORIALS"]) / "incompressibleFluid" / "windAroundBuildings" / "constant" / "geometry" / "buildings.obj.gz"
    snappy = SnappyMesher(
        parent=solver._solver,
        castellatedMesh=True,
        snap=True,
        addLayers=False,
    )
    surface_file = snappy.import_reference_surface(reference_surface)
    snappy.locationInMesh = (10, 50, 5)  # point inside fluid domain
    snappy.castellatedMeshControls["maxLocalCells"] = 200000
    snappy.castellatedMeshControls["maxGlobalCells"] = 4000000

    # Write surfaceFeaturesDict and snappyHexMeshDict
    snappy.write_surface_features_dict(
        stl_list_for_emesh=[surface_file.name],
        included_angle=60,
    )
    snappy.write_snappyHexMeshDict()

    # Run: blockMesh → surfaceFeatureExtract → snappyHexMesh
    snappy.run()

    # --- 4. Écrire les fichiers constants (turbulent k-epsilon) ---
    print("2. Ecriture des proprietes physiques (turbulent k-epsilon) ...")
    solver.constant.write()

    # --- 5. Generate 0/ field files (initial conditions) ---
    solver.setup_case()

    # --- 6. Conditions aux limites ---
    print("3. Configuration des conditions aux limites ...")
    solver.boundary.initialize_boundary()

    # U — vitesse d'entrée 10 m/s (wind)
    solver.boundary.set_raw_condition("inlet", "U", {
        "type": "fixedValue",
        "value": "uniform (10 0 0)",
    })
    solver.boundary.set_raw_condition("outlet", "U", {
        "type": "pressureInletOutletVelocity",
        "value": "uniform (0 0 0)",
    })

    # Set wall conditions for all wall-type patches
    # (buildings, ground, leftWall, rightWall, ceiling)
    for patch_name in solver.boundary.fields["U"]:
        if patch_name in ("inlet", "outlet"):
            continue
        solver.boundary.set_raw_condition(patch_name, "U", {
            "type": "noSlip",
        })

    # k — turbulence intensity I=0.1 → k = 1.5*(I*U)^2 = 1.5*(1)^2 = 1.5
    k_inlet = 1.5
    solver.boundary.set_raw_condition("inlet", "k", {
        "type": "fixedValue",
        "value": f"uniform {k_inlet}",
    })
    solver.boundary.set_raw_condition("outlet", "k", {
        "type": "inletOutlet",
        "inletValue": f"uniform {k_inlet}",
        "value": f"uniform {k_inlet}",
    })
    for patch_name in solver.boundary.fields["k"]:
        if patch_name in ("inlet", "outlet"):
            continue
        solver.boundary.set_raw_condition(patch_name, "k", {
            "type": "kqRWallFunction",
            "value": f"uniform {k_inlet}",
        })

    # epsilon — C_mu^0.75 * k^1.5 / L, L=10 → 0.03
    eps_inlet = 0.03
    solver.boundary.set_raw_condition("inlet", "epsilon", {
        "type": "fixedValue",
        "value": f"uniform {eps_inlet}",
    })
    solver.boundary.set_raw_condition("outlet", "epsilon", {
        "type": "inletOutlet",
        "inletValue": f"uniform {eps_inlet}",
        "value": f"uniform {eps_inlet}",
    })
    for patch_name in solver.boundary.fields["epsilon"]:
        if patch_name in ("inlet", "outlet"):
            continue
        solver.boundary.set_raw_condition(patch_name, "epsilon", {
            "type": "epsilonWallFunction",
            "value": f"uniform {eps_inlet}",
        })

    # p — pression
    solver.boundary.set_raw_condition("inlet", "p", {
        "type": "zeroGradient",
    })
    solver.boundary.set_raw_condition("outlet", "p", {
        "type": "fixedValue",
        "value": "uniform 0",
    })
    for patch_name in solver.boundary.fields["p"]:
        if patch_name in ("inlet", "outlet"):
            continue
        solver.boundary.set_raw_condition(patch_name, "p", {
            "type": "zeroGradient",
        })

    # nut — turbulent viscosity (calculated on inlet/outlet, wall function on walls)
    solver.boundary.set_raw_condition("inlet", "nut", {
        "type": "calculated",
        "value": "uniform 0",
    })
    solver.boundary.set_raw_condition("outlet", "nut", {
        "type": "calculated",
        "value": "uniform 0",
    })
    for patch_name in solver.boundary.fields["nut"]:
        if patch_name in ("inlet", "outlet"):
            continue
        solver.boundary.set_raw_condition(patch_name, "nut", {
            "type": "nutkWallFunction",
            "value": "uniform 0",
        })

    # Write boundary condition files — OpenFOAMFile.write_boundary_file adds
    # #includeEtc "caseDicts/setConstraintTypes" by default.
    solver.boundary.write_boundary_conditions()

    # --- 7. Lancer la simulation ---
    print("\n" + "=" * 60)
    print("Lancement de la simulation (incompressibleFluid — buildingAero)")
    print("=" * 60)
    solver.run_simulation(nb_proc=1)

    # --- 8. Post-traitement ---
    print("\n" + "=" * 60)
    print("Post-traitement")
    print("=" * 60)

    # --- 8a. Résidus ---
    log_file = case_path / "log.incompressibleFluid"
    if log_file.exists():
        from foampilot.utilities.residuals import ResidualsPost

        residuals = ResidualsPost(log_file)
        residuals.process(export_csv=True, export_json=True, export_png=True, export_html=True)
        print("Résidus exportés (CSV + JSON + PNG + HTML).")

    # --- 8b. FoamPostProcessing — conversion VTK et visualisations ---
    foam_post = FoamPostProcessing(case_path=case_path)
    foam_post.foamToVTK()
    print("Conversion foamToVTK terminée.")

    time_steps = foam_post.get_all_time_steps()
    print(f"Pas de temps disponibles : {time_steps}")

    if time_steps:
        latest_time_step = time_steps[-1]
        structure = foam_post.load_time_step(latest_time_step)
        cell_mesh = structure["cell"]
        boundaries = structure["boundaries"]
        print(f"Maillage principal chargé (pas de temps {latest_time_step}) : {cell_mesh.n_cells} cellules")
        print(f"Frontières chargées : {list(boundaries.keys())}")

        # --- Visualisations ---
        print("\n--- Génération de visualisations ---")

        # Plot slice (pression)
        print("Génération du plot de coupe (p)...")
        foam_post.plot_slice(
            structure=structure,
            plane="z",
            scalars="p",
            opacity=0.25,
            path_filename=case_path / "slice_plot.png",
        )

        # Contour plot (pression)
        print("Génération du contour de pression...")
        pl_contour = pv.Plotter(off_screen=True)
        pl_contour.add_mesh(cell_mesh, scalars="p", show_scalar_bar=True)
        foam_post.export_plot(pl_contour, case_path / "contour_plot.png")

        # Vector plot (vitesse) — factor adapté à la taille du domaine
        print("Génération du champ de vecteurs (U)...")
        bounds = cell_mesh.bounds
        domain_length = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        glyph_factor = domain_length * 0.002

        # Subsample arrows for clarity — keep ~2000 glyphs max
        n_cells = cell_mesh.n_cells
        max_glyphs = 2000
        if n_cells > max_glyphs:
            step = max(1, n_cells // max_glyphs)
            subsample_indices = np.arange(0, n_cells, step)
            cell_mesh_reduced = cell_mesh.extract_cells(subsample_indices)
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

        # Mesh wireframe
        print("Génération du maillage fil de fer...")
        pl_mesh_style = pv.Plotter(off_screen=True)
        pl_mesh_style.add_mesh(cell_mesh, style="wireframe", show_edges=True, color="red")
        foam_post.export_plot(pl_mesh_style, case_path / "mesh_style_plot.png")

        # --- Analyse de flux avancée ---
        print("\n--- Analyse de flux avancée ---")

        # Q-criterion
        print("Calcul du critère Q...")
        mesh_with_q = foam_post.calculate_q_criterion(mesh=cell_mesh, velocity_field="U")
        if "q_criterion" in mesh_with_q.point_data:
            q_range = mesh_with_q.point_data["q_criterion"]
            print(f"  Critère Q : min={q_range.min():.2e}, max={q_range.max():.2e}")

        # Vorticité
        print("Calcul de la vorticité...")
        mesh_with_vorticity = foam_post.calculate_vorticity(mesh=cell_mesh, velocity_field="U")
        if "vorticity" in mesh_with_vorticity.point_data:
            vort_range = mesh_with_vorticity.point_data["vorticity"]
            print(f"  Vorticité : min={vort_range.min():.2e}, max={vort_range.max():.2e}")

        # --- Statistiques ---
        print("\n--- Statistiques ---")

        mesh_stats = foam_post.get_mesh_statistics(cell_mesh)
        print(f"Statistiques du maillage : {mesh_stats}")

        cell_region_stats = foam_post.get_region_statistics(structure, "cell", "U")
        print(f"Statistiques de la région 'cell' pour 'U' : {cell_region_stats}")

        cell_p_stats = foam_post.get_region_statistics(structure, "cell", "p")
        print(f"Statistiques de la région 'cell' pour 'p' : {cell_p_stats}")

        # Export cell data to CSV
        print("Export des données de la région 'cell' vers CSV...")
        foam_post.export_region_data_to_csv(structure, "cell", ["U", "p"], case_path / "cell_data.csv")

        # Compile and export statistics to JSON
        all_stats = {
            "mesh_stats": mesh_stats,
            "cell_region_stats_U": cell_region_stats,
            "cell_region_stats_p": cell_p_stats,
        }
        foam_post.export_statistics_to_json(all_stats, case_path / "all_stats.json")
        print("Statistiques exportées vers all_stats.json.")

        # Animation
        print("Création d'une animation du champ de vitesse...")
        foam_post.create_animation(scalars="U", filename=case_path / "animation_test.gif", fps=5)

    else:
        print("Aucun pas de temps trouvé.")

    print("\n" + "=" * 60)
    print("Simulation terminée avec succès !")
    print(f"  Cas      : {case_path}")
    print(f"  Log      : {case_path / 'log.incompressibleFluid'}")
    print(f"  Résultats: {case_path / 'postProcessing'}")
    print(f"  Visualisations: {case_path / 'slice_plot.png'}")
    print(f"  Contour  : {case_path / 'contour_plot.png'}")
    print(f"  Vecteurs : {case_path / 'vector_plot.png'}")
    print(f"  Maillage : {case_path / 'mesh_style_plot.png'}")
    print(f"  Stats    : {case_path / 'all_stats.json'}")
    print(f"  Données  : {case_path / 'cell_data.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
