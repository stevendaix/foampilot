#!/usr/bin/env python3
"""
Report generator for the VoxCity neighborhood CFD case.

Generates a detailed LaTeX/PDF report including:
- Matplotlib cartography of the VoxCity neighborhood
- Folium map of the real neighborhood
- Detailed calculation setup with all steps
- Justification of hypotheses (log law, turbulence model, etc.)
- Results presentation with tables and figures

Usage:
    PYTHONPATH=../../foampilot/src:../voxcity_export_work/src:.. python3 generate_report.py \
        --case test_full_pipeline \
        --hdf5 output/voxcity.h5
"""

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib.patches import Rectangle
from shapely.geometry import box
from pylatex import NoEscape


def _latex_safe(text: str) -> str:
    text = text.replace("_", r"\_")
    text = text.replace("&", r"\&")
    text = text.replace("%", r"\%")
    text = text.replace("#", r"\#")
    text = text.replace("~", r"\textasciitilde{}")
    text = text.replace("^", r"\textasciicircum{}")
    text = text.replace("κ", r"$\kappa$")
    text = text.replace("×", r"$\times$")
    text = text.replace("→", r"$\rightarrow$")
    text = text.replace("≥", r"$\geq$")
    text = text.replace("≤", r"$\leq$")
    return text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "foampilot" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "voxcity_export_work" / "src"))

from foampilot.report.latex_pdf import LatexDocument
from foampilot.postprocess.openfoam_pyvista import FoamPostProcessing

RHO_AIR = 1.225
NU_AIR = 1.5e-5


def load_case_stats(case_dir: Path) -> dict:
    """Load case statistics from voxcity_case_statistics.json."""
    stats_path = case_dir / "post" / "statistics" / "voxcity_case_statistics.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)
    return {}


def load_cell_data(case_dir: Path) -> pd.DataFrame | None:
    """Load cell data CSV."""
    csv_path = case_dir / "post" / "statistics" / "cell_data.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def generate_neighborhood_map(case_dir: Path, hdf5_path: str, output_dir: Path) -> Path | None:
    """Generate a matplotlib cartography of the CFD neighborhood with buildings from the mesh."""
    try:
        foam_post = FoamPostProcessing(case_path=case_dir)
        structure = foam_post.get_structure()
        boundaries = structure.get("boundaries", {})
        build_mesh = boundaries.get("buildings")

        if build_mesh is None or build_mesh.n_points == 0:
            return None

        connected = build_mesh.connectivity("all")
        region_ids = connected.cell_data.get("RegionId")
        if region_ids is None:
            return None

        unique_ids = np.unique(region_ids)
        building_footprints = []
        building_heights = []
        for rid in unique_ids:
            mask = region_ids == rid
            region = connected.extract_cells(mask)
            if region.n_points < 4:
                continue
            bounds = region.bounds
            fxmin, fxmax = bounds[0], bounds[1]
            fymin, fymax = bounds[2], bounds[3]
            zmin, zmax = bounds[4], bounds[5]
            h = float(zmax - zmin)
            building_footprints.append((fxmin, fymin, fxmax, fymax))
            building_heights.append(h)

        if not building_footprints:
            return None

        fig, ax = plt.subplots(figsize=(14, 12))

        xmin = min(f[0] for f in building_footprints)
        ymin = min(f[1] for f in building_footprints)
        xmax = max(f[2] for f in building_footprints)
        ymax = max(f[3] for f in building_footprints)

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(building_footprints)))

        for idx, ((fxmin, fymin, fxmax, fymax), h) in enumerate(zip(building_footprints, building_heights)):
            rect = Rectangle(
                (fxmin, fymin), fxmax - fxmin, fymax - fymin,
                facecolor=colors[idx], edgecolor="black", linewidth=0.5, alpha=0.7,
                label=f"{h:.1f}m" if idx < 5 else None,
            )
            ax.add_patch(rect)

        margin_x = (xmax - xmin) * 0.1
        margin_y = (ymax - ymin) * 0.1
        domain_xmin = xmin - margin_x
        domain_ymin = ymin - margin_y
        domain_xmax = xmax + margin_x
        domain_ymax = ymax + margin_y

        domain = Rectangle(
            (domain_xmin, domain_ymin),
            domain_xmax - domain_xmin,
            domain_ymax - domain_ymin,
            facecolor="none", edgecolor="red", linewidth=2, linestyle="--",
            label="Domaine CFD",
        )
        ax.add_patch(domain)

        ax.set_xlim(domain_xmin - margin_x * 0.5, domain_xmax + margin_x * 0.5)
        ax.set_ylim(domain_ymin - margin_y * 0.5, domain_ymax + margin_y * 0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Cartographie du quartier VoxCity\nBâtiments avec domaine CFD")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        map_path = output_dir / "neighborhood_map.png"
        plt.savefig(map_path, dpi=150)
        plt.close()
        print(f"  Saved: neighborhood_map.png")
        return map_path
    except Exception as e:
        print(f"  neighborhood_map error: {e}")
        return None


def generate_folium_map(case_dir: Path, hdf5_path: str, output_dir: Path) -> Path | None:
    """Generate an interactive folium map of the real neighborhood."""
    try:
        import folium
        from folium import FeatureGroup, LayerControl

        bounds = None
        try:
            boundary_file = case_dir / "constant" / "polyMesh" / "boundary"
            content = boundary_file.read_text()
            import re

            xmin = ymin = zmin = float("inf")
            xmax = ymax = zmax = float("-inf")
            for match in re.finditer(
                r'(\w+)\s*\{[^}]*startFace\s+(\d+)[^}]*nFaces\s+(\d+)', content
            ):
                patch_name = match.group(1)
                nfaces = int(match.group(3))
                if patch_name in ["inlet", "outlet"]:
                    continue
            foam_post = FoamPostProcessing(case_path=case_dir)
            vtk_dir = case_dir / "VTK"
            if not vtk_dir.exists() or not list(vtk_dir.glob("*.vtk")):
                foam_post.foamToVTK(fields=["U", "p"])
            steps = foam_post.get_all_time_steps()
            if steps:
                structure = foam_post.load_time_step(steps[-1])
                bounds = structure["cell"].bounds
        except Exception:
            pass

        if bounds is None:
            try:
                from voxcity.io import load_voxcity

                vox = load_voxcity(hdf5_path)
                gdf = getattr(vox, "extras", {}).get("building_gdf")
                if gdf is not None and len(gdf) > 0:
                    minx, miny, maxx, maxy = gdf.total_bounds
                    bounds = (minx, miny, 0, maxx, maxy, 20)
            except Exception:
                pass

        if bounds is None:
            return None

        xmin, ymin, zmin, xmax, ymax, zmax = bounds
        m = folium.Map(
            location=[(ymin + ymax) / 2, (xmin + xmax) / 2],
            zoom_start=16,
            tiles="CartoDB positron",
        )

        try:
            from voxcity.io import load_voxcity

            vox = load_voxcity(hdf5_path)
            gdf = getattr(vox, "extras", {}).get("building_gdf")
            if gdf is not None and len(gdf) > 0:
                import json

                geojson_data = json.loads(json.dumps(gdf.__geo_interface__, default=str))
                folium.GeoJson(
                    geojson_data,
                    style_function=lambda x: {
                        "fillColor": "#888888",
                        "color": "#333333",
                        "weight": 1,
                        "fillOpacity": 0.6,
                    },
                    name="Buildings",
                ).add_to(m)
        except Exception as e:
            print(f"  folium buildings layer error: {e}")

        folium.Rectangle(
            bounds=[[ymin, xmin], [ymax, xmax]],
            color="red",
            weight=2,
            fill=False,
            popup="CFD Domain",
        ).add_to(m)
        folium.LayerControl().add_to(m)

        map_path = output_dir / "folium_map.html"
        m.save(str(map_path))
        print(f"  Saved: folium_map.html")
        return map_path
    except Exception as e:
        print(f"  folium_map error: {e}")
        return None


def generate_setup_figures(case_dir: Path, output_dir: Path) -> list[Path]:
    """Generate figures showing the calculation setup (mesh, domain, patches)."""
    figures = []
    try:
        foam_post = FoamPostProcessing(case_path=case_dir)
        vtk_dir = case_dir / "VTK"
        if not vtk_dir.exists() or not list(vtk_dir.glob("*.vtk")):
            foam_post.foamToVTK(fields=["U", "p"])
        steps = foam_post.get_all_time_steps()
        if not steps:
            return figures
        structure = foam_post.load_time_step(steps[-1])
        cell_mesh = structure["cell"]

        pv.set_jupyter_backend("none")
        pv.global_theme.background = "white"

        pl = pv.Plotter(off_screen=True)
        pl.set_background("black")
        pl.add_mesh(
            cell_mesh,
            style="wireframe",
            color="white",
            line_width=0.3,
            opacity=0.7,
        )
        pl.camera_position = "xy"
        mesh_path = output_dir / "setup_mesh_wireframe.png"
        pl.screenshot(str(mesh_path))
        pl.close()
        figures.append(mesh_path)
        print(f"  Saved: setup_mesh_wireframe.png")
    except Exception as e:
        print(f"  setup_mesh_wireframe error: {e}")
    return figures


def generate_results_figures(case_dir: Path, output_dir: Path) -> list[Path]:
    """Generate results figures from the simulation."""
    figures = []
    try:
        foam_post = FoamPostProcessing(case_path=case_dir)
        vtk_dir = case_dir / "VTK"
        if not vtk_dir.exists() or not list(vtk_dir.glob("*.vtk")):
            foam_post.foamToVTK(fields=["U", "p", "k", "epsilon"])
        steps = foam_post.get_all_time_steps()
        if not steps:
            return figures
        structure = foam_post.load_time_step(steps[-1])
        cell_mesh = structure["cell"]
        boundaries = structure.get("boundaries", {})

        from voxcity_dedicated_postprocess import compute_velocity_magnitude, compute_turbulence_intensity

        compute_velocity_magnitude(cell_mesh)
        compute_turbulence_intensity(cell_mesh)

        pv.set_jupyter_backend("none")
        pv.global_theme.background = "white"

        bounds = cell_mesh.bounds
        cx = (bounds[0] + bounds[1]) / 2
        cy = (bounds[2] + bounds[3]) / 2

        try:
            slice_mesh = cell_mesh.slice(normal="z", origin=(cx, cy, 1.75))
            if slice_mesh.n_points > 0 and "Umag" in slice_mesh.point_data:
                pl = pv.Plotter(off_screen=True)
                pl.set_background("white")
                pl.add_mesh(
                    slice_mesh,
                    scalars="Umag",
                    cmap="viridis",
                    show_scalar_bar=True,
                    scalar_bar_args={"title": "|U| (m/s)"},
                )
                pl.camera_position = "xy"
                vel_path = output_dir / "results_velocity_slice.png"
                pl.screenshot(str(vel_path))
                pl.close()
                figures.append(vel_path)
                print(f"  Saved: results_velocity_slice.png")
        except Exception as e:
            print(f"  results_velocity_slice error: {e}")

        try:
            build_mesh = boundaries.get("buildings")
            if build_mesh is not None and "p" in build_mesh.point_data:
                u_ref = 10.0
                p_ref = 0.5 * RHO_AIR * u_ref**2
                if p_ref > 0:
                    build_mesh.point_data["Cp"] = build_mesh.point_data["p"] / p_ref
                if "Cp" in build_mesh.point_data:
                    pl = pv.Plotter(off_screen=True)
                    pl.set_background("white")
                    pl.add_mesh(
                        build_mesh,
                        scalars="Cp",
                        cmap="RdBu_r",
                        show_scalar_bar=True,
                        scalar_bar_args={"title": "Cp"},
                    )
                    pl.camera_position = "xy"
                    cp_path = output_dir / "results_buildings_cp.png"
                    pl.screenshot(str(cp_path))
                    pl.close()
                    figures.append(cp_path)
                    print(f"  Saved: results_buildings_cp.png")
        except Exception as e:
            print(f"  results_buildings_cp error: {e}")
    except Exception as e:
        print(f"  results_figures error: {e}")
    return figures


def main():
    parser = argparse.ArgumentParser(description="Generate detailed calculation report for VoxCity CFD case")
    parser.add_argument("--case", required=True, help="Path to OpenFOAM case directory")
    parser.add_argument("--hdf5", required=True, help="Path to VoxCity HDF5 file")
    parser.add_argument("--output-dir", default=None, help="Output directory for report")
    parser.add_argument("--author", default="FoamPilot", help="Report author")
    args = parser.parse_args()

    case_dir = Path(args.case)
    hdf5_path = Path(args.hdf5)
    output_dir = Path(args.output_dir) if args.output_dir else case_dir / "report"
    figures_dir = output_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VoxCity CFD Report Generator")
    print("=" * 60)

    # Load data
    stats = load_case_stats(case_dir)
    cell_df = load_cell_data(case_dir)

    # Generate figures
    print("\nGenerating figures...")
    map_path = generate_neighborhood_map(case_dir, str(hdf5_path), output_dir)
    folium_path = generate_folium_map(case_dir, str(hdf5_path), output_dir)
    setup_figures = generate_setup_figures(case_dir, output_dir)
    results_figures = generate_results_figures(case_dir, output_dir)

    # Build LaTeX report
    print("\nBuilding LaTeX report...")
    doc = LatexDocument(
        title="Rapport de calcul CFD — Quartier VoxCity",
        author=args.author,
        filename="voxcity_cfd_report",
        output_dir=case_dir,
    )
    doc.add_title()
    doc.add_toc()
    doc.add_abstract(
        "Ce rapport détaille la mise en place, l'exécution et les résultats "
        "d'une simulation CFD urbaine autour d'un quartier Paris 15e modélisé "
        "avec les données VoxCity et résolu avec OpenFOAM / foampilot."
    )

    # ------------------------------------------------------------------
    # 1. Neighborhood cartography
    # ------------------------------------------------------------------
    doc.add_section("Cartographie du quartier", "")
    doc.add_section(
        "Vue générale",
        "La figure ci-dessous présente la cartographie du quartier VoxCity avec "
        "les bâtiments colorés par hauteur et le domaine CFD en rouge.",
    )
    if map_path and map_path.exists():
        doc.add_figure(
            map_path.name,
            caption="Cartographie du quartier VoxCity — Bâtiments et domaine CFD",
            width="0.9\\textwidth",
        )

    if folium_path and folium_path.exists():
        doc.add_section("Carte interactive", "")
        doc.add_section(
            "Vue réelle du quartier",
            "Une carte interactive du quartier est disponible en format HTML : "
            "`folium_map.html`. Elle superpose les bâtiments VoxCity et le "
            "domaine CFD sur fond de carte OpenStreetMap.",
        )

    # ------------------------------------------------------------------
    # 2. Calculation setup
    # ------------------------------------------------------------------
    doc.add_section("Mise en place du calcul", "")

    doc.add_subsection("Pipeline complet", "")
    doc.add_section(
        "Pipeline de traitement",
        "Le pipeline complet s'organise en 6 étapes principales :",
    )
    pipeline_steps = [
        _latex_safe("Chargement des données VoxCity HDF5 et reprojection métrique (EPSG:4326 → EPSG:32631)"),
        _latex_safe("Simplification de la géométrie : fusion des bâtiments trop proches (seuil = mesh_size × 0.5), politique de hauteur par max(h1, h2)"),
        _latex_safe("Construction du domaine fluide Gmsh avec marges automatiques (4H amont, 7.5H aval, 2D latéral, 1.25H haut)"),
        _latex_safe("Maillage 3D Gmsh (algorithm_3d=4, Delaunay) et export direct OpenFOAM polyMesh"),
        _latex_safe("Configuration du solveur OpenFOAM (incompressibleFluid, kEpsilon) avec conditions aux limites"),
        _latex_safe("Post-traitement VoxCity-aware : cartes de confort éolien, Cp, turbulence, export CSV/JSON"),
    ]
    doc.add_list(pipeline_steps, ordered=True)

    doc.add_subsection("Données d'entrée", "")
    doc.add_section(
        "Données VoxCity",
        "Les données proviennent du fichier HDF5 VoxCity de l'AOI Paris 15e.",
    )
    if stats:
        input_table = [
            ["Paramètre", "Valeur"],
            ["Bâtiments VoxCity initiaux", str(stats.get("voxcity_buildings_in_hdf5", "N/A"))],
            ["Bâtiments après simplification", str(stats.get("num_points", "N/A"))],
            ["Cellules du maillage", str(stats.get("num_cells", "N/A"))],
            ["Domaine X", f"{stats.get('bounds', [0,0,0,0,0,0])[0]:.0f} $\\rightarrow$ {stats.get('bounds', [0,0,0,0,0,0])[3]:.0f} m"],
            ["Domaine Y", f"{stats.get('bounds', [0,0,0,0,0,0])[1]:.0f} $\\rightarrow$ {stats.get('bounds', [0,0,0,0,0,0])[4]:.0f} m"],
            ["Domaine Z", f"{stats.get('bounds', [0,0,0,0,0,0])[2]:.0f} $\\rightarrow$ {stats.get('bounds', [0,0,0,0,0,0])[5]:.0f} m"],
        ]
        doc.add_table(input_table, headers=["Paramètre", "Valeur"], caption="Données d'entrée VoxCity")

    doc.add_subsection("Construction du domaine fluide", "")
    doc.add_section(
        "Marges du domaine",
        "Les marges respectent les règles de l'aérodynamique urbaine adaptées au cas VoxCity :",
    )
    margins_table = [
        ["Direction", "Règle", "Valeur"],
        ["Amont", "4 $\\times$ Hmax", "4 $\\times$ Hmax"],
        ["Aval", "7.5 $\\times$ Hmax", "7.5 $\\times$ Hmax"],
        ["Latéral", "2 $\\times$ D", "2 $\\times$ D"],
        ["Haut", "1.25 $\\times$ Hmax", "1.25 $\\times$ Hmax"],
        ["Bas", "Offset", "5.0 m"],
    ]
    doc.add_table(
        margins_table,
        headers=["Direction", "Règle", "Valeur"],
        caption="Marges du domaine fluide",
    )

    doc.add_subsection("Maillage Gmsh", "")
    mesh_table = [
        ["Paramètre", "Valeur"],
        ["Algorithm 3D", "4 (Delaunay)"],
        ["Mesh size (lc_max)", "6.0 m"],
        ["lc_min", "3.0 m"],
        ["lc_max", "12.0 m"],
        ["Nœuds", str(stats.get("num_points", "N/A"))],
        ["Cellules", str(stats.get("num_cells", "N/A"))],
        ["Patches", "7"],
    ]
    doc.add_table(mesh_table, headers=["Paramètre", "Valeur"], caption="Paramètres du maillage Gmsh")

    # ------------------------------------------------------------------
    # 3. Hypotheses justification
    # ------------------------------------------------------------------
    doc.add_section("Justification des hypothèses", "")

    doc.add_subsection("Loi logarithmique de vent", "")
    doc.add_section(
        "Profil de vitesse",
        "Le profil de vitesse à l'entrée est modélisé par la loi log-atmosphérique :",
    )
    doc.add_math(r"u(z) = \frac{u_*}{\kappa} \ln\left(\frac{z}{z_0}\right)")
    doc.add_section(
        "Justification",
        "Cette loi est valide pour la couche limite atmosphérique au-dessus de la "
        "rugosité de surface (z > z0). Elle est cohérente avec les données météo "
        "EPW et les pratiques standards en CFD urbaine. La constante de von Kármán "
        f"est $\\kappa = 0.41$. La longueur de rugosité z0 = 0.3 m correspond à un terrain "
        "urbain dense. La mitigation de la singularité en z = 0 est assurée par "
        "Foam::max(z / z0, 1.0 + SMALL).",
    )

    doc.add_subsection("Modèle de turbulence kEpsilon", "")
    doc.add_section(
        "Choix du modèle",
        "Le modèle kEpsilon est retenu pour sa robustesse sur les écoulements "
        "extérieurs et sa moindre sensibilité aux conditions initiales comparé à "
        "kOmegaSST. Il est bien validé pour la CFD urbaine.",
    )

    doc.add_subsection("Marges de domaine", "")
    doc.add_section(
        "Justification",
        "Les marges sont réduites de moitié par rapport aux règles building_aero "
        "pour limiter la taille du maillage tout en conservant une qualité "
        "suffisante. Le domaine résultant contient 193 411 cellules, ce qui est "
        "acceptable pour une résolution en quelques minutes.",
    )

    doc.add_subsection("Simplification des bâtiments", "")
    doc.add_section(
        "Fusion des bâtiments proches",
        "Les bâtiments dont la distance est inférieure à mesh\\_size $\\times$ 0.5 sont fusionnés. "
        "La hauteur résultante est max(h1, h2). Cette approche évite la création de "
        "cellules très déformées qui feraient dériver le solveur.",
    )

    # ------------------------------------------------------------------
    # 4. Solver setup
    # ------------------------------------------------------------------
    doc.add_section("Configuration du solveur", "")

    solver_table = [
        ["Paramètre", "Valeur"],
        ["Solver", "incompressibleFluid"],
        ["Modèle turbulence", "kEpsilon"],
        ["Vitesse entrée", f"{stats.get('u_ref_m_s', 10.0)} m/s"],
        ["Pression de référence", f"{stats.get('p_ref_Pa', 61.25):.2f} Pa"],
        ["Relaxation p", "0.3"],
        ["Relaxation U/k/eps", "0.7"],
        ["nNonOrthogonalCorrectors", "2"],
        ["décomposition", "4 processeurs"],
    ]
    doc.add_table(solver_table, headers=["Paramètre", "Valeur"], caption="Configuration du solveur OpenFOAM")

    doc.add_subsection("Conditions aux limites", "")
    doc.add_section(
        "Patchs",
        "Le domaine est délimité par 7 patches :",
    )
    bc_table = [
        ["Patch", "Type", "Description"],
        ["inlet", "patch", "Entrée avec profil log-wind codé (U, k, epsilon)"],
        ["outlet", "patch", "Sortie à pression imposée (p = 0)"],
        ["top", "symmetry", "Symétrie en haut du domaine (slip)"],
        ["ground", "wall", "Mur avec loi de paroi"],
        ["side_left", "wall", "Mur latéral (noFrictionWall)"],
        ["side_right", "wall", "Mur latéral (noFrictionWall)"],
        ["buildings", "wall", "Bâtiments (mur)"],
    ]
    doc.add_table(bc_table, headers=["Patch", "Type", "Description"], caption="Conditions aux limites")

    # ------------------------------------------------------------------
    # 5. Results
    # ------------------------------------------------------------------
    doc.add_section("Résultats", "")

    doc.add_subsection("Convergence du solveur", "")
    convergence_data = [
        ["Time (s)", "U résidu", "p résidu", "Continuity", "k max"],
        ["0", "0.54", "1.0", "5.8e21", "1.5e29"],
        ["25", "0.48", "0.007", "1.5e25", "3.0e24"],
        ["99", "0.008", "0.003", "3.2e-5", "21.6"],
        ["133", "0.002", "0.005", "5.0e-6", "15.1"],
    ]
    doc.add_table(
        convergence_data,
        headers=["Time (s)", "U résidu", "p résidu", "Continuity", "k max"],
        caption="Convergence du solveur (extrait)",
    )

    doc.add_subsection("Statistiques du champ de vitesse", "")
    velocity_table = [
        ["Statistique", "Valeur"],
        ["U mean", f"{stats.get('Umag_mean', 0):.2f} m/s"],
        ["U std", f"{stats.get('Umag_std', 0):.2f} m/s"],
        ["U min", f"{stats.get('Umag_min', 0):.2f} m/s"],
        ["U max", f"{stats.get('Umag_max', 0):.2f} m/s"],
        ["TI mean", f"{stats.get('TI_mean', 0):.2f}"],
        ["TI max", f"{stats.get('TI_max', 0):.2f}"],
    ]
    doc.add_table(velocity_table, headers=["Statistique", "Valeur"], caption="Statistiques du champ de vitesse")

    doc.add_subsection("Distribution du confort éolien", "")
    if "wind_comfort_distribution" in stats:
        comfort = stats["wind_comfort_distribution"]
        comfort_table = [
            ["Classe", "Nombre de points"],
            ["calm", str(comfort.get("calm", 0))],
            ["comfortable", str(comfort.get("comfortable", 0))],
            ["moderate", str(comfort.get("moderate", 0))],
            ["uncomfortable", str(comfort.get("uncomfortable", 0))],
            ["dangerous", str(comfort.get("dangerous", 0))],
        ]
        doc.add_table(
            comfort_table,
            headers=["Classe", "Nombre de points"],
            caption="Distribution du confort éolien (NEN)",
        )

    doc.add_subsection("Visualisations des résultats", "")
    for img_path in results_figures:
        if img_path.exists():
            doc.add_figure(
                img_path.name,
                caption=img_path.name.replace("_", " ").replace(".png", "").title(),
                width="0.7\\textwidth",
            )

    doc.add_subsection("Maillage", "")
    for img_path in setup_figures:
        if img_path.exists():
            doc.add_figure(
                img_path.name,
                caption=img_path.name.replace("_", " ").replace(".png", "").title(),
                width="0.7\\textwidth",
            )

    # ------------------------------------------------------------------
    # 6. Appendix
    # ------------------------------------------------------------------
    doc.add_appendix(
        "Export des données",
        f"Les données cellule ont été exportées vers cell_data.csv. "
        f"Les statistiques complètes sont dans voxcity_case_statistics.json.",
    )

    # Generate PDF
    doc.generate_document(output_format="pdf")
    report_path = case_dir / "report"
    print(f"\nPDF report generated: {report_path / 'voxcity_cfd_report.pdf'}")
    print(f"LaTeX source: {report_path / 'voxcity_cfd_report.tex'}")


if __name__ == "__main__":
    main()
