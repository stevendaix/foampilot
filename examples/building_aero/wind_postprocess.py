#!/usr/bin/env python3
"""
Post-process wind rose CFD cases and compute Lawson pedestrian wind comfort maps.

Pipeline:
  1. Charge la rose des vents (wind_rose.json).
  2. Pour chaque cas CFD, charge le dernier time-step VTK et extrait la sensibilité
     piétonne via WindCaseResult (u_ref = vitesse d'entrée réelle du cas).
  3. Agrège tous les cas dans un WindEnsemble.
  4. Calcule les cartes Lawson (probability-of-exceedance) pour 4 seuils de confort.
  5. Génère les visualisations par cas (slices, Cp sur bâtiments, maillage).
  6. Génère les roses des vents agrégées (Cp, vitesse max, indice de confort).

Usage:
    PYTHONPATH=src python3 wind_postprocess.py \\
        [--cases-dir cases] [--wind-rose wind_rose.json] [--pedestrian-height 1.75]
"""

import argparse
import json
import sys
import numpy as np
import pyvista as pv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "foampilot" / "src"))

from foampilot.postprocess import FoamPostProcessing
from foampilot.postprocess.wind_analysis import (
    WindRose,
    WindCaseResult,
    WindEnsemble,
    LawsonProcessor,
    LawsonVisualizer,
    LAWSON_THRESHOLDS,
)

RHO_AIR = 1.225


def load_wind_rose(wind_rose_path: Path) -> WindRose:
    """Load wind rose from JSON (keys are strings of direction degrees)."""
    with open(wind_rose_path) as f:
        data = json.load(f)
    data = {float(k): v for k, v in data.items()}
    return WindRose(data)


def load_case_speed(case_dir: Path) -> float:
    """Read the actual inlet wind speed (m/s) stored in case_metadata.json.

    Falls back to 10.0 if the metadata file is missing.
    This speed is the u_ref used for sensitivity normalisation so that
    S = |U| / u_actual_case_speed (not a fixed 10 m/s).
    """
    meta_path = case_dir / "case_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return float(meta.get("speed_10m", 10.0))
    print("    ⚠ No case_metadata.json — falling back to u_ref=10.0")
    return 10.0


def build_ensemble(cases_dir, reference_height_speed=10.0, pedestrian_height=1.75):
    """Load all CFD cases and build a WindEnsemble.

    For each case directory (named wind_<direction>deg):
      - Vérifie que le dernier time-step est utilisé (get_structure → steps[-1])
      - foamToVTK si nécessaire
      - WindCaseResult avec u_ref = speed_10m du cas (pas 10.0 fixe)
    """
    ensemble = WindEnsemble()

    case_dirs = sorted(
        [d for d in cases_dir.iterdir()
         if d.is_dir() and d.name.startswith("wind_")],
        key=lambda d: float(d.name.replace("wind_", "").replace("deg", "")),
    )

    if not case_dirs:
        print(f"No case directories found in {cases_dir}")
        return ensemble

    for case_dir in case_dirs:
        direction = float(case_dir.name.replace("wind_", "").replace("deg", ""))
        actual_speed = load_case_speed(case_dir)

        print(f"\n  Loading case: {case_dir.name} (direction={direction}°, u_ref={actual_speed:.2f} m/s)")

        foam_post = FoamPostProcessing(case_path=case_dir)

        # Ensure VTK exists
        vtk_dir = case_dir / "VTK"
        if not vtk_dir.exists() or not list(vtk_dir.glob("*.vtk")):
            print("    Running foamToVTK...")
            foam_post.foamToVTK(fields=["U", "p"])

        # Vérifie on utilise bien le dernier time-step
        time_steps = foam_post.get_all_time_steps()
        if time_steps:
            latest = time_steps[-1]
            print(f"    Latest time-step: {latest} (sur {len(time_steps)} disponibles)")
        else:
            print("    ⚠ Aucun time-step VTK trouvé — conversion foamToVTK...")
            foam_post.foamToVTK(fields=["U", "p"])

        case = WindCaseResult(
            post=foam_post,
            direction_deg=direction,
            u_ref=actual_speed,
            field_name="U",
            pedestrian_height=pedestrian_height,
        )
        ensemble.add_case(direction, case)

    return ensemble


def run_lawson_analysis(ensemble, wind_rose, output_dir):
    """Run Lawson processor and generate probability maps.

    Uses sector_half_width=0 (exact direction match) — no angular regrouping.
    Chaque cas CFD est apparié à la direction de vent exacte dans wind_rose.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run all case extractions
    print("\nExtracting pedestrian planes and computing sensitivity...")
    ensemble.run_all()

    # Lawson analysis — no sector cut, exact match
    print("\nRunning Lawson analysis (sector_half_width=0, match exact)...")
    lawson = LawsonProcessor(
        ensemble=ensemble,
        wind_rose=wind_rose,
        sector_half_width=0.0,
    )

    lawson_maps = lawson.compute_lawson_maps()

    # Visualize using the reference (first) case mesh
    reference_case = next(iter(ensemble.cases.values()))
    reference_mesh = reference_case.mesh
    viz = LawsonVisualizer(reference_mesh)

    for label, data in lawson_maps.items():
        field_name = f"lawson_{label}"
        viz.add_probability_field(field_name, data)
        png_path = output_dir / f"lawson_{label}.png"
        viz.plot(field=field_name, filename=png_path)
        print(f"  Saved: {png_path}")
        if data is not None:
            print(f"    {label} (threshold={LAWSON_THRESHOLDS[label]} m/s): "
                  f"max={np.max(data):.6f}, mean={np.mean(data):.6f}")

    # Save the mesh with all Lawson fields
    vtk_path = output_dir / "lawson_results.vtk"
    reference_mesh.save(str(vtk_path))
    print(f"  Saved: {vtk_path}")

    return lawson_maps


def generate_per_case_visualizations(cases_dir, output_dir, pedestrian_height=1.75):
    """Generate per-case visualizations using the original (un-sampled) cell mesh.

    For each case:
      - Coupe horizontale piéton (Z=1.75m) — |U| contour
      - Coupe horizontale piéton — |U| contour avec bâtiments superposés
      - Champ de vecteurs |U| (glyphs) au niveau piéton
      - Coupe verticale Y-normale — `p`
      - Coupe verticale Y-normale — `|U|`
      - Cp sur les bâtiments (boundary "buildings")
      - Profil de vitesse d'entrée vs loi log
      - Cartographie qualité du maillage (aspect ratio)
      - Visualisation du maillage (wireframe)
    """
    from wind_profile import log_wind_profile, Z_REF

    viz_dir = output_dir / "carto_par_cas"
    viz_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(
        [d for d in cases_dir.iterdir()
         if d.is_dir() and d.name.startswith("wind_")],
        key=lambda d: float(d.name.replace("wind_", "").replace("deg", "")),
    )

    pv.set_jupyter_backend("none")
    pv.global_theme.background = "white"

    for case_dir in case_dirs:
        case_name = case_dir.name
        print(f"\n  Visualizing: {case_name}")

        foam_post = FoamPostProcessing(case_path=case_dir)
        time_steps = foam_post.get_all_time_steps()
        if not time_steps:
            print("    No time steps found, skipping.")
            continue

        latest = time_steps[-1]
        structure = foam_post.load_time_step(latest)
        cell_mesh = structure["cell"]
        boundaries = structure["boundaries"]
        bounds = cell_mesh.bounds

        # Load case metadata for Cp and inlet profile computation
        actual_speed = load_case_speed(case_dir)
        p_ref = 0.5 * RHO_AIR * actual_speed ** 2
        meta_path = case_dir / "case_metadata.json"
        z0 = 0.3
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            z0 = float(meta.get("z0", 0.3))

        # --- 1. Horizontal slice at pedestrian height — |U| ---
        try:
            slice_mesh = cell_mesh.slice(normal="z", origin=(0, 0, pedestrian_height))
            if slice_mesh.n_points > 0:
                U = slice_mesh.point_data.get("U")
                if U is not None:
                    mag = np.linalg.norm(U, axis=1)
                    slice_mesh.point_data["velocity_magnitude"] = mag
                pl = pv.Plotter(off_screen=True)
                pl.set_background("white")
                pl.add_mesh(
                    slice_mesh, scalars="velocity_magnitude",
                    cmap="viridis", show_scalar_bar=True,
                    scalar_bar_args={"title": "|U| (m/s)"},
                )
                pl.camera_position = "xy"
                pl.screenshot(str(viz_dir / f"{case_name}_slice_pedestrian.png"))
                pl.close()
        except Exception as e:
            print(f"    slice_pedestrian error: {e}")

        # --- 2. Vertical slice Y-normal — p (perpendicular to wind direction) ---
        try:
            cy = (bounds[2] + bounds[3]) / 2
            slice_v = cell_mesh.slice(normal="y", origin=(0, cy, 0))
            if slice_v.n_points > 0:
                pl = pv.Plotter(off_screen=True)
                pl.set_background("white")
                pl.add_mesh(
                    slice_v, scalars="p", cmap="RdBu_r",
                    show_scalar_bar=True, scalar_bar_args={"title": "p (Pa)"},
                )
                pl.camera_position = "xz"
                pl.screenshot(str(viz_dir / f"{case_name}_slice_vertical_p.png"))
                pl.close()
        except Exception as e:
            print(f"    slice_vertical_p error: {e}")

        # --- 3. Vertical slice Y-normal — |U| (velocity magnitude) ---
        try:
            if slice_v.n_points > 0 and "U" in slice_v.point_data:
                U = slice_v.point_data["U"]
                u_mag = np.linalg.norm(U, axis=1)
                slice_v.point_data["velocity_magnitude"] = u_mag
                pl = pv.Plotter(off_screen=True)
                pl.set_background("white")
                pl.add_mesh(
                    slice_v, scalars="velocity_magnitude",
                    cmap="viridis", show_scalar_bar=True,
                    scalar_bar_args={"title": "|U| (m/s)"},
                )
                pl.camera_position = "xz"
                pl.screenshot(str(viz_dir / f"{case_name}_slice_vertical_u.png"))
                pl.close()
        except Exception as e:
            print(f"    slice_vertical_u error: {e}")

        # --- 4. Horizontal slice |U| at pedestrian height ---
        try:
            ped_slice = cell_mesh.slice(normal="z", origin=(0, 0, pedestrian_height))
            if ped_slice.n_points > 0 and "U" in ped_slice.point_data:
                U = ped_slice.point_data["U"]
                u_mag = np.linalg.norm(U, axis=1)
                ped_slice.point_data["velocity_magnitude"] = u_mag
                pl = pv.Plotter(off_screen=True)
                pl.set_background("white")
                pl.add_mesh(
                    ped_slice, scalars="velocity_magnitude",
                    cmap="viridis", show_scalar_bar=True,
                    scalar_bar_args={"title": "|U| (m/s)"},
                )
                pl.camera_position = "xy"
                pl.screenshot(str(viz_dir / f"{case_name}_horizontal_u.png"))
                pl.close()
        except Exception as e:
            print(f"    horizontal_u error: {e}")

        # --- 4b. Horizontal slice |U| at 1m height ---
        try:
            slice_1m = cell_mesh.slice(normal="z", origin=(0, 0, 1.0))
            if slice_1m.n_points > 0 and "U" in slice_1m.point_data:
                U = slice_1m.point_data["U"]
                u_mag = np.linalg.norm(U, axis=1)
                slice_1m.point_data["velocity_magnitude"] = u_mag
                pl = pv.Plotter(off_screen=True)
                pl.set_background("white")
                pl.add_mesh(
                    slice_1m, scalars="velocity_magnitude",
                    cmap="viridis", show_scalar_bar=True,
                    scalar_bar_args={"title": "|U| (m/s)"},
                )
                pl.camera_position = "xy"
                pl.screenshot(str(viz_dir / f"{case_name}_horizontal_u_1m.png"))
                pl.close()
        except Exception as e:
            print(f"    horizontal_u_1m error: {e}")

        # --- 5. Cp on buildings boundary ---
        try:
            build_mesh = boundaries.get("buildings")
            if build_mesh is not None and "p" in build_mesh.point_data:
                p = build_mesh.point_data["p"]
                cp = p / p_ref
                build_mesh.point_data["Cp"] = cp
                pl = pv.Plotter(off_screen=True)
                pl.set_background("white")
                pl.add_mesh(
                    build_mesh, scalars="Cp", cmap="RdBu_r",
                    show_scalar_bar=True, scalar_bar_args={"title": "Cp"},
                    cpos="xy",
                )
                pl.camera_position = "xy"
                pl.screenshot(str(viz_dir / f"{case_name}_cp_buildings.png"))
                pl.close()
        except Exception as e:
            print(f"    cp_buildings error: {e}")

        # --- 5. Contour |U| at pedestrian height with buildings overlay ---
        try:
            ped_slice = cell_mesh.slice(normal="z", origin=(0, 0, pedestrian_height))
            if ped_slice.n_points > 0 and "U" in ped_slice.point_data:
                U = ped_slice.point_data["U"]
                mag = np.linalg.norm(U, axis=1)
                ped_slice.point_data["velocity_magnitude"] = mag
                pl = pv.Plotter(off_screen=True)
                pl.set_background("white")
                pl.add_mesh(
                    ped_slice, scalars="velocity_magnitude",
                    cmap="viridis", show_scalar_bar=True,
                    scalar_bar_args={"title": "|U| (m/s)"},
                )
                build_mesh = boundaries.get("buildings")
                if build_mesh is not None:
                    pl.add_mesh(build_mesh, color="black", opacity=0.4)
                pl.camera_position = "xy"
                pl.screenshot(str(viz_dir / f"{case_name}_contour_velocity_pedestrian.png"))
                pl.close()
        except Exception as e:
            print(f"    contour_velocity error: {e}")

        # --- 6. Vector plot (glyphs) |U| at pedestrian height ---
        try:
            ped_slice = cell_mesh.slice(normal="z", origin=(0, 0, pedestrian_height))
            if ped_slice.n_points > 0 and "U" in ped_slice.point_data:
                ped_slice.set_active_vectors("U")
                n_cells = ped_slice.n_cells
                max_glyphs = 500
                if n_cells > max_glyphs:
                    step = max(1, n_cells // max_glyphs)
                    ped_slice = ped_slice.extract_cells(np.arange(0, n_cells, step))
                domain_size = max(
                    bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]
                )
                glyph_factor = domain_size * 0.001
                arrows = ped_slice.glyph(orient="U", factor=glyph_factor, clamping=True)
                pl = pv.Plotter(off_screen=True)
                pl.set_background("white")
                pl.add_mesh(arrows, color="blue")
                pl.add_mesh(ped_slice, color="gray", opacity=0.1, show_edges=False)
                pl.camera_position = "xy"
                pl.screenshot(str(viz_dir / f"{case_name}_vector_pedestrian.png"))
                pl.close()
        except Exception as e:
            print(f"    vector_pedestrian error: {e}")

        # --- 7. Inlet velocity profile vs log law ---
        try:
            inlet = boundaries.get("INLET")
            if inlet is not None and "U" in inlet.point_data:
                U_in = inlet.point_data["U"]
                u_mag = np.linalg.norm(U_in, axis=1)
                pts = inlet.points
                z_vals = pts[:, 2]

                z_min = max(z_vals.min(), z0 + 1e-6)
                z_theory = np.linspace(z_min, z_vals.max(), 100)
                u_theory = log_wind_profile(z_theory, actual_speed, z0, Z_REF)

                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(z_vals, u_mag, s=10, alpha=0.5, label="CFD (OpenFOAM)", color="blue")
                ax.plot(z_theory, u_theory, "r-", linewidth=2, label="Loi log (théorique)")
                ax.axvline(x=10.0, color="green", linestyle="--", label="z_ref = 10 m")
                ax.set_xlabel("Hauteur z (m)")
                ax.set_ylabel("Vitesse |U| (m/s)")
                ax.set_title(f"Profil d'entrée — {case_name} (u_10m={actual_speed:.1f} m/s, z0={z0})")
                ax.legend()
                fig.tight_layout()
                fig.savefig(viz_dir / f"{case_name}_inlet_profile_check.png", dpi=150)
                plt.close(fig)
                print(f"    Saved: {case_name}_inlet_profile_check.png")
        except Exception as e:
            print(f"    inlet_profile error: {e}")

        # --- 8. Cartographie qualité du maillage (aspect ratio) ---
        try:
            quality_mesh = cell_mesh.compute_cell_quality()
            if "cell_quality" in quality_mesh.cell_data:
                ar = quality_mesh.cell_data["cell_quality"]
                mid_z = (bounds[4] + bounds[5]) / 2
                q_slice = quality_mesh.slice(normal="z", origin=(0, 0, mid_z))
                if q_slice.n_points > 0:
                    pl = pv.Plotter(off_screen=True)
                    pl.set_background("white")
                    ar_clipped = np.clip(
                        q_slice.point_data["cell_quality"], 0,
                        np.percentile(ar, 95) if len(ar) > 0 else 10.0,
                    )
                    q_slice.point_data["aspect_ratio"] = ar_clipped
                    pl.add_mesh(
                        q_slice, scalars="aspect_ratio",
                        cmap="plasma", show_scalar_bar=True,
                        scalar_bar_args={"title": "Aspect ratio"},
                    )
                    pl.camera_position = "xy"
                    pl.screenshot(str(viz_dir / f"{case_name}_mesh_quality.png"))
                    pl.close()
        except Exception as e:
            print(f"    mesh_quality error: {e}")

        # --- 9. Mesh visualization (wireframe) ---
        try:
            pl = pv.Plotter(off_screen=True)
            pl.set_background("black")
            pl.add_mesh(
                cell_mesh, style="wireframe", color="white",
                line_width=0.3, opacity=0.7,
            )
            pl.camera_position = "xy"
            pl.screenshot(str(viz_dir / f"{case_name}_mesh.png"))
            pl.close()
        except Exception as e:
            print(f"    mesh_viz error: {e}")

        print(f"    Done: {case_name}")


def generate_wind_rose_plots(ensemble, wind_rose, output_dir):
    """Generate aggregated wind rose plots.

    Produces (in output_dir/regroupement/):
      - cp_vs_direction.png : Cp moyen sur bâtiments vs angle du vent
      - max_velocity_vs_direction.png : Vitesse max piétonne vs angle du vent
      - comfort_vs_direction.png : Indice de confort (1 - proba walking) vs angle du vent
      - cp_distribution.png : Cp moyen par angle avec rose des vents superposée
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regroupe_dir = output_dir / "regroupement"
    regroupe_dir.mkdir(parents=True, exist_ok=True)

    df = ensemble.compute_case_metrics(rho=RHO_AIR)
    directions = df.index.values

    # Normalize Lawson walking probability per direction
    from foampilot.postprocess.wind_analysis import LawsonProcessor
    lawson = LawsonProcessor(ensemble, wind_rose, sector_half_width=0.0)
    walking_map = lawson.compute_probability_map(LAWSON_THRESHOLDS["walking"])
    walking_prob_per_case = {}
    for direction, case in ensemble.cases.items():
        n = case.mesh.n_points
        walking_prob_per_case[direction] = float(np.mean(walking_map[:n])) if walking_map is not None else 0.0

    # Wind rose frequencies per direction
    wind_freqs = []
    for d in directions:
        freq = 0.0
        for wd, wc_list in wind_rose.data.items():
            if abs(wd - d) < 1e-6:
                freq = sum(sb["frequency"] for sb in wc_list)
                break
        wind_freqs.append(freq)
    wind_freqs = np.array(wind_freqs)

    # --- Plot 1: Cp moyen sur bâtiments vs angle ---
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": "polar"})
    angles_rad = np.radians(directions)
    cp_vals = df["mean_cp_buildings"].values
    ax.plot(angles_rad, cp_vals, "ro-", markersize=6, linewidth=1.5)
    ax.fill(angles_rad, cp_vals, alpha=0.15, color="red")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Cp moyen sur les bâtiments vs direction de vent", fontsize=13, pad=20)
    fig.savefig(regroupe_dir / "cp_vs_direction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {regroupe_dir / 'cp_vs_direction.png'}")

    # --- Plot 2: Vitesse max piétonne vs angle ---
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": "polar"})
    max_vel = df["max_velocity_street"].values
    ax.plot(angles_rad, max_vel, "bo-", markersize=6, linewidth=1.5)
    ax.fill(angles_rad, max_vel, alpha=0.15, color="blue")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Vitesse max piétonne entre bâtiments vs direction de vent", fontsize=13, pad=20)
    fig.savefig(regroupe_dir / "max_velocity_vs_direction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {regroupe_dir / 'max_velocity_vs_direction.png'}")

    # --- Plot 3: Indice de confort vs angle ---
    comfort = np.array([
        1.0 - walking_prob_per_case.get(d, 0.0) for d in directions
    ])
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": "polar"})
    ax.plot(angles_rad, comfort, "go-", markersize=6, linewidth=1.5)
    ax.fill(angles_rad, comfort, alpha=0.15, color="green")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_title("Indice de confort piéton (1 - proba walking) vs direction de vent", fontsize=12, pad=20)
    fig.savefig(regroupe_dir / "comfort_vs_direction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {regroupe_dir / 'comfort_vs_direction.png'}")

    # --- Plot 4: Cp distribution with wind frequency overlay ---
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(np.arange(len(directions)), cp_vals, color="red", alpha=0.6, label="Cp moyen sur bâtiments", width=20)
    ax1.set_xlabel("Direction de vent (°)")
    ax1.set_ylabel("Cp moyen sur bâtiments", color="red")
    ax1.tick_params(axis="y", labelcolor="red")

    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(directions)), wind_freqs * 100, "b.-", markersize=8, label="Fréquence rose des vents (%)")
    ax2.set_ylabel("Fréquence (%)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    direction_labels = [f"{int(d)}" for d in directions]
    ax1.set_xticks(np.arange(len(directions)))
    ax1.set_xticklabels(direction_labels, rotation=45, fontsize=8)
    ax1.set_title("Cp moyen sur bâtiments & rose des vents par direction")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    fig.tight_layout()
    fig.savefig(regroupe_dir / "cp_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {regroupe_dir / 'cp_distribution.png'}")

    # --- Plot 5: Wind rose polar plot of EPW frequencies ---
    all_dirs = sorted(wind_rose.data.keys())
    dir_rad = [np.radians(d) for d in all_dirs]

    # Also plot per-speed-bin frequencies (layered)
    speed_bins_all = sorted(set(
        sb["speed"] for d in all_dirs for sb in wind_rose.data[d]
    ))

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    colors = plt.cm.viridis(np.linspace(0, 1, len(speed_bins_all)))
    width = 0.8 * (22.5 * np.pi / 180)  # bin width in radians

    for i, sb_speed in enumerate(speed_bins_all):
        freqs = []
        for d in all_dirs:
            found = False
            for sb in wind_rose.data[d]:
                if abs(sb["speed"] - sb_speed) < 1e-6:
                    freqs.append(sb["frequency"])
                    found = True
                    break
            if not found:
                freqs.append(0.0)
        ax.bar(dir_rad, freqs, width=width, bottom=i * 0.001,
               color=colors[i], alpha=0.6, label=f"{sb_speed:.0f} m/s")

    ax.set_title("Rose des vents EPW — fréquence par direction & vitesse", fontsize=13, pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0), fontsize=8)
    fig.savefig(regroupe_dir / "wind_rose_epw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {regroupe_dir / 'wind_rose_epw.png'}")

    # Save metrics as CSV
    df.insert(0, "wind_frequency", wind_freqs)
    df.insert(len(df.columns), "comfort_index", comfort)
    df.to_csv(regroupe_dir / "case_metrics.csv")
    print(f"  Saved: {regroupe_dir / 'case_metrics.csv'}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Post-process wind rose CFD cases with Lawson analysis")
    parser.add_argument("--cases-dir", default="cases", help="Directory with case subfolders")
    parser.add_argument("--wind-rose", default="wind_rose.json", help="Path to wind_rose.json")
    parser.add_argument("--pedestrian-height", type=float, default=1.75, help="Pedestrian height (m)")
    parser.add_argument("--skip-viz", action="store_true", help="Skip per-case visualizations")
    parser.add_argument("--skip-rose", action="store_true", help="Skip wind rose aggregation plots")
    parser.add_argument("--output-dir", default="post", help="Output directory for results")
    args = parser.parse_args()

    cases_dir = Path(args.cases_dir)
    wind_rose_path = Path(args.wind_rose)
    output_dir = Path(args.output_dir)

    # --- Load wind rose ---
    wind_rose = load_wind_rose(wind_rose_path)
    n_directions = len(wind_rose.data)
    total_freq = sum(sum(sb["frequency"] for sb in sb_list) for sb_list in wind_rose.data.values())
    print(f"Wind rose: {n_directions} sector directions, total frequency = {total_freq:.4f}")

    # --- Build ensemble ---
    ensemble = build_ensemble(
        cases_dir, reference_height_speed=10.0,
        pedestrian_height=args.pedestrian_height,
    )
    n_cases = len(ensemble.cases)
    print(f"\nEnsemble built: {n_cases} cases")

    if n_cases == 0:
        print("No cases found. Run generate_wind_cases.py first.")
        sys.exit(1)

    # --- Lawson analysis ---
    lawson_dir = output_dir / "lawson"
    print("\nRunning Lawson analysis...")
    run_lawson_analysis(ensemble, wind_rose, lawson_dir)

    # --- Per-case visualizations ---
    if not args.skip_viz:
        print("\nGenerating per-case visualizations...")
        generate_per_case_visualizations(cases_dir, output_dir,
                                          pedestrian_height=args.pedestrian_height)

    # --- Wind rose aggregation plots ---
    if not args.skip_rose:
        print("\nGenerating wind rose aggregation plots...")
        generate_wind_rose_plots(ensemble, wind_rose, output_dir)

    print(f"\n{'=' * 60}")
    print(f"Post-processing complete. Results in: {output_dir}")
    print(f"  {output_dir}/lawson/          — cartes Lawson + VTK")
    print(f"  {output_dir}/carto_par_cas/   — visualisations par cas")
    print(f"  {output_dir}/regroupement/    — roses des vents agrégées")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
