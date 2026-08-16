#!/usr/bin/env python3
"""
Post-processing for patient58 CFD case using foampilot + PyVista APIs.

Reads the OpenFOAM case directly (no foamToVTK, foamLog required for log parsing only)
and generates velocity/pressure visualizations and statistics.
"""
import json
import logging
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import numpy as np
import pandas as pd
import pyvista as pv

from foampilot.postprocess import OpenFOAMDirectReader
from foampilot.postprocess.openfoam_pyvista import NumpyEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

pv.OFF_SCREEN = True
CASE_DIR = Path(__file__).resolve().parent.parent


def parse_residuals(log_path: Path) -> dict:
    """Parse simpleFoam log for final residuals."""
    residuals = {"Ux": [], "Uy": [], "Uz": [], "p": []}
    if not log_path.exists():
        logger.warning(f"Log file not found: {log_path}")
        return {"final": {}, "converged": False, "n_steps": 0, "history": {}}

    text = log_path.read_text()
    for line in text.split("\n"):
        for var in ["Ux", "Uy", "Uz", "p"]:
            pattern = rf"Solving for {var}.*Initial residual = ([\d.e+-]+)"
            m = re.search(pattern, line)
            if m:
                residuals[var].append(float(m.group(1)))

    final = {k: v[-1] for k, v in residuals.items() if v}
    converged = all(v < 1e-5 for v in final.values()) if final else False
    n_steps = max(len(v) for v in residuals.values()) if residuals else 0
    history = {k: v for k, v in residuals.items() if v}
    return {"final": final, "converged": converged, "n_steps": n_steps, "history": history}


def main():
    logger.info("=== Patient 58 CFD Post-Processing (foampilot native) ===")

    time_step = "500"
    log_path = CASE_DIR / "log.simpleFoam"

    # Read mesh + fields directly via Foampilot's OpenFOAMDirectReader
    reader = OpenFOAMDirectReader(case_path=CASE_DIR)
    logger.info(f"Patches: {list(reader.boundary_patches.keys())}")
    logger.info(f"Time steps: {reader.get_time_steps()}")

    # Load mesh with U and p fields attached as point data
    mesh = reader.to_pyvista(fields=["U", "p"], time_step=time_step, as_point_data=True)
    logger.info(f"Loaded mesh: {mesh.n_points} points, {mesh.n_cells} cells")

    # Compute U magnitude from the velocity vector
    if "U" in mesh.point_data:
        U = mesh.point_data["U"]
    elif "U" in mesh.cell_data:
        U = mesh.cell_data["U"]
    else:
        logger.error("Field 'U' not found in mesh")
        return 1

    if U.ndim == 1:
        U_mag_point = np.abs(U)
    else:
        U_mag_point = np.linalg.norm(U, axis=1)

    # Ensure we have U_mag as point data for rendering
    if "U" in mesh.point_data:
        mesh.point_data["U_mag"] = U_mag_point
    else:
        mesh.cell_data["U_mag"] = U_mag_point

    if "p" in mesh.point_data:
        p_data = mesh.point_data["p"]
    elif "p" in mesh.cell_data:
        p_data = mesh.cell_data["p"]
    else:
        p_data = np.zeros(mesh.n_points)

    p_range = [float(np.min(p_data)), float(np.max(p_data))]
    U_mag_range = [float(np.min(U_mag_point)), float(np.max(U_mag_point))]
    logger.info(f"Mesh bounds: {mesh.bounds}")
    logger.info(f"U_mag range: [{U_mag_range[0]:.4f}, {U_mag_range[1]:.4f}]")
    logger.info(f"p range: [{p_range[0]:.4f}, {p_range[1]:.4f}]")

    # Load centerline from file
    cl_path = CASE_DIR / "centerline.npy"
    cl = None
    if cl_path.exists():
        cl = np.load(cl_path)
        logger.info(f"Centerline: {len(cl)} points, start={cl[0]}, end={cl[-1]}")

    center = np.array(mesh.bounds).reshape(3, 2).mean(axis=1)

    # Check if the loaded centerline passes through the flow region.
    # If not, compute a corrected centerline through the high-velocity cells.
    # U_mag_point may be point or cell level; use cell centers for consistency.
    all_centers_arr = mesh.cell_centers().points
    if len(U_mag_point) == mesh.n_points:
        # Point data: sample at cell centers for centerline computation
        cell_U_mag_for_cl = np.zeros(mesh.n_cells)
        for i in range(mesh.n_cells):
            center = all_centers_arr[i]
            dists = np.sum((mesh.points - center) ** 2, axis=1)
            nearest = np.argmin(dists)
            cell_U_mag_for_cl[i] = U_mag_point[nearest]
        pts_for_cl = all_centers_arr
    else:
        cell_U_mag_for_cl = U_mag_point
        pts_for_cl = all_centers_arr

    nonzero_mask = cell_U_mag_for_cl > 1e-10
    if cl is not None and nonzero_mask.sum() > 10:
        # Check if nearest mesh point to centerline samples has significant velocity
        cl_end_in_flow = False
        for cl_pt in [cl[0], cl[-1], cl[len(cl)//2]]:
            dists = np.linalg.norm(pts_for_cl - cl_pt, axis=1)
            nearest = np.argmin(dists)
            if cell_U_mag_for_cl[nearest] > 1e-10 and dists[nearest] < 0.003:
                cl_end_in_flow = True
                break

        if not cl_end_in_flow:
            logger.info("Loaded centerline is outside flow region — computing corrected centerline...")
            flow_pts = pts_for_cl[nonzero_mask]
            flow_mean = flow_pts.mean(axis=0)
            flow_centered = flow_pts - flow_mean
            flow_cov = flow_centered.T @ flow_centered / (len(flow_pts) - 1)
            flow_eigvals, flow_eigvecs = np.linalg.eigh(flow_cov)
            flow_order = np.argsort(flow_eigvals)[::-1]
            principal_axis = flow_eigvecs[:, flow_order[0]]
            # Align the principal axis to start from the low-projection end
            projects = flow_centered @ principal_axis
            if projects.min() > -projects.max():
                principal_axis = -principal_axis
                projects = -projects
            n_cl = 20
            cl_proj = np.linspace(projects.min(), projects.max(), n_cl)
            cl = flow_mean + np.outer(cl_proj, principal_axis)
            logger.info(f"Corrected centerline: {len(cl)} points")
            logger.info(f"  Start: {cl[0]}")
            logger.info(f"  End: {cl[-1]}")
            logger.info(f"  Principal axis: {principal_axis}")

    # Build statistics
    boundary = reader.boundary_patches
    stats = {
        "case": str(CASE_DIR),
        "time_step": time_step,
        "mesh": {
            "n_cells": int(mesh.n_cells),
            "n_points": int(mesh.n_points),
            "n_boundary_faces": int(sum(b["nFaces"] for b in boundary.values())),
            "patches": list(boundary.keys()),
        },
        "fields": {
            "U": {
                "mean_magnitude": float(np.mean(U_mag_point)),
                "min_magnitude": float(np.min(U_mag_point)),
                "max_magnitude": float(np.max(U_mag_point)),
            },
            "p": {
                "min": float(np.min(p_data)),
                "max": float(np.max(p_data)),
                "mean": float(np.mean(p_data)),
            },
        },
    }

    res = parse_residuals(log_path)
    stats["convergence"] = res

    # --- Generate visualizations ---
    report_dir = CASE_DIR / "report"
    report_dir.mkdir(exist_ok=True)

    # Ensure cl_spline is always defined
    cl_spline = None
    if cl is not None:
        cl_spline = pv.Spline(cl, 200)

    # 1. Velocity magnitude 3D render
    p = pv.Plotter(off_screen=True)
    p.add_mesh(mesh, scalars="U_mag", cmap="viridis", lighting=False,
               clim=U_mag_range, scalar_bar_args={"title": "Velocity (m/s)"})
    if cl_spline is not None:
        p.add_mesh(cl_spline, color="red", line_width=4, opacity=0.8)
    p.reset_camera()
    p.screenshot(str(report_dir / "velocity_magnitude_3d.png"), window_size=[1600, 1000])
    logger.info("Saved: velocity_magnitude_3d.png")

    # 2. Velocity cross-sections via matplotlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_centers = mesh.cell_centers().points
    n_cells = len(all_centers)

    # Get cell data U_mag for cross-sections
    if "U_mag" in mesh.cell_data:
        cell_U_mag = mesh.cell_data["U_mag"]
    elif "U" in mesh.cell_data:
        cell_U = mesh.cell_data["U"]
        cell_U_mag = np.linalg.norm(cell_U, axis=1) if cell_U.ndim > 1 else np.abs(cell_U)
    else:
        cell_U_mag = U_mag_point[:n_cells] if len(U_mag_point) >= n_cells else np.full(n_cells, float(np.mean(U_mag_point)))

    if len(cell_U_mag) != n_cells:
        cell_U_mag = np.full(n_cells, float(np.mean(cell_U_mag)))

    cell_nonzero = cell_U_mag > 1e-10

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Velocity Cross-Sections (perpendicular to vessel)", fontsize=14, fontweight="bold")

    # Determine slices along the centerline direction
    # (the vessel is not aligned with any coordinate axis, so we slice
    #  perpendicular to the principal flow direction)
    if cl is not None:
        cl_dir = cl[-1] - cl[0]
        cl_dir = cl_dir / np.linalg.norm(cl_dir)
        cl_start = cl[0]
        cl_end = cl[-1]
        cl_length = np.linalg.norm(cl_end - cl_start)
        # Project cell centers onto centerline direction
        proj = (all_centers - cl_start) @ cl_dir
        # Find cells within the vessel region (projected between start and end)
        vessel_mask = (proj > -0.005) & (proj < cl_length + 0.005) & cell_nonzero
    else:
        # Fallback: use X-axis
        cl_dir = np.array([1, 0, 0])
        proj = all_centers[:, 0]
        vessel_mask = cell_nonzero

    proj_min, proj_max = float(proj[vessel_mask].min()), float(proj[vessel_mask].max())

    for ax_i, (frac, label) in enumerate([(0.25, "Inlet (25%)"), (0.50, "Mid (50%)"), (0.75, "Outlet (75%)")]):
        proj_slice = proj_min + frac * (proj_max - proj_min)
        if cl is not None:
            cl_pt = cl_start + proj_slice * cl_dir
            # Perpendicular distance to centerline axis
            rel = all_centers - cl_pt
            proj_dist = rel @ cl_dir
            perp_dist = np.linalg.norm(rel - np.outer(proj_dist, cl_dir), axis=1)
            mask = (np.abs(proj_dist) < 0.003) & (perp_dist < 0.003)
            if len(cell_U_mag) == len(all_centers):
                mask = mask & (cell_U_mag > 1e-10)
        else:
            mask = np.abs(proj - proj_slice) < 0.003
            if len(cell_U_mag) == len(all_centers):
                mask = mask & (cell_U_mag > 1e-10)

        y_vals = all_centers[mask, 1]
        z_vals = all_centers[mask, 2]
        u_vals = cell_U_mag[mask] if len(cell_U_mag) == len(all_centers) else np.full(mask.sum(), float(np.mean(cell_U_mag)))

        ax = axes[ax_i]
        if len(y_vals) < 3:
            ax.text(0.5, 0.5, "No cells\nat this section", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{label}\n(proj={proj_slice:.4f} m)")
            ax.set_aspect("equal")
            continue

        tc = ax.tricontourf(y_vals, z_vals, u_vals, levels=15, cmap="viridis")
        fig.colorbar(tc, ax=ax, label="Velocity (m/s)")
        ax.set_title(f"{label} (proj={proj_slice:.4f} m, {len(y_vals)} cells)")
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("Z (m)")
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(str(report_dir / "velocity_slices.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: velocity_slices.png")

    # 3. Pressure 3D render
    p = pv.Plotter(off_screen=True)
    p.set_background("white")
    if p_range[0] == p_range[1]:
        logger.info(f"  Pressure is uniform ({p_range[0]:.4f} Pa) - rendering as solid color")
        p.add_mesh(mesh, color="lightgray", lighting=False,
                   scalar_bar_args={"title": f"Pressure (Pa) = {p_range[0]:.4f}"})
    else:
        p.add_mesh(mesh, scalars="p", cmap="coolwarm", lighting=False,
                   clim=p_range, scalar_bar_args={"title": "Pressure (Pa)"})
    if cl is not None:
        p.add_mesh(cl_spline, color="white", line_width=4)
    p.reset_camera()
    p.screenshot(str(report_dir / "pressure_3d.png"), window_size=[1600, 1000])
    logger.info("Saved: pressure_3d.png")

    # Find the cross-section perpendicular to the centerline with highest average velocity
    cell_pts = mesh.cell_centers().points
    n_cells = len(cell_pts)

    if "U" in mesh.cell_data and len(mesh.cell_data["U"]) == mesh.n_cells:
        u_vec_cells = mesh.cell_data["U"]
        u_mag_cells = np.linalg.norm(u_vec_cells, axis=1) if u_vec_cells.ndim > 1 else np.abs(u_vec_cells)
    else:
        # Interpolate point data U to cell centers
        if "U" in mesh.point_data:
            u_vec_pt = mesh.point_data["U"]
        else:
            u_vec_pt = U_mag_point[:, np.newaxis] if U_mag_point.ndim == 1 else U_mag_point

        u_mag_cells = np.zeros(n_cells)
        u_vec_cells = np.zeros((n_cells, 3))
        for i in range(n_cells):
            dists = np.sum((mesh.points - cell_pts[i]) ** 2, axis=1)
            nearest = np.argmin(dists)
            if u_vec_pt.ndim > 1 and u_vec_pt.shape[1] == 3:
                u_vec_cells[i] = u_vec_pt[nearest]
                u_mag_cells[i] = np.linalg.norm(u_vec_pt[nearest])
            else:
                u_mag_cells[i] = abs(u_vec_pt[nearest])

    # Project onto centerline direction for slice selection
    if cl is not None:
        cl_dir_g = cl[-1] - cl[0]
        cl_dir_g = cl_dir_g / np.linalg.norm(cl_dir_g)
        cl_start_g = cl[0]
        proj_g = (cell_pts - cl_start_g) @ cl_dir_g
    else:
        cl_dir_g = np.array([1, 0, 0])
        proj_g = cell_pts[:, 0]

    proj_min, proj_max = float(proj_g[u_mag_cells > 1e-10].min()) if (u_mag_cells > 1e-10).any() else float(proj_g.min()), float(proj_g[u_mag_cells > 1e-10].max()) if (u_mag_cells > 1e-10).any() else float(proj_g.max())
    slice_bins = np.linspace(proj_min, proj_max, 20)
    best_proj = None
    best_score = 0
    for i in range(len(slice_bins) - 1):
        mask_s = (proj_g >= slice_bins[i]) & (proj_g < slice_bins[i + 1]) & (u_mag_cells > 1e-10)
        if mask_s.sum() > 10:
            avg_u = np.mean(u_mag_cells[mask_s])
            if avg_u > best_score:
                best_score = avg_u
                best_proj = (slice_bins[i] + slice_bins[i + 1]) / 2

    if best_proj is not None and best_score > 1e-6:
        mask_cells = np.abs(proj_g - best_proj) < (slice_bins[1] - slice_bins[0]) / 2 * 1.5
        if mask_cells.sum() > 5:
            idx = np.where(mask_cells)[0]
            step = max(1, len(idx) // 30)
            idx = idx[::step]
            pts = cell_pts[idx]

            if u_vec_cells is not None and len(u_vec_cells) == mesh.n_cells:
                u_vec = u_vec_cells[idx]
                u_mag_s = u_mag_cells[idx]
            elif u_vec_cells is not None and len(u_vec_cells) == mesh.n_points:
                u_vec = u_vec_cells[:mesh.n_cells][idx]
                u_mag_s = u_mag_cells[:mesh.n_cells][idx]
            else:
                u_vec = np.zeros((len(idx), 3))
                u_mag_s = np.zeros(len(idx))

            if u_vec.ndim == 1:
                u_vec = u_vec.reshape(-1, 3)
            if u_vec.shape[1] != 3:
                u_vec = np.zeros((len(idx), 3))

            valid = u_mag_s > 1e-10
            if valid.sum() > 3:
                pts_valid = pts[valid]
                u_vec_valid = u_vec[valid]
                u_mag_valid = u_mag_s[valid]
                u_vec_normed = u_vec_valid / (u_mag_valid[:, np.newaxis] + 1e-15)
                # Scale glyphs to fit within the vessel — use 0.5% of domain size per m/s
                domain_size = float(max(mesh.bounds[1::2]))
                glyph_scale = min(0.0001, domain_size * 0.02 / max(u_mag_valid.max(), 1e-6))
                scaled_vec = u_vec_normed * (u_mag_valid[:, np.newaxis] * glyph_scale)

                p = pv.Plotter(off_screen=True)
                p.set_background("white")
                p.add_mesh(mesh, color="lightgray", style="wireframe", line_width=0.3, opacity=0.3)
                for i in range(len(pts_valid)):
                    p.add_mesh(pv.Line(pts_valid[i], pts_valid[i] + scaled_vec[i]), color="red", line_width=2)
                if cl is not None:
                    p.add_mesh(cl_spline, color="blue", line_width=2)
                p.reset_camera()
                p.screenshot(str(report_dir / "velocity_glyphs.png"), window_size=[1600, 1000])
                logger.info("Saved: velocity_glyphs.png")
            else:
                logger.info("Skipped velocity_glyphs.png (all velocities zero at slice)")
        else:
            logger.info("Skipped velocity_glyphs.png (no cells at Z slice)")
    else:
        logger.info("Skipped velocity_glyphs.png (no non-zero velocity region found)")

    # 5. Wall velocity render
    p = pv.Plotter(off_screen=True)
    p.add_mesh(mesh, scalars="U_mag", cmap="plasma", lighting=False,
               clim=U_mag_range, scalar_bar_args={"title": "Velocity (m/s)"})
    p.add_mesh(mesh, style="wireframe", color="black", line_width=0.5, opacity=0.3)
    if cl is not None:
        p.add_mesh(cl_spline, color="cyan", line_width=3)
    p.reset_camera()
    p.screenshot(str(report_dir / "wall_velocity.png"), window_size=[1600, 1000])
    logger.info("Saved: wall_velocity.png")

    # 6. Multi-view patch visualization (4 views, colored boundaries)
    # Use actual boundary face data from the OpenFOAM boundary file for correct patch IDs
    reader._ensure_mesh_loaded()
    bf_points = reader._points
    bf_faces = reader._faces
    bf_boundary = reader._boundary

    patch_map = {"INLET": 1, "OUTLET": 2, "WALL": 3}

    # Compute vessel axis direction from centerline (or flow region)
    if cl is not None:
        cl_dir = cl[-1] - cl[0]
        cl_dir = cl_dir / np.linalg.norm(cl_dir)
    elif nonzero_mask.sum() > 10:
        flow_pts = pts_for_cl[nonzero_mask]
        flow_mean = flow_pts.mean(axis=0)
        flow_centered = flow_pts - flow_mean
        flow_cov = flow_centered.T @ flow_centered / (len(flow_pts) - 1)
        _, flow_eigvecs = np.linalg.eigh(flow_cov)
        cl_dir = flow_eigvecs[:, -1]
        cl_dir = cl_dir / np.linalg.norm(cl_dir)
    else:
        cl_dir = np.array([1, 0, 0])

    # Build boundary surface and classify patches geometrically:
    # - Axial faces (normal within 10° of vessel axis) → INLET/OUTLET (split by dot direction)
    # - Non-axial faces (normal > 80° from vessel axis) → WALL
    # - Use OpenFOAM patch name when available, but refine with geometry
    poly_face_list = []
    poly_patch_ids = []
    patch_counts = {"INLET": 0, "OUTLET": 0, "WALL": 0, "OTHER": 0}

    for pname, pinfo in bf_boundary.items():
        start = pinfo["startFace"]
        nfaces = pinfo["nFaces"]
        base_pid = patch_map.get(pname, 0)
        for fi in range(start, start + nfaces):
            face_verts = bf_faces[fi]
            # Compute face normal geometrically
            fv = np.array(bf_points[face_verts])
            if len(fv) >= 3:
                n1 = np.cross(fv[1] - fv[0], fv[2] - fv[0])
                n1_norm = np.linalg.norm(n1)
                if n1_norm > 1e-12:
                    face_normal = n1 / n1_norm
                else:
                    face_normal = np.array([0, 0, 1])
            else:
                face_normal = np.array([0, 0, 1])

            # Angle between face normal and vessel axis
            cos_angle = abs(np.dot(face_normal, cl_dir))
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, 0, 1)))

            # Classify: use OpenFOAM patch name; WALL faces with angle > 80°
            # from axis are confirmed as wall; faces < 10° are inlet/outlet-like
            if pname == "INLET" or (pname == "WALL" and angle_deg < 10 and np.dot(face_normal, cl_dir) < 0):
                pid = 1
                patch_counts["INLET"] += 1
            elif pname == "OUTLET" or (pname == "WALL" and angle_deg < 10 and np.dot(face_normal, cl_dir) >= 0):
                pid = 2
                patch_counts["OUTLET"] += 1
            else:
                pid = 3
                patch_counts["WALL"] += 1

            poly_face_list.append(len(face_verts))
            poly_face_list.extend(face_verts.tolist())
            poly_patch_ids.append(pid)

    surf = pv.PolyData(bf_points, np.array(poly_face_list, dtype=np.int64))
    surf.cell_data["patch_id"] = np.array(poly_patch_ids, dtype=int)
    logger.info(f"Boundary surface: {surf.n_cells} faces")
    logger.info(f"  Patch classification: {patch_counts}")

    surf_center = surf.points.mean(axis=0)

    # Build 4 views: axial (along vessel), and three orthogonal cross-sections
    # rotated so the vessel axis is prominent
    up_vec = np.array([0.0, 0.0, 1.0])

    # Avoid degenerate cross product when cl_dir is parallel to up
    if abs(np.dot(cl_dir, up_vec)) > 0.95:
        up_vec = np.array([0.0, 1.0, 0.0])

    # Camera positions aligned with vessel angle
    # View 1: Inlet (looking towards outlet along vessel axis)
    cam_inlet = (surf_center - cl_dir * 0.5, surf_center, up_vec)
    # View 2: Outlet (opposite direction)
    cam_outlet = (surf_center + cl_dir * 0.5, surf_center, up_vec)
    # View 3: Axial cross-section (perpendicular to vessel axis)
    perp = np.cross(cl_dir, up_vec)
    perp = perp / np.linalg.norm(perp)
    cam_axial = (surf_center + perp * 0.3, surf_center, cl_dir)
    # View 4: Isometric
    cam_iso = "iso"

    views = [("Inlet", cam_inlet), ("Outlet", cam_outlet), ("Axial", cam_axial), ("Iso", cam_iso)]
    p = pv.Plotter(off_screen=True, shape=(2, 2))
    p.set_background("white")

    for view_i, (view_name, cam_pos) in enumerate(views):
        p.subplot(view_i // 2, view_i % 2)
        p.add_mesh(
            surf,
            scalars="patch_id",
            lighting=False,
            cmap=["red", "blue", "green", "orange"],
            clim=[0.5, 3.5],
            scalar_bar_args={"title": "Patches (Red=Inlet, Blue=Outlet, Green=Wall)", "height": 0.4},
            show_scalar_bar=(view_i == 3),
            show_edges=False,
            opacity=0.8,
        )
        if cl is not None:
            p.add_mesh(cl_spline, color="yellow", line_width=3, opacity=0.7)
        p.reset_camera()
        if cam_pos != "iso":
            p.camera_position = cam_pos
        p.add_text(f"{view_name} view", font_size=8, color="black")

    p.screenshot(str(report_dir / "patch_views_4panel.png"), window_size=[1600, 1000])
    logger.info("Saved: patch_views_4panel.png")
    p.close()

    # --- Export data ---
    for patch_name, patch_info in boundary.items():
        logger.info(f"  Patch {patch_name}: {patch_info.get('nFaces', 0)} faces")

    stats_path = report_dir / "postprocess_statistics.json"
    stats_path.write_text(json.dumps(stats, indent=2, cls=NumpyEncoder))
    logger.info("Saved: postprocess_statistics.json")

    # Convergence plot
    history = res.get("history", {})
    if history:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Convergence History (simpleFoam)", fontsize=14, fontweight="bold")
        for ax, var in zip(axes.flat, ["Ux", "Uy", "Uz", "p"]):
            residual_history = history.get(var, [])
            if residual_history:
                ax.semilogy(residual_history)
                ax.set_title(f"{var} residual")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Initial residual")
                ax.axhline(y=1e-5, color="r", linestyle="--", alpha=0.5)
                ax.grid(True, alpha=0.3)
            else:
                ax.set_title(f"{var} - no data")
        plt.tight_layout()
        plt.savefig(str(report_dir / "convergence_history.png"), dpi=150)
        plt.close()
        logger.info("Saved: convergence_history.png")

    # CSV export
    if "U" in mesh.point_data:
        U_pt = mesh.point_data["U"]
    elif "U" in mesh.cell_data:
        U_pt = mesh.cell_data["U"]
    else:
        U_pt = np.zeros((mesh.n_points, 3))

    if U_pt.ndim == 1:
        U_pt = np.column_stack([U_pt, np.zeros_like(U_pt), np.zeros_like(U_pt)])

    if "p" in mesh.point_data:
        p_pt = mesh.point_data["p"]
    elif "p" in mesh.cell_data:
        p_pt = mesh.cell_data["p"]
    else:
        p_pt = np.zeros(mesh.n_points)

    p_pt = np.full(mesh.n_points, float(np.mean(p_pt))) if len(p_pt) != mesh.n_points else p_pt

    df_data = {
        "X": mesh.points[:, 0],
        "Y": mesh.points[:, 1],
        "Z": mesh.points[:, 2],
        "U_x": U_pt[:, 0] if len(U_pt) == mesh.n_points else np.zeros(mesh.n_points),
        "U_y": U_pt[:, 1] if len(U_pt) == mesh.n_points else np.zeros(mesh.n_points),
        "U_z": U_pt[:, 2] if len(U_pt) == mesh.n_points else np.zeros(mesh.n_points),
        "U_mag": U_mag_point if len(U_mag_point) == mesh.n_points else np.full(mesh.n_points, float(np.mean(U_mag_point))),
        "p": p_pt,
    }
    df = pd.DataFrame(df_data)
    csv_path = report_dir / "field_data_500.csv"
    df.to_csv(str(csv_path), index=False)
    logger.info(f"Saved: field_data_500.csv ({len(df)} rows)")

    # 6. Velocity profile along centerline (point-by-point sampling)
    # Use the corrected centerline (cl) rather than recomputing PCA
    if cl is not None:
        cl_for_profile = cl
    else:
        # Fallback: compute centerline from flow region
        if "U" in mesh.point_data:
            U_pt = mesh.point_data["U"]
            U_mag_pt = np.linalg.norm(U_pt, axis=1) if U_pt.ndim > 1 else np.abs(U_pt)
            mesh_pts = mesh.points
            nonzero = U_mag_pt > 1e-10
            if nonzero.sum() > 10:
                flow_pts = mesh_pts[nonzero]
                mean = flow_pts.mean(axis=0)
                centered = flow_pts - mean
                cov = centered.T @ centered / (len(flow_pts) - 1)
                eigvals, eigvecs = np.linalg.eigh(cov)
                order = np.argsort(eigvals)[::-1]
                principal_axis = eigvecs[:, order[0]]
                projects = centered @ principal_axis
                n_pts = 40
                cl_pts_scalar = np.linspace(projects.min(), projects.max(), n_pts)
                cl_for_profile = mean + np.outer(cl_pts_scalar, principal_axis)
            else:
                cl_for_profile = None
        else:
            cl_for_profile = None

    if cl_for_profile is not None:
        # Build KD-tree on the appropriate mesh entities
        from scipy.spatial import cKDTree
        if "U" in mesh.point_data:
            sample_pts = mesh.points
            sample_vals = np.linalg.norm(mesh.point_data["U"], axis=1) if mesh.point_data["U"].ndim > 1 else np.abs(mesh.point_data["U"])
        else:
            # Use cell centers for cell data
            sample_pts = mesh.cell_centers().points
            sample_vals = np.linalg.norm(mesh.cell_data["U"], axis=1) if mesh.cell_data["U"].ndim > 1 else np.abs(mesh.cell_data["U"])

        tree = cKDTree(sample_pts)

        # Compute cumulative distance along centerline
        cl_distances = np.zeros(len(cl_for_profile))
        cumulative = np.cumsum(np.linalg.norm(np.diff(cl_for_profile, axis=0), axis=1))
        cl_distances[1:] = cumulative
        cl_distances_mm = cl_distances * 1000

        # Sample velocity at each centerline point
        dists, nearest = tree.query(cl_for_profile, k=1, distance_upper_bound=0.005)
        valid = dists < 0.005
        nearest = np.clip(nearest, 0, len(sample_vals) - 1)
        sample_ums = np.where(valid, sample_vals[nearest], np.nan)

        # Filter out NaN samples (points too far from any mesh element)
        valid_ums = sample_ums[valid]
        valid_dists = cl_distances_mm[valid]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(valid_dists, valid_ums, 'b-o', linewidth=2, markersize=5, label="U_mag")
        ax.axhline(y=U_mag_range[1], color='r', linestyle='--', alpha=0.5, label=f"Max ({U_mag_range[1]:.2f} m/s)")
        ax.set_xlabel("Distance along centerline (mm)")
        ax.set_ylabel("Velocity magnitude (m/s)")
        ax.set_title("Velocity Profile Along Centerline")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(report_dir / "velocity_profile.png"), dpi=150)
        plt.close()
        logger.info(f"Saved: velocity_profile.png ({len(valid_ums)} valid samples)")

    logger.info("\n=== Post-processing complete ===")
    logger.info(f"Report dir: {report_dir}")
    logger.info(f"Convergence: {stats.get('convergence', {}).get('converged', 'N/A')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())