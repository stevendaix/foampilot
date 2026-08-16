#!/usr/bin/env python3
"""
Diagnostic: phase, ordre et sections manquantes dans le loft.
"""

import sys
from pathlib import Path

import numpy as np
import trimesh
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "foampilot" / "src"))

from foampilot.geometry.topology import TopologySectionExtractor
from foampilot.geometry.topology.section_extractor import _process_section_polylines

BASE_DIR = Path(__file__).resolve().parent
STL_PATH = BASE_DIR.parent / "patient58_cfd_example" / "constant" / "triSurface" / "tbad_TL_walls.stl"
CENTERLINE_PATH = BASE_DIR.parent / "patient58_cfd_example" / "centerline.npy"


def build_local_frame(normal):
    n = normal / np.linalg.norm(normal)
    if abs(n[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def analyze_sections():
    mesh = trimesh.load(str(STL_PATH), process=True)
    centerline = np.load(str(CENTERLINE_PATH))

    axis = centerline[-1] - centerline[0]
    axis = axis / np.linalg.norm(axis)

    # --- Extraction “centerline-based” comme dans step3c ---
    sections = []
    step = max(1, len(centerline) // 10)
    used_indices = []
    for i in range(0, len(centerline), step):
        center = centerline[i]
        try:
            section = mesh.section(plane_origin=center, plane_normal=axis)
            points = _process_section_polylines(section, axis, center, n_resample=64)
            if points is not None:
                sections.append(points)
                used_indices.append(i)
        except Exception:
            pass

    print(f"Nombre de sections extraites: {len(sections)}")
    print(f"Indices centerline utilisés: {used_indices}")

    u, v = build_local_frame(axis)

    # Vérifier la phase / continuité entre sections consécutives
    print("\n=== Vérification phase / continuité ===")
    for i in range(len(sections) - 1):
        pts0 = sections[i]
        pts1 = sections[i + 1]

        c0 = pts0.mean(axis=0)
        c1 = pts1.mean(axis=0)

        # Coordonnées 2D dans le repère local
        pts0_2d = np.column_stack(((pts0 - c0) @ u, (pts0 - c0) @ v))
        pts1_2d = np.column_stack(((pts1 - c1) @ u, (pts1 - c1) @ v))

        # Point le plus proche du centre en 2D
        d0 = np.linalg.norm(pts0_2d, axis=1)
        d1 = np.linalg.norm(pts1_2d, axis=1)
        idx0 = int(np.argmin(d0))
        idx1 = int(np.argmin(d1))

        # Écart angulaire entre les directions “vers le centre”
        dir0 = -pts0_2d[idx0]
        dir1 = -pts1_2d[idx1]
        if np.linalg.norm(dir0) > 1e-12 and np.linalg.norm(dir1) > 1e-12:
            dir0 = dir0 / np.linalg.norm(dir0)
            dir1 = dir1 / np.linalg.norm(dir1)
            dot = np.clip(np.dot(dir0, dir1), -1.0, 1.0)
            angle = np.degrees(np.arccos(dot))
        else:
            angle = np.nan

        # Décalage cyclique probable
        shift = (idx1 - idx0) % len(pts0_2d)

        print(
            f"  Section {i} -> {i+1}: "
            f"shift={shift:3d}, "
            f"angle_centre={angle:6.1f}°, "
            f"n_pts={len(pts0_2d)}"
        )

    # Vérifier les sauts de rayon / aire
    print("\n=== Rayons / aires ===")
    radii = []
    areas = []
    for i, pts in enumerate(sections):
        c = pts.mean(axis=0)
        d = np.linalg.norm(pts - c, axis=1)
        r = np.mean(d)
        a = np.pi * r ** 2
        radii.append(r)
        areas.append(a)
        print(f"  Section {i}: r={r*1000:.2f} mm, area={a*1e6:.2f} mm²")

    radii = np.array(radii)
    areas = np.array(areas)

    if len(radii) > 1:
        dr = np.abs(np.diff(radii))
        da = np.abs(np.diff(areas))
        print(f"\nSaut de rayon max: {dr.max()*1000:.2f} mm")
        print(f"Saut d'aire max: {da.max()*1e6:.2f} mm²")

    # Intégration de volume
    volume = 0.0
    for i in range(len(areas) - 1):
        centers = np.array([s.mean(axis=0) for s in sections])
        ds = np.linalg.norm(centers[i + 1] - centers[i])
        volume += (areas[i] + areas[i + 1]) / 2.0 * ds
    print(f"\nVolume intégré (sections): {volume:.6e} m³")
    print(f"Volume loft OCC:        6.50e-05 m³")
    print(f"Volume sweep:           5.14e-06 m³")

    # Plot diagnostic
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Sections dans le plan local
    ax = axes[0, 0]
    for i, pts in enumerate(sections):
        c = pts.mean(axis=0)
        pts2d = np.column_stack(((pts - c) @ u, (pts - c) @ v))
        ax.plot(pts2d[:, 0] * 1000, pts2d[:, 1] * 1000, label=f"Sec {i}")
    ax.set_xlabel("X local (mm)")
    ax.set_ylabel("Y local (mm)")
    ax.set_title("Sections dans le repère local")
    ax.legend()
    ax.set_aspect("equal")

    # 2. Rayons
    ax = axes[0, 1]
    ax.plot(radii * 1000, "o-")
    ax.set_xlabel("Index section")
    ax.set_ylabel("Rayon (mm)")
    ax.set_title("Rayon par section")
    ax.grid(True)

    # 3. Aires
    ax = axes[1, 0]
    ax.plot(areas * 1e6, "s-", color="green")
    ax.set_xlabel("Index section")
    ax.set_ylabel("Aire (mm²)")
    ax.set_title("Aire par section")
    ax.grid(True)

    # 4. Comparaison volumes
    ax = axes[1, 1]
    vols = [volume * 1e6, 5.144965e-06 * 1e6, 6.499805e-05 * 1e6]
    labels = ["Intégré", "Sweep", "Loft OCC"]
    colors = ["steelblue", "coral", "forestgreen"]
    ax.bar(labels, vols, color=colors, alpha=0.7)
    ax.set_ylabel("Volume (cm³)")
    ax.set_title("Comparaison des volumes")
    ax.grid(True, axis="y")
    for bar, v in zip(ax.patches, vols):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    out = BASE_DIR / "diagnostic_phase_volume.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {out}")


if __name__ == "__main__":
    analyze_sections()
