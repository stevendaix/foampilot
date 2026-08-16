#!/usr/bin/env python3
"""
Exploration de reconstruction aortique avec build123d.

Approche :
1. Charger la centerline patient58 (76 points, 2mm spacing)
2. Créer un solide par sweep d'un profil circulaire le long de la centerline
3. Export STEP pour inspection

Usage:
    python3 explore_build123d_aorta.py
"""

import sys
from pathlib import Path

import numpy as np

import build123d as b123d

# Paths
CASE_DIR = Path(__file__).resolve().parent.parent / "patient58_cfd_example"
CENTERLINE_PATH = CASE_DIR / "centerline.npy"
OUTPUT_DIR = Path(__file__).resolve().parent


def load_centerline(path: Path) -> np.ndarray:
    """Load centerline points in meters."""
    pts = np.load(str(path))
    if pts.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) centerline, got {pts.shape}")
    return pts


def build_aorta_sweep(centerline: np.ndarray, radius_mm: float = 3.0) -> b123d.Part | None:
    """Build aorta by sweeping a circular profile along the centerline.

    Args:
        centerline: (N, 3) array of centerline points in meters.
        radius_mm: Pipe radius in millimeters.

    Returns:
        build123d Part object, or None if failed.
    """
    radius = radius_mm / 1000.0  # meters

    # Build path from centerline points
    with b123d.BuildLine() as line:
        b123d.Polyline([tuple(p) for p in centerline])
    path = line.wire()

    # Create circular profile
    with b123d.BuildSketch() as sketch:
        b123d.Circle(radius)
    face = sketch.faces()[0]

    try:
        with b123d.BuildPart() as part:
            b123d.sweep(face, path)
        return part.part
    except Exception as exc:
        print(f"  Sweep failed: {exc}")
        return None


def build_aorta_loft(centerline: np.ndarray, radius_mm: float = 3.0, step: int = 5) -> b123d.Part | None:
    """Build aorta by lofting circular profiles along the centerline.

    Each profile is oriented perpendicular to the local centerline tangent.

    Args:
        centerline: (N, 3) array of centerline points in meters.
        radius_mm: Profile radius in millimeters.
        step: Sample every `step` points along centerline.

    Returns:
        build123d Part object, or None if failed.
    """
    radius = radius_mm / 1000.0
    faces = []

    for i in range(0, len(centerline), step):
        p = centerline[i]
        if i < len(centerline) - 1:
            tangent = centerline[i + 1] - p
        else:
            tangent = p - centerline[i - 1]
        tangent = tangent / (np.linalg.norm(tangent) + 1e-12)

        # Build local frame
        if abs(tangent[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        u = np.cross(tangent, ref)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(tangent, u)

        plane = b123d.Plane(
            origin=b123d.Vector(tuple(p)),
            x_dir=b123d.Vector(tuple(u)),
            z_dir=b123d.Vector(tuple(-tangent)),
        )

        with b123d.BuildSketch(plane) as sketch:
            b123d.Circle(radius)
        faces.append(sketch.faces()[0])

    if len(faces) < 2:
        print("  Not enough faces for loft")
        return None

    try:
        with b123d.BuildPart() as part:
            b123d.loft(faces, ruled=False)
        return part.part
    except Exception as exc:
        print(f"  Loft failed: {exc}")
        return None


def build_aorta_revolve(centerline: np.ndarray, radius_mm: float = 3.0) -> b123d.Part | None:
    """Build aorta by revolving a circle around the centerline axis.

    This is a rough approximation: revolve a circle around the principal axis.

    Args:
        centerline: (N, 3) array of centerline points in meters.
        radius_mm: Profile radius in millimeters.

    Returns:
        build123d Part object, or None if failed.
    """
    radius = radius_mm / 1000.0
    length = float(np.linalg.norm(centerline[-1] - centerline[0]))

    try:
        with b123d.BuildPart() as part:
            with b123d.BuildSketch() as sketch:
                b123d.Circle(radius)
            b123d.revolve(axis=b123d.Axis.Z, revolution_arc=360)
        return part.part
    except Exception as exc:
        print(f"  Revolve failed: {exc}")
        return None


def main() -> int:
    print("=== Exploration build123d : reconstruction aortique ===")

    if not CENTERLINE_PATH.exists():
        print(f"Centerline not found: {CENTERLINE_PATH}")
        return 1

    centerline = load_centerline(CENTERLINE_PATH)
    print(f"Centerline chargée: {len(centerline)} points")
    print(f"  Inlet: {centerline[0]}")
    print(f"  Outlet: {centerline[-1]}")
    print(f"  Longueur: {np.linalg.norm(centerline[-1] - centerline[0])*1000:.1f} mm")

    # Approach 1: Sweep along centerline
    print("\n--- Approche 1: Sweep along centerline ---")
    part1 = build_aorta_sweep(centerline, radius_mm=3.0)
    if part1 is not None:
        print(f"  Sweep OK: volume={part1.volume:.6f} m³, area={part1.area:.6f} m²")
        b123d.export_step(part1, str(OUTPUT_DIR / "aorta_sweep.step"))
    else:
        print("  Sweep échoué")

    # Approach 2: Loft with oriented profiles
    print("\n--- Approche 2: Loft avec profiles orientés ---")
    part2 = build_aorta_loft(centerline, radius_mm=3.0, step=4)
    if part2 is not None:
        print(f"  Loft OK: volume={part2.volume:.6f} m³, area={part2.area:.6f} m²")
        b123d.export_step(part2, str(OUTPUT_DIR / "aorta_loft.step"))
    else:
        print("  Loft échoué")

    # Approach 3: Revolve (simplified)
    print("\n--- Approche 3: Revolve simple ---")
    part3 = build_aorta_revolve(centerline, radius_mm=3.0)
    if part3 is not None:
        print(f"  Revolve OK: volume={part3.volume:.6f} m³")
        b123d.export_step(part3, str(OUTPUT_DIR / "aorta_revolve.step"))
    else:
        print("  Revolve échoué")

    print(f"\nFichiers STEP dans: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
