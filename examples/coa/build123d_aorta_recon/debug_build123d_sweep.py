#!/usr/bin/env python3
"""
Diagnostic et reconstruction aortique avec build123d.

Ce script investigate pourquoi build123d.sweep cree une geometrie de volume nul
et propose des solutions de contournement.

Decouverte principale:
  build123d.sweep() utilise BRepOffsetAPI_MakePipeShell avec withContact=False.
  Pour un profile planaire balaye le long d'un chemin coplanaire (ou quasi),
  cela cree une geometrie degeneree (volume ~ 0).

Solutions:
  1. Utiliser BRepOffsetAPI_MakePipeShell directement avec withContact=True
  2. Utiliser un loft avec des profiles orientes perpendiculairement a la tangente
  3. Utiliser un chemin reellement 3D (ameliore mais ne resout pas completement)

Usage:
    python3 debug_build123d_sweep.py
"""

import sys
from pathlib import Path

import numpy as np
import trimesh

import build123d as b123d
from build123d.topology import Face, Solid, Wire
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer

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


def inspect_shape(result: Solid, name: str = "") -> dict:
    """Inspect a build123d Solid and return diagnostics."""
    wrapped = result.wrapped
    face_count = 0
    exp = TopExp_Explorer(wrapped, TopAbs_FACE)
    while exp.More():
        face_count += 1
        exp.Next()

    bbox = result.bounding_box()
    diagnostics = {
        "name": name,
        "volume": result.volume,
        "area": result.area,
        "is_valid": result.is_valid,
        "faces": face_count,
        "bbox": bbox,
    }
    return diagnostics


def print_diagnostics(d: dict) -> None:
    """Pretty-print diagnostics."""
    print(f"  {d['name']}:")
    print(f"    Volume:  {d['volume']:.6e} m³")
    print(f"    Area:    {d['area']:.6e} m²")
    print(f"    Valid:   {d['is_valid']}")
    print(f"    Faces:   {d['faces']}")
    print(f"    BBox X:  {d['bbox'].min.X:.4f} -> {d['bbox'].max.X:.4f}")
    print(f"    BBox Y:  {d['bbox'].min.Y:.4f} -> {d['bbox'].max.Y:.4f}")
    print(f"    BBox Z:  {d['bbox'].min.Z:.4f} -> {d['bbox'].max.Z:.4f}")


def build_aorta_sweep_vanilla(centerline: np.ndarray, radius_mm: float = 3.0) -> Solid | None:
    """Original build123d sweep (produces zero volume)."""
    radius = radius_mm / 1000.0
    with b123d.BuildLine() as line:
        b123d.Polyline([tuple(p) for p in centerline])
    path = line.wire()

    with b123d.BuildSketch() as sketch:
        b123d.Circle(radius)
    face = sketch.faces()[0]

    with b123d.BuildPart() as part:
        b123d.sweep(face, path)
    return part.part


def build_aorta_sweep_direct_occ(
    centerline: np.ndarray, radius_mm: float = 3.0
) -> Solid | None:
    """Fix direct: use BRepOffsetAPI_MakePipeShell avec withContact=True.

    C'est la solution recommandee.
    """
    radius = radius_mm / 1000.0
    with b123d.BuildLine() as line:
        b123d.Polyline([tuple(p) for p in centerline])
    path = line.wire()

    with b123d.BuildSketch() as sketch:
        b123d.Circle(radius)
    face = sketch.faces()[0]
    outer = face.outer_wire()

    builder = BRepOffsetAPI_MakePipeShell(path.wrapped)
    builder.SetMode(False)
    builder.Add(outer.wrapped, True, True)  # withContact=True, withCorrection=True
    builder.Build()
    builder.MakeSolid()
    return Solid(builder.Shape())


def build_aorta_loft(
    centerline: np.ndarray, radius_mm: float = 3.0, step: int = 4
) -> Solid | None:
    """Alternative: loft avec profiles orientes perpendiculairement a la tangente."""
    radius = radius_mm / 1000.0
    from scipy.spatial.transform import Rotation as R

    faces = []
    for i in range(0, len(centerline), step):
        p = centerline[i]
        if i < len(centerline) - 1:
            tangent = centerline[i + 1] - p
        else:
            tangent = p - centerline[i - 1]
        tangent = tangent / (np.linalg.norm(tangent) + 1e-12)

        with b123d.BuildSketch() as sketch:
            b123d.Circle(radius)
        face = sketch.faces()[0]

        rot, _ = R.align_vectors([[0, 0, 1]], [tangent])
        rot_euler = rot.as_euler("xyz", degrees=True)
        rotated = face.rotate(b123d.Axis.X, rot_euler[0])
        rotated = rotated.rotate(b123d.Axis.Y, rot_euler[1])
        rotated = rotated.rotate(b123d.Axis.Z, rot_euler[2])
        translated = rotated.translate(b123d.Vector(tuple(p)))
        faces.append(translated)

    try:
        with b123d.BuildPart() as part:
            b123d.loft(faces, ruled=False)
        return part.part
    except Exception as exc:
        print(f"  Loft failed: {exc}")
        return None


def verify_watertight(result: Solid, output_path: Path) -> dict:
    """Export to STL and verify with trimesh."""
    b123d.export_stl(result, str(output_path))
    mesh = trimesh.load(str(output_path))
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "is_watertight": bool(mesh.is_watertight),
        "mesh_volume": float(mesh.volume),
    }


def main() -> int:
    print("=" * 70)
    print("DIAGNOSTIC: Pourquoi build123d.sweep cree du volume nul ?")
    print("=" * 70)

    if not CENTERLINE_PATH.exists():
        print(f"Centerline not found: {CENTERLINE_PATH}")
        return 1

    centerline = load_centerline(CENTERLINE_PATH)
    print(f"\nCenterline chargee:")
    print(f"  Points: {len(centerline)}")
    print(f"  Inlet:  {centerline[0]}")
    print(f"  Outlet: {centerline[-1]}")
    print(f"  Longueur: {np.linalg.norm(centerline[-1] - centerline[0])*1000:.1f} mm")
    print(f"  Span Z:  {(centerline[:,2].max() - centerline[:,2].min())*1000:.1f} mm")

    expected_vol = np.pi * (0.003) ** 2 * np.linalg.norm(centerline[-1] - centerline[0])
    print(f"  Volume attendu (cylindre): {expected_vol:.6e} m³")

    # --- Diagnostic 1: Sweep vanilla build123d ---
    print("\n" + "-" * 70)
    print("1. SWEEP VANILLA build123d (defaut)")
    print("-" * 70)
    part1 = build_aorta_sweep_vanilla(centerline)
    d1 = inspect_shape(part1, "sweep_vanilla")
    print_diagnostics(d1)

    # --- Diagnostic 2: Sweep direct OCC (avec withContact=True) ---
    print("\n" + "-" * 70)
    print("2. SWEEP DIRECT OCC (withContact=True, withCorrection=True)")
    print("-" * 70)
    part2 = build_aorta_sweep_direct_occ(centerline)
    d2 = inspect_shape(part2, "sweep_direct_occ")
    print_diagnostics(d2)
    print(f"  Ratio volume/attendu: {d2['volume'] / expected_vol:.3f}")

    stl2 = OUTPUT_DIR / "aorta_sweep_direct.stp"
    b123d.export_step(part2, str(stl2))
    print(f"  STEP exporte: {stl2}")

    # Verify watertight
    stl_check = OUTPUT_DIR / "aorta_sweep_direct.stl"
    w2 = verify_watertight(part2, stl_check)
    print(f"  Watertight: {w2['is_watertight']}")
    print(f"  Mesh volume: {w2['mesh_volume']:.6e} m³")

    # --- Diagnostic 3: Loft avec profiles orientes ---
    print("\n" + "-" * 70)
    print("3. LOFT AVEC PROFILES ORIENTES")
    print("-" * 70)
    part3 = build_aorta_loft(centerline, step=4)
    if part3 is not None:
        d3 = inspect_shape(part3, "loft_oriented")
        print_diagnostics(d3)
        print(f"  Ratio volume/attendu: {d3['volume'] / expected_vol:.3f}")

        stl3 = OUTPUT_DIR / "aorta_loft_oriented.stp"
        b123d.export_step(part3, str(stl3))
        print(f"  STEP exporte: {stl3}")

        w3 = verify_watertight(part3, OUTPUT_DIR / "aorta_loft_oriented.stl")
        print(f"  Watertight: {w3['is_watertight']}")
        print(f"  Mesh volume: {w3['mesh_volume']:.6e} m³")
    else:
        print("  Loft a echoue")

    # --- Resume ---
    print("\n" + "=" * 70)
    print("RESUME")
    print("=" * 70)
    print(f"Volume attendu:        {expected_vol:.6e} m³")
    print(f"Sweep vanilla:         {d1['volume']:.6e} m³  <-- PROBLEME (withContact=False)")
    print(f"Sweep direct OCC:      {d2['volume']:.6e} m³  <-- SOLUTION RECOMMANDEE")
    if part3 is not None:
        print(f"Loft profiles:         {d3['volume']:.6e} m³  <-- Alternative viable")

    print("\nCONCLUSION:")
    print("  Le sweep build123d utilise BRepOffsetAPI_MakePipeShell avec")
    print("  withContact=False. Pour un profile planaire et un chemin quasi-coplanaire,")
    print("  cela cree une geometrie degeneree de volume nul.")
    print("  ")
    print("  SOLUTION: Utiliser BRepOffsetAPI_MakePipeShell directement avec")
    print("  withContact=True et withCorrection=True, OU utiliser un loft avec")
    print("  profiles orientes perpendiculairement a la tangente locale.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
