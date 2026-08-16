#!/usr/bin/env python3
"""
Test minimal du loft build123d pour reconstruction aortique.

Approche progressive :
1. D'abord avec des ellipses simples dans des plans différents
2. Puis avec des sections STL réelles
"""

import sys
from pathlib import Path

import numpy as np
import build123d as b123d

OUTPUT_DIR = Path(__file__).resolve().parent


def test_simple_loft():
    """Test 1: Loft avec ellipses simples en plans différents."""
    print("=== Test 1: Loft simple (ellipses) ===")
    
    profiles = []
    for i, (z, rx, ry) in enumerate([
        (0.0, 0.003, 0.003),
        (0.01, 0.0035, 0.003),
        (0.02, 0.004, 0.0035),
        (0.03, 0.0045, 0.004),
    ]):
        with b123d.BuildSketch(b123d.Plane((0, 0, z))) as sketch:
            b123d.Ellipse(rx, ry)
        profiles.append(sketch.faces()[0])
    
    try:
        with b123d.BuildPart() as part:
            b123d.loft(profiles, ruled=False)
        result = part.part
        print(f"  ✅ Loft OK: volume={result.volume:.6e} m³")
        b123d.export_step(result, str(OUTPUT_DIR / "test_loft_simple.step"))
        return True
    except Exception as exc:
        print(f"  ❌ Loft failed: {exc}")
        return False


def test_wire_loft():
    """Test 2: Loft avec Wires fermés (Spline périodique)."""
    print("\n=== Test 2: Loft avec Wires fermés ===")
    
    wires = []
    for i, z in enumerate([0.0, 0.01, 0.02, 0.03]):
        # Create circular points in XY plane
        n_pts = 32
        theta = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
        r = 0.003 + 0.0005 * i
        pts = [(r*np.cos(t), r*np.sin(t), z) for t in theta]
        
        # Create spline curve
        with b123d.BuildLine() as line:
            b123d.Spline(pts, periodic=True)
        wire = line.wire()
        wires.append(wire)
    
    try:
        with b123d.BuildPart() as part:
            b123d.loft(wires, ruled=False)
        result = part.part
        print(f"  ✅ Loft OK: volume={result.volume:.6e} m³")
        b123d.export_step(result, str(OUTPUT_DIR / "test_loft_wires.step"))
        return True
    except Exception as exc:
        print(f"  ❌ Loft failed: {exc}")
        return False


def test_wire_loft_3d_planes():
    """Test 3: Loft avec Wires en plans 3D différents (comme sections aorte)."""
    print("\n=== Test 3: Loft avec Wires en plans 3D ===")
    
    wires = []
    centerline = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.01],
        [0.0, 0.0, 0.02],
        [0.01, 0.0, 0.03],
        [0.01, 0.0, 0.04],
    ])
    
    for i, p in enumerate(centerline):
        # Create local frame perpendicular to centerline tangent
        if i < len(centerline) - 1:
            tangent = centerline[i+1] - p
        else:
            tangent = p - centerline[i-1]
        tangent = tangent / np.linalg.norm(tangent)
        
        # Build orthonormal basis
        if abs(tangent[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        u = np.cross(tangent, ref)
        u = u / np.linalg.norm(u)
        v = np.cross(tangent, u)
        
        # Create circle points in local frame
        n_pts = 32
        theta = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
        r = 0.003
        circle_pts = []
        for t in theta:
            pt = p + r * (np.cos(t) * u + np.sin(t) * v)
            circle_pts.append((float(pt[0]), float(pt[1]), float(pt[2])))
        
        with b123d.BuildLine() as line:
            b123d.Spline(circle_pts, periodic=True)
        wire = line.wire()
        wires.append(wire)
    
    try:
        with b123d.BuildPart() as part:
            b123d.loft(wires, ruled=False)
        result = part.part
        print(f"  ✅ Loft OK: volume={result.volume:.6e} m³")
        b123d.export_step(result, str(OUTPUT_DIR / "test_loft_3d.step"))
        return True
    except Exception as exc:
        print(f"  ❌ Loft failed: {exc}")
        return False


def test_few_real_sections():
    """Test 4: Loft avec 5 vraies sections STL."""
    print("\n=== Test 4: Loft avec 5 sections STL réelles ===")
    
    from foampilot.geometry.topology import TopologySectionExtractor
    import trimesh
    
    stl_path = Path("/home/steven/foampilot/examples/coa/patient58_cfd_example/constant/triSurface/tbad_TL_walls.stl")
    centerline_path = Path("/home/steven/foampilot/examples/coa/patient58_cfd_example/centerline.npy")
    
    if not stl_path.exists() or not centerline_path.exists():
        print("  ❌ Fichiers manquants")
        return False
    
    mesh = trimesh.load(str(stl_path), process=True)
    centerline = np.load(str(centerline_path))
    
    extractor = TopologySectionExtractor(spacing_mm=2.0)
    axis = centerline[-1] - centerline[0]
    length = np.linalg.norm(axis)
    n_steps = 5
    sections = extractor.extract_along_axis(mesh, axis, centerline[0], n_steps=n_steps)
    
    print(f"  Sections extraites: {len(sections)}")
    
    wires = []
    for i, section in enumerate(sections):
        pts = section.points
        if len(pts) < 3:
            print(f"  ⚠️ Section {i}: trop peu de points ({len(pts)})")
            continue
        
        # Ensure closed contour
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
        
        # Resample to 64 points
        from scipy.interpolate import interp1d
        diffs = np.diff(pts, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        cumulative = np.insert(np.cumsum(seg_len), 0, 0.0)
        total = cumulative[-1]
        if total < 1e-9:
            continue
        uniform_t = np.linspace(0, total, 64)
        x = np.interp(uniform_t, cumulative, pts[:, 0])
        y = np.interp(uniform_t, cumulative, pts[:, 1])
        z = np.interp(uniform_t, cumulative, pts[:, 2])
        resampled = np.column_stack([x, y, z])
        
        try:
            with b123d.BuildLine() as line:
                b123d.Spline(resampled.tolist(), periodic=True)
            wire = line.wire()
            wires.append(wire)
            print(f"  Section {i}: {len(resampled)} points → wire OK")
        except Exception as exc:
            print(f"  Section {i}: wire failed: {exc}")
    
    if len(wires) < 2:
        print("  ❌ Pas assez de wires pour loft")
        return False
    
    try:
        with b123d.BuildPart() as part:
            b123d.loft(wires, ruled=False)
        result = part.part
        print(f"  ✅ Loft OK: volume={result.volume:.6e} m³")
        b123d.export_step(result, str(OUTPUT_DIR / "test_loft_real_sections.step"))
        
        # Verify watertight
        import trimesh
        b123d.export_stl(result, str(OUTPUT_DIR / "test_loft_real_sections.stl"))
        mesh = trimesh.load(str(OUTPUT_DIR / "test_loft_real_sections.stl"))
        print(f"  Watertight: {mesh.is_watertight}")
        return True
    except Exception as exc:
        print(f"  ❌ Loft failed: {exc}")
        return False


def main():
    print("=" * 60)
    print("Tests loft build123d pour reconstruction aortique")
    print("=" * 60)
    
    results = {}
    results["simple_ellipse"] = test_simple_loft()
    results["wire_circle"] = test_wire_loft()
    results["wire_3d"] = test_wire_loft_3d_planes()
    results["real_sections"] = test_few_real_sections()
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {name}: {status}")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
