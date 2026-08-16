#!/usr/bin/env python3
"""
Test progressif du loft build123d pour reconstruction aortique.

Approche recommandée :
1. Ellipses simples en plans différents
2. Cercles orientés le long d'un chemin
3. Sections STL réelles

Usage:
    python3 test_loft_progressive.py
"""

import sys
from pathlib import Path

import numpy as np
import build123d as b123d

OUTPUT_DIR = Path(__file__).resolve().parent


def test_1_simple_ellipses():
    """Test 1: Loft avec ellipses simples en plans différents."""
    print("=== Test 1: Ellipses simples ===")
    
    profiles = []
    for z, rx, ry in [(0, 0.003, 0.003), (0.01, 0.0035, 0.003), (0.02, 0.004, 0.0035)]:
        with b123d.BuildSketch(b123d.Plane((0, 0, z))) as sketch:
            b123d.Ellipse(rx, ry)
        profiles.append(sketch.faces()[0])
    
    try:
        with b123d.BuildPart() as part:
            b123d.loft(profiles, ruled=False)
        result = part.part
        print(f"  Volume: {result.volume:.6e} m³")
        print(f"  Area: {result.area:.6e} m²")
        b123d.export_step(result, str(OUTPUT_DIR / "test1_ellipses.step"))
        return True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def test_2_wires_from_sketches():
    """Test 2: Loft avec Wires fermés depuis des sketches."""
    print("\n=== Test 2: Wires depuis sketches ===")
    
    wires = []
    for i, z in enumerate([0.0, 0.01, 0.02]):
        r = 0.003 + 0.0005 * i
        with b123d.BuildSketch(b123d.Plane((0, 0, z))) as sketch:
            b123d.Circle(r)
        wire = sketch.faces()[0].outer_wire()
        wires.append(wire)
    
    try:
        with b123d.BuildPart() as part:
            b123d.loft(wires, ruled=False)
        result = part.part
        print(f"  Volume: {result.volume:.6e} m³")
        b123d.export_step(result, str(OUTPUT_DIR / "test2_wires.step"))
        return True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def test_3_splines_3d():
    """Test 3: Loft avec Splines 3D périodiques."""
    print("\n=== Test 3: Splines 3D ===")
    
    wires = []
    for i, z in enumerate([0.0, 0.01, 0.02, 0.03]):
        r = 0.003 + 0.0005 * i
        n_pts = 24
        theta = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
        pts = [(r*np.cos(t), r*np.sin(t), z) for t in theta]
        
        with b123d.BuildLine() as line:
            b123d.Spline(pts, periodic=True)
        wire = line.wire()
        wires.append(wire)
    
    try:
        with b123d.BuildPart() as part:
            b123d.loft(wires, ruled=False)
        result = part.part
        print(f"  Volume: {result.volume:.6e} m³")
        b123d.export_step(result, str(OUTPUT_DIR / "test3_splines.step"))
        return True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def test_4_splines_3d_orientated():
    """Test 4: Loft avec Splines 3D orientées le long d'un chemin courbé."""
    print("\n=== Test 4: Splines 3D orientées ===")
    
    centerline = np.array([
        [0.0, 0.0, 0.0],
        [0.005, 0.0, 0.01],
        [0.01, 0.005, 0.02],
        [0.01, 0.01, 0.03],
        [0.0, 0.015, 0.04],
    ])
    
    wires = []
    for i, p in enumerate(centerline):
        if i < len(centerline) - 1:
            tangent = centerline[i+1] - p
        else:
            tangent = p - centerline[i-1]
        tangent = tangent / np.linalg.norm(tangent)
        
        if abs(tangent[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        u = np.cross(tangent, ref)
        u = u / np.linalg.norm(u)
        v = np.cross(tangent, u)
        
        r = 0.003
        n_pts = 24
        theta = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
        pts = []
        for t in theta:
            pt = p + r * (np.cos(t) * u + np.sin(t) * v)
            pts.append((float(pt[0]), float(pt[1]), float(pt[2])))
        
        with b123d.BuildLine() as line:
            b123d.Spline(pts, periodic=True)
        wire = line.wire()
        wires.append(wire)
    
    try:
        with b123d.BuildPart() as part:
            b123d.loft(wires, ruled=False)
        result = part.part
        print(f"  Volume: {result.volume:.6e} m³")
        b123d.export_step(result, str(OUTPUT_DIR / "test4_splines_3d.step"))
        return True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def test_5_real_sections():
    """Test 5: Loft avec 5 vraies sections STL."""
    print("\n=== Test 5: 5 sections STL réelles ===")
    
    from foampilot.geometry.topology import TopologySectionExtractor
    import trimesh
    
    stl_path = Path("/home/steven/foampilot/examples/coa/patient58_cfd_example/constant/triSurface/tbad_TL_walls.stl")
    centerline_path = Path("/home/steven/foampilot/examples/coa/patient58_cfd_example/centerline.npy")
    
    if not stl_path.exists() or not centerline_path.exists():
        print("  FAIL: fichiers manquants")
        return False
    
    mesh = trimesh.load(str(stl_path), process=True)
    centerline = np.load(str(centerline_path))
    
    extractor = TopologySectionExtractor(spacing_mm=20.0)
    axis = centerline[-1] - centerline[0]
    sections = extractor.extract_along_axis(mesh, axis, centerline[0], n_steps=5)
    
    print(f"  Sections extraites: {len(sections)}")
    
    wires = []
    for i, section in enumerate(sections):
        pts = section.points
        if len(pts) < 3:
            print(f"  SKIP section {i}: {len(pts)} points")
            continue
        
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
        
        try:
            with b123d.BuildLine() as line:
                b123d.Spline(pts.tolist(), periodic=True)
            wire = line.wire()
            wires.append(wire)
            print(f"  Section {i}: wire OK ({len(pts)} pts)")
        except Exception as exc:
            print(f"  Section {i}: FAIL ({exc})")
    
    if len(wires) < 2:
        print("  FAIL: pas assez de wires")
        return False
    
    try:
        with b123d.BuildPart() as part:
            b123d.loft(wires, ruled=False)
        result = part.part
        print(f"  Volume: {result.volume:.6e} m³")
        b123d.export_step(result, str(OUTPUT_DIR / "test5_real_sections.step"))
        
        b123d.export_stl(result, str(OUTPUT_DIR / "test5_real_sections.stl"))
        import trimesh
        mesh = trimesh.load(str(OUTPUT_DIR / "test5_real_sections.stl"))
        print(f"  Watertight: {mesh.is_watertight}")
        return True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def main():
    print("=" * 60)
    print("Tests progressifs du loft build123d")
    print("=" * 60)
    
    results = {}
    results["1_ellipses"] = test_1_simple_ellipses()
    results["2_wires"] = test_2_wires_from_sketches()
    results["3_splines"] = test_3_splines_3d()
    results["4_splines_3d"] = test_4_splines_3d_orientated()
    results["5_real_sections"] = test_5_real_sections()
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {name}: {status}")
    
    if not all(results.values()):
        print("\n⚠️  Certains tests ont échoué")
        print("Le problème vient probablement de:")
        print("  - La version de build123d")
        print("  - L'API loft() qui nécessite des faces et pas des wires")
        print("  - La géométrie des profils")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
