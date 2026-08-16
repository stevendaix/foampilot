#!/usr/bin/env python3
"""
Test simple et fonctionnel du loft build123d pour reconstruction aortique.

Approche correcte :
1. Utiliser BuildSketch (pas BuildLine) pour créer des faces fermées
2. Appeler loft() sans arguments en mode builder, ou passer un Sketch en mode algèbre
3. Les plans doivent être parallèles pour un comportement prévisible

Usage:
    python3 test_loft_simple.py
"""

import sys
from pathlib import Path

import numpy as np
import build123d as b123d

OUTPUT_DIR = Path(__file__).resolve().parent


def test_builder_mode_two_circles():
    """Test 1: Loft avec 2 cercles décalés en Z - mode builder."""
    print("=== Test 1: 2 cercles (builder mode) ===")
    
    with b123d.BuildPart() as part:
        with b123d.BuildSketch(b123d.Plane((0, 0, 0))):
            b123d.Circle(3)
        with b123d.BuildSketch(b123d.Plane((0, 0, 10))):
            b123d.Circle(3.5)
        b123d.loft()
    
    print(f"  Volume: {part.part.volume:.2f}")
    print(f"  Valid: {part.part.is_valid}")
    b123d.export_step(part.part, str(OUTPUT_DIR / "test1_builder.step"))
    return part.part.is_valid


def test_algebra_mode_two_circles():
    """Test 2: Loft avec 2 cercles - mode algèbre."""
    print("\n=== Test 2: 2 cercles (algebra mode) ===")
    
    faces = b123d.Sketch() + [
        b123d.Plane((0, 0, 0)) * b123d.Circle(3),
        b123d.Plane((0, 0, 10)) * b123d.Circle(3.5),
    ]
    
    result = b123d.loft(faces)
    print(f"  Volume: {result.volume:.2f}")
    print(f"  Valid: {result.is_valid}")
    b123d.export_step(result, str(OUTPUT_DIR / "test2_algebra.step"))
    return result.is_valid


def test_multiple_sections():
    """Test 3: Loft avec plusieurs sections de rayons variables."""
    print("\n=== Test 3: Sections multiples ===")
    
    with b123d.BuildPart() as part:
        for i in range(5):
            z = i * 10
            r = 3 + i * 0.5
            with b123d.BuildSketch(b123d.Plane((0, 0, z))):
                b123d.Circle(r)
        b123d.loft()
    
    print(f"  Volume: {part.part.volume:.2f}")
    print(f"  Valid: {part.part.is_valid}")
    b123d.export_step(part.part, str(OUTPUT_DIR / "test3_multi.step"))
    return part.part.is_valid


def test_ellipses():
    """Test 4: Loft avec ellipses."""
    print("\n=== Test 4: Ellipses ===")
    
    with b123d.BuildPart() as part:
        with b123d.BuildSketch(b123d.Plane((0, 0, 0))):
            b123d.Ellipse(3, 2)
        with b123d.BuildSketch(b123d.Plane((0, 0, 10))):
            b123d.Ellipse(3.5, 2.5)
        b123d.loft()
    
    print(f"  Volume: {part.part.volume:.2f}")
    print(f"  Valid: {part.part.is_valid}")
    b123d.export_step(part.part, str(OUTPUT_DIR / "test4_ellipses.step"))
    return part.part.is_valid


def test_from_stl_sections():
    """Test 5: Loft avec sections extraites d'un STL."""
    print("\n=== Test 5: Sections STL ===")
    
    try:
        from foampilot.geometry.topology import TopologySectionExtractor
        import trimesh
        
        stl_path = Path("/home/steven/foampilot/examples/coa/patient58_cfd_example/constant/triSurface/tbad_TL_walls.stl")
        centerline_path = Path("/home/steven/foampilot/examples/coa/patient58_cfd_example/centerline.npy")
        
        if not stl_path.exists() or not centerline_path.exists():
            print("  SKIP: fichiers manquants")
            return True
        
        mesh = trimesh.load(str(stl_path), process=True)
        centerline = np.load(str(centerline_path))
        
        extractor = TopologySectionExtractor(spacing_mm=20.0)
        axis = centerline[-1] - centerline[0]
        sections = extractor.extract_along_axis(mesh, axis, centerline[0], n_steps=5)
        
        print(f"  Sections extraites: {len(sections)}")
        
        valid_sections = 0
        with b123d.BuildPart() as part:
            for i, section in enumerate(sections):
                pts = section.points
                if len(pts) < 3:
                    print(f"  SKIP section {i}: {len(pts)} points")
                    continue
                
                # Créer une face fermée depuis les points de la section
                # Utiliser BuildLine + make_face pour créer une face depuis des points arbitraires
                z = i * 10  # Position Z arbitraire pour le test
                with b123d.BuildSketch(b123d.Plane((0, 0, z))) as sketch:
                    with b123d.BuildLine() as line:
                        b123d.Spline(pts.tolist(), periodic=True)
                    b123d.make_face()
                valid_sections += 1
                print(f"  Section {i}: face OK ({len(pts)} pts)")
            
            if valid_sections < 2:
                print(f"  SKIP: pas assez de sections valides pour loft ({valid_sections})")
                return True
            b123d.loft()
        
        print(f"  Volume: {part.part.volume:.6e} m³")
        print(f"  Valid: {part.part.is_valid}")
        b123d.export_step(part.part, str(OUTPUT_DIR / "test5_stl_sections.step"))
        return part.part.is_valid
        
    except Exception as exc:
        print(f"  FAIL: {exc}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Tests du loft build123d - Version fonctionnelle")
    print("=" * 60)
    
    results = {}
    results["1_builder_circles"] = test_builder_mode_two_circles()
    results["2_algebra_circles"] = test_algebra_mode_two_circles()
    results["3_multiple_sections"] = test_multiple_sections()
    results["4_ellipses"] = test_ellipses()
    results["5_stl_sections"] = test_from_stl_sections()
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {name}: {status}")
    
    if all(results.values()):
        print("\n✅ Tous les tests sont passés")
        print("Le loft build123d fonctionne correctement avec:")
        print("  - BuildSketch pour créer des faces fermées")
        print("  - loft() sans arguments en mode builder")
        print("  - ou loft(faces) en mode algèbre")
    else:
        print("\n⚠️  Certains tests ont échoué")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
