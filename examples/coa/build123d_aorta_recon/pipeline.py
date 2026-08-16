#!/usr/bin/env python3
"""
Pipeline de reconstruction aortique patient-specific.

Architecture (plan.md):
  STL → VMTK (centerlines + sections) → Build123 (loft/sweep) → STEP/BREP → Gmsh → OpenFOAM

Étapes :
  1. Extraction centerline avec VMTK local
  2. Extraction sections perpendiculaires + rayons
  3. Reconstruction CAD avec build123d (sweep OCC direct + loft orienté)
  4. Reconstruction CAD avec sections STL réelles + B-splines (Gmsh occ)
  5. Export STEP + STL
  6. Comparaison STL original vs CAD
  7. Préparation pour Gmsh/OpenFOAM

Usage:
    python3 pipeline.py
"""

import sys
from pathlib import Path

import gmsh
import numpy as np
import trimesh

import build123d as b123d

# foampilot local VMTK
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "foampilot" / "src"))

from foampilot.geometry.topology import (
    BoundaryRole,
    OpenProfile,
    OpenProfileClassifier,
    SurfaceTopologyAnalyzer,
    TopologyCenterlineExtractor,
    TopologySectionExtractor,
)
from foampilot.geometry.topology.vmtk.vmtkcenterlines import vmtkCenterlines, _trimesh_to_vtk_polydata
from foampilot.geometry.cad.bspline_fitter import BSplineFitter
from foampilot.geometry.cad.occ_builder import OCCBuilder

# Paths
BASE_DIR = Path(__file__).resolve().parent
STL_PATH = BASE_DIR.parent / "patient58_cfd_example" / "constant" / "triSurface" / "tbad_TL_walls.stl"
CENTERLINE_PATH = BASE_DIR.parent / "patient58_cfd_example" / "centerline.npy"
OUTPUT_DIR = BASE_DIR


def load_stl(path: Path) -> trimesh.Trimesh:
    """Load and process STL mesh."""
    mesh = trimesh.load(str(path), process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    return mesh


def step1_extract_centerline(mesh: trimesh.Trimesh) -> np.ndarray:
    """Step 1: Extract centerline using local VMTK implementation.

    Returns:
        (N, 3) array of centerline points in meters.
    """
    print("\n=== Step 1: Extraction centerline VMTK ===")

    # Use local VMTK implementation
    centerliner = vmtkCenterlines()
    centerliner.Surface = _trimesh_to_vtk_polydata(mesh)

    # Auto-detect inlet/outlet using PCA
    verts = mesh.vertices
    mean = verts.mean(axis=0)
    centered = verts - mean
    cov = centered.T @ centered / max(len(verts) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    principal_axis = eigvecs[:, order[0]]
    projections = centered @ principal_axis
    proj_min = projections.min()
    proj_max = projections.max()
    length = proj_max - proj_min
    thresh = length * 0.01
    inlet_vidx = np.where(projections <= proj_min + thresh)[0]
    outlet_vidx = np.where(projections >= proj_max - thresh)[0]
    inlet_center = verts[inlet_vidx].mean(axis=0) if len(inlet_vidx) > 0 else verts[projections.argmin()]
    outlet_center = verts[outlet_vidx].mean(axis=0) if len(outlet_vidx) > 0 else verts[projections.argmax()]

    source_ids = [int(np.argmin(np.linalg.norm(mesh.vertices - inlet_center, axis=1)))]
    target_ids = [int(np.argmin(np.linalg.norm(mesh.vertices - outlet_center, axis=1)))]

    centerliner.SeedSelectorName = "idlist"
    centerliner.SourceIds = source_ids
    centerliner.TargetIds = target_ids
    centerliner.ResamplingStepLength = 0.002  # 2mm
    centerliner.Execute()

    if centerliner.Centerlines is None:
        raise RuntimeError("Centerline extraction failed")

    n_pts = centerliner.Centerlines.GetNumberOfPoints()
    centerline = np.array([centerliner.Centerlines.GetPoint(i) for i in range(n_pts)])
    print(f"  Centerline: {len(centerline)} points")
    print(f"  Inlet: {centerline[0]}")
    print(f"  Outlet: {centerline[-1]}")
    print(f"  Longueur: {np.linalg.norm(centerline[-1] - centerline[0])*1000:.1f} mm")

    return centerline


def step2_extract_sections(mesh: trimesh.Trimesh, centerline: np.ndarray) -> list:
    """Step 2: Extract sections perpendicular to centerline.

    Returns:
        List of Section objects with position, normal, points, radius.
    """
    print("\n=== Step 2: Extraction sections VMTK ===")

    axis = centerline[-1] - centerline[0]
    length = np.linalg.norm(axis)
    n_steps = max(2, int(length / 0.002))  # 2mm spacing

    extractor = TopologySectionExtractor(spacing_mm=0.002)
    sections = extractor.extract_along_axis(
        mesh=mesh,
        axis=axis,
        origin=centerline[0],
        n_steps=n_steps,
    )

    print(f"  Sections extraites: {len(sections)}")
    if sections:
        radii = [s.radius for s in sections if s.radius > 0]
        areas = [s.area for s in sections if s.area > 0]
        if radii:
            print(f"  Rayon moyen: {np.mean(radii)*1000:.1f} mm")
            print(f"  Rayon min: {np.min(radii)*1000:.1f} mm")
            print(f"  Rayon max: {np.max(radii)*1000:.1f} mm")
            print(f"  Aire moyenne: {np.mean(areas)*1e6:.2f} mm²")

    return sections


def step2b_diagnostic_sections(centerline: np.ndarray, mesh: trimesh.Trimesh) -> None:
    """Diagnostic: verify section geometry before loft."""
    print("\n=== Step 2b: Diagnostic sections ===")

    from foampilot.geometry.topology.section_extractor import _process_section_polylines

    sample_indices = _sample_centerline_by_spacing(centerline, spacing_mm=2.0)
    sections = []
    for i in sample_indices:
        center = centerline[i]
        if i == 0:
            tangent = centerline[1] - centerline[0]
        elif i >= len(centerline) - 1:
            tangent = centerline[-1] - centerline[-2]
        else:
            tangent = centerline[i + 1] - centerline[i - 1]
        tangent = tangent / np.linalg.norm(tangent)

        try:
            section = mesh.section(plane_origin=center, plane_normal=tangent)
            points = _process_section_polylines(section, tangent, center, n_resample=64)
            if points is not None:
                sections.append((i, points, tangent))
        except Exception:
            pass

    print(f"  Sections diagnostiquees: {len(sections)}")

    if len(sections) < 2:
        print("  Pas assez de sections pour diagnostic")
        return

    # Compute metrics
    table = []
    for idx, pts, tangent in sections:
        c = pts.mean(axis=0)
        d = np.linalg.norm(pts - c, axis=1)
        r = np.mean(d)
        a = np.pi * r ** 2
        table.append((idx, r, a, c))

    print(f"\n  {'Station':>7} {'Idx CL':>7} {'r_eq (mm)':>10} {'Area (mm2)':>12} {'Centroid':>30}")
    for idx, r, a, c in table:
        print(f"  {idx:7d} {idx:7d} {r*1000:10.2f} {a*1e6:12.2f} {str(c.round(4)):>30}")

    # Check phase continuity
    print("\n  === Continuite entre sections ===")
    for k in range(len(sections) - 1):
        idx0, pts0, _ = sections[k]
        idx1, pts1, _ = sections[k + 1]
        c0 = pts0.mean(axis=0)
        c1 = pts1.mean(axis=0)
        d0 = np.linalg.norm(pts0 - c0, axis=1)
        d1 = np.linalg.norm(pts1 - c1, axis=1)
        r0 = np.mean(d0)
        r1 = np.mean(d1)
        dr = abs(r1 - r0) * 1000
        ds = np.linalg.norm(c1 - c0) * 1000
        print(f"    Sec {idx0} -> {idx1}: dr={dr:6.2f} mm, ds={ds:6.2f} mm")


def step3_build_cad_sweep(centerline: np.ndarray, radius_mm: float = 3.0) -> b123d.Part | None:
    """Step 3a: Build CAD using direct OCC sweep (recommended).

    Uses BRepOffsetAPI_MakePipeShell with withContact=True.
    """
    print("\n=== Step 3a: Reconstruction CAD (sweep OCC direct) ===")

    radius = radius_mm / 1000.0

    # Build path from centerline
    with b123d.BuildLine() as line:
        b123d.Polyline([tuple(p) for p in centerline])
    path = line.wire()

    # Create circular profile
    with b123d.BuildSketch() as sketch:
        b123d.Circle(radius)
    face = sketch.faces()[0]
    outer = face.outer_wire()

    # Direct OCC sweep with withContact=True
    from OCP.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
    builder = BRepOffsetAPI_MakePipeShell(path.wrapped)
    builder.SetMode(False)
    builder.Add(outer.wrapped, True, True)  # withContact=True, withCorrection=True
    builder.Build()
    builder.MakeSolid()

    part = b123d.Solid(builder.Shape())

    print(f"  Volume: {part.volume:.6e} m³")
    print(f"  Area: {part.area:.6e} m²")
    print(f"  Valid: {part.is_valid}")

    return part


def step3_build_cad_loft(centerline: np.ndarray, sections: list, radius_mm: float = 3.0) -> b123d.Part | None:
    """Step 3b: Build CAD using build123d loft with projected sections.

    Projects STL sections onto parallel XY planes and lofts them.
    This works because build123d loft requires approximately parallel planes.
    """
    print("\n=== Step 3b: Reconstruction CAD (loft build123d sections projetees) ===")

    if len(sections) < 2:
        print("  Not enough sections")
        return None

    try:
        with b123d.BuildPart() as part:
            for i, section in enumerate(sections):
                pts = section.points
                if len(pts) < 3:
                    continue

                # Project section onto XY plane (perpendicular to centerline axis)
                # Place sections along Z axis
                z = i * 0.015  # 15mm spacing
                center_pt = pts.mean(axis=0)
                centered_pts = pts - center_pt

                with b123d.BuildSketch(b123d.Plane((0, 0, z))) as sketch:
                    with b123d.BuildLine() as line:
                        b123d.Spline(centered_pts.tolist(), periodic=True)
                    b123d.make_face()

            b123d.loft()

        part_obj = part.part
        if part_obj.is_valid:
            print(f"  Volume: {part_obj.volume:.6e} m³")
            print(f"  Area: {part_obj.area:.6e} m²")
            print(f"  Valid: {part_obj.is_valid}")
            return part_obj
        else:
            print("  Loft created but invalid")
            return None
    except Exception as exc:
        print(f"  Loft failed: {exc}")
        return None


def _build_loft_faces(centerline, radius, step, ruled=True):
    """Helper to build loft faces."""
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
    return faces


def _try_loft(centerline, radius, step, ruled=True):
    """Try loft with given parameters."""
    try:
        faces = _build_loft_faces(centerline, radius, step, ruled)
        if len(faces) < 2:
            return None

        with b123d.BuildPart() as part:
            b123d.loft(faces, ruled=ruled)
        part_obj = part.part
        if part_obj.is_valid:
            print(f"  Volume: {part_obj.volume:.6e} m³")
            print(f"  Area: {part_obj.area:.6e} m²")
            print(f"  Valid: {part_obj.is_valid}")
            return part_obj
        else:
            return None
    except Exception as exc:
        print(f"  Loft failed: {exc}")
        return None


def _try_loft_wires(centerline, radius, step, ruled=True):
    """Try loft using wires directly instead of faces."""
    try:
        wires = []
        for i in range(0, len(centerline), step):
            p = centerline[i]
            if i < len(centerline) - 1:
                tangent = centerline[i + 1] - p
            else:
                tangent = p - centerline[i - 1]
            tangent = tangent / (np.linalg.norm(tangent) + 1e-12)

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
            face = sketch.faces()[0]
            wires.append(face.outer_wire())

        if len(wires) < 2:
            return None

        with b123d.BuildPart() as part:
            b123d.loft(wires, ruled=ruled)
        part_obj = part.part
        if part_obj.is_valid:
            print(f"  Volume: {part_obj.volume:.6e} m³")
            print(f"  Area: {part_obj.area:.6e} m²")
            print(f"  Valid: {part_obj.is_valid}")
            return part_obj
        else:
            return None
    except Exception as exc:
        print(f"  Wire loft failed: {exc}")
        return None


def _try_loft_fixed_orientation(centerline, radius, step, ruled=True):
    """Try loft with fixed plane orientation (all planes parallel)."""
    try:
        faces = []
        # Use a fixed normal direction
        normal = centerline[-1] - centerline[0]
        normal = normal / np.linalg.norm(normal)

        for i in range(0, len(centerline), step):
            p = centerline[i]
            plane = b123d.Plane(
                origin=b123d.Vector(tuple(p)),
                z_dir=b123d.Vector(tuple(normal)),
            )

            with b123d.BuildSketch(plane) as sketch:
                b123d.Circle(radius)
            faces.append(sketch.faces()[0])

        if len(faces) < 2:
            return None

        with b123d.BuildPart() as part:
            b123d.loft(faces, ruled=ruled)
        part_obj = part.part
        if part_obj.is_valid:
            print(f"  Volume: {part_obj.volume:.6e} m³")
            print(f"  Area: {part_obj.area:.6e} m²")
            print(f"  Valid: {part_obj.is_valid}")
            return part_obj
        else:
            return None
    except Exception as exc:
        print(f"  Fixed orientation loft failed: {exc}")
        return None


def _sample_centerline_by_spacing(centerline: np.ndarray, spacing_mm: float):
    """Sample centerline indices at roughly fixed arc-length spacing."""
    diffs = np.diff(centerline, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cumulative[-1]
    if total_length <= 0:
        return list(range(len(centerline)))
    n_samples = max(2, int(total_length / (spacing_mm / 1000.0)))
    targets = np.linspace(0, total_length, n_samples)
    indices = [int(np.argmin(np.abs(cumulative - t))) for t in targets]
    seen = set()
    unique_indices = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    return unique_indices


def step3b_build_cad_loft_stl_sections(
    centerline: np.ndarray, mesh: trimesh.Trimesh, radius_mm: float = 3.0
) -> b123d.Part | None:
    """Step 3c: Build CAD from actual STL sections using OCC ThruSections.

    Extracts cross-sections from STL at centerline points and lofts them
    using BRepOffsetAPI_ThruSections. This produces a patient-specific
    CAD model with variable cross-section along the centerline.
    """
    print("\n=== Step 3c: Reconstruction CAD (sections STL reelles + OCC loft) ===")

    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeEdge
    from OCP.BRep import BRep_Tool
    from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Ax3, gp_Trsf
    from foampilot.geometry.topology.section_extractor import _process_section_polylines

    sample_indices = _sample_centerline_by_spacing(centerline, spacing_mm=2.0)
    print(f"  Centerline samples: {len(sample_indices)}")

    # Extract sections from STL at centerline points using LOCAL tangent
    sections = []
    for i in sample_indices:
        center = centerline[i]
        if i == 0:
            tangent = centerline[1] - centerline[0]
        elif i >= len(centerline) - 1:
            tangent = centerline[-1] - centerline[-2]
        else:
            tangent = centerline[i + 1] - centerline[i - 1]
        tangent = tangent / np.linalg.norm(tangent)

        try:
            section = mesh.section(plane_origin=center, plane_normal=tangent)
            points = _process_section_polylines(section, tangent, center, n_resample=64)
            if points is not None:
                sections.append((i, points, tangent))
        except Exception:
            pass

    print(f"  Sections STL extraites: {len(sections)}")

    if len(sections) < 2:
        print("  Pas assez de sections pour loft")
        return None

    # Build OCC wires from STL sections
    wires = []
    for idx, pts, tangent in sections:
        p = centerline[idx]

        if abs(tangent[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        u = np.cross(tangent, ref)
        u = u / np.linalg.norm(u)

        # Points are already in GLOBAL coordinates from mesh.section()
        # No transformation needed - use them directly
        corners = [
            gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
            for pt in pts
        ]

        wire_builder = BRepBuilderAPI_MakeWire()
        for j in range(len(corners)):
            edge = BRepBuilderAPI_MakeEdge(
                corners[j], corners[(j + 1) % len(corners)]
            ).Edge()
            wire_builder.Add(edge)

        wire = wire_builder.Wire()
        wires.append(wire)

    # Build loft
    loft = BRepOffsetAPI_ThruSections(True, False)  # filled=True, ruled=False
    loft.CheckCompatibility(True)
    for w in wires:
        loft.AddWire(w)
    loft.Build()

    if loft.IsDone():
        shape = loft.Shape()
        part = b123d.Solid(shape)
        print(f"  Volume: {part.volume:.6e} m³")
        print(f"  Area: {part.area:.6e} m²")
        print(f"  Valid: {part.is_valid}")
        return part
    else:
        print("  OCC loft not done")
        return None


def step4_export(part: b123d.Part, name: str) -> None:
    """Step 4: Export CAD to STEP and STL."""
    print(f"\n=== Step 4: Export {name} ===")

    step_path = OUTPUT_DIR / f"{name}.step"
    stl_path = OUTPUT_DIR / f"{name}.stl"

    b123d.export_step(part, str(step_path))
    b123d.export_stl(part, str(stl_path))

    print(f"  STEP: {step_path}")
    print(f"  STL: {stl_path}")


def step4b_build_cad_from_actual_sections(centerline: np.ndarray, sections: list) -> dict:
    """Step 4b: Build CAD from actual STL cross-sections using Gmsh occ.

    This uses the actual cross-sectional contours from the STL, fits B-splines,
    and creates a lofted solid in Gmsh. Also exports to STL and STEP.
    """
    print("\n=== Step 4b: Reconstruction CAD (sections STL reelles + B-splines) ===")

    if len(sections) < 2:
        print("  Not enough sections")
        return {"method": "none", "volume_tag": -1}

    builder = OCCBuilder(n_samples=40, mesh_size_factor=1.0)
    try:
        result = builder.build_from_sections(sections)
        print(f"  Method: {result.get('method', 'unknown')}")
        print(f"  Sections used: {result.get('sections', 0)}")
        print(f"  Curves used: {result.get('curves', 0)}")

        # Export Gmsh geometry to STL
        vol_tag = result.get("volume_tag", -1)
        if vol_tag > 0:
            stl_path = OUTPUT_DIR / "aorta_gmsh_loft.stl"
            step_path = OUTPUT_DIR / "aorta_gmsh_loft.step"
            gmsh.model.mesh.generate(2)
            gmsh.write(str(stl_path))
            print(f"  STL export: {stl_path}")
            result["stl_path"] = str(stl_path)

        return result
    except Exception as exc:
        print(f"  Gmsh build failed: {exc}")
        return {"method": "none", "volume_tag": -1}


def step5_compare_with_stl(cad_stl_path: Path, reference_stl_path: Path) -> dict:
    """Step 5: Compare CAD STL with original STL."""
    print(f"\n=== Step 5: Comparaison STL ===")

    cad_mesh = trimesh.load(str(cad_stl_path), process=True)
    ref_mesh = trimesh.load(str(reference_stl_path), process=True)

    if not isinstance(cad_mesh, trimesh.Trimesh):
        cad_mesh = cad_mesh.dump(concatenate=True)
    if not isinstance(ref_mesh, trimesh.Trimesh):
        ref_mesh = ref_mesh.dump(concatenate=True)

    print(f"  CAD: {len(cad_mesh.vertices)} vertices, {len(cad_mesh.faces)} faces")
    print(f"  Référence: {len(ref_mesh.vertices)} vertices, {len(ref_mesh.faces)} faces")

    # Align CAD to reference
    cad_center = cad_mesh.centroid
    ref_center = ref_mesh.centroid
    translation = ref_center - cad_center
    cad_aligned = cad_mesh.copy()
    cad_aligned.vertices += translation

    # Hausdorff distance
    from scipy.spatial import cKDTree
    pts1 = ref_mesh.sample(5000)
    pts2 = cad_aligned.sample(5000)
    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)
    dist1, _ = tree2.query(pts1, k=1)
    dist2, _ = tree1.query(pts2, k=1)
    hausdorff = float(max(dist1.max(), dist2.max()))
    mean_dist = float((dist1.mean() + dist2.mean()) / 2)

    print(f"  Hausdorff: {hausdorff*1000:.2f} mm")
    print(f"  Distance moyenne: {mean_dist*1000:.2f} mm")

    return {
        "hausdorff_mm": hausdorff * 1000,
        "mean_distance_mm": mean_dist * 1000,
        "cad_volume": float(cad_aligned.volume) if cad_aligned.is_watertight else None,
        "ref_volume": float(ref_mesh.volume) if ref_mesh.is_watertight else None,
    }


def step6_verify_outputs() -> dict:
    """Step 6: Verify all exported files exist and check watertightness."""
    print("\n=== Step 6: Verification des fichiers ===")

    files_to_check = [
        "aorta_sweep_direct_v2.step",
        "aorta_sweep_direct_v2.stl",
        "aorta_loft_occ.step",
        "aorta_loft_occ.stl",
        "aorta_loft_stl_sections.step",
        "aorta_loft_stl_sections.stl",
        "aorta_gmsh_loft.stl",
        "centerline_reconstructed.npy",
    ]

    results = {}
    for fname in files_to_check:
        path = OUTPUT_DIR / fname
        exists = path.exists()
        results[fname] = exists
        status = "OK" if exists else "MANQUANT"
        print(f"  {fname}: {status}")

    # Check STL watertightness
    for fname in [
        "aorta_sweep_direct_v2.stl",
        "aorta_loft_occ.stl",
        "aorta_loft_stl_sections.stl",
        "aorta_gmsh_loft.stl",
    ]:
        path = OUTPUT_DIR / fname
        if path.exists():
            mesh = trimesh.load(str(path), process=True)
            if isinstance(mesh, trimesh.Trimesh):
                watertight = mesh.is_watertight
                results[f"{fname}_watertight"] = watertight
                print(f"  {fname} watertight: {watertight}")
            else:
                results[f"{fname}_watertight"] = False
                print(f"  {fname} watertight: False (not a single mesh)")

    return results


def main() -> int:
    print("=" * 70)
    print("PIPELINE RECONSTRUCTION AORTIQUE PATIENT-SPECIFIC")
    print("Architecture: STL → VMTK → Build123 → STEP/BREP → Gmsh → OpenFOAM")
    print("=" * 70)

    if not STL_PATH.exists():
        print(f"STL not found: {STL_PATH}")
        return 1

    # Load STL
    mesh = load_stl(STL_PATH)
    print(f"STL chargé: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"  Area: {mesh.area:.6f} m²")
    print(f"  Bounds: {mesh.bounds}")

    # Step 1: Centerline extraction
    centerline = step1_extract_centerline(mesh)
    np.save(str(OUTPUT_DIR / "centerline_reconstructed.npy"), centerline)

    # Step 2: Section extraction
    sections = step2_extract_sections(mesh, centerline)

    # Step 2b: Diagnostic sections
    step2b_diagnostic_sections(centerline, mesh)

    # Step 3a: CAD reconstruction - sweep direct OCC
    part_sweep = step3_build_cad_sweep(centerline, radius_mm=3.0)
    if part_sweep is not None:
        step4_export(part_sweep, "aorta_sweep_direct_v2")

    # Step 3b: CAD reconstruction - loft orienté
    part_loft = step3_build_cad_loft(centerline, sections, radius_mm=3.0)
    if part_loft is not None:
        step4_export(part_loft, "aorta_loft_oriented_v2")

    # Step 3c: CAD reconstruction from actual STL sections (OCC ThruSections)
    part_loft_stl = step3b_build_cad_loft_stl_sections(centerline, mesh, radius_mm=3.0)
    if part_loft_stl is not None:
        step4_export(part_loft_stl, "aorta_loft_stl_sections")

    # Step 5: Comparison
    metrics = {}
    if part_sweep is not None:
        metrics["sweep"] = step5_compare_with_stl(
            OUTPUT_DIR / "aorta_sweep_direct_v2.stl",
            STL_PATH,
        )
        print(f"\nMétriques sweep: {metrics['sweep']}")

    if part_loft is not None:
        metrics["loft"] = step5_compare_with_stl(
            OUTPUT_DIR / "aorta_loft_oriented_v2.stl",
            STL_PATH,
        )
        print(f"\nMétriques loft: {metrics['loft']}")

    if part_loft_stl is not None:
        metrics["loft_stl"] = step5_compare_with_stl(
            OUTPUT_DIR / "aorta_loft_stl_sections.stl",
            STL_PATH,
        )
        print(f"\nMétriques loft STL sections: {metrics['loft_stl']}")

    # Step 6: Verify outputs
    verify_results = step6_verify_outputs()

    # Save metrics
    import json
    metrics_path = OUTPUT_DIR / "pipeline_metrics.json"
    with open(str(metrics_path), "w") as f:
        json.dump({
            "metrics": metrics,
            "verify": verify_results,
        }, f, indent=2, default=str)
    print(f"\nMétriques sauvegardées: {metrics_path}")

    print("\n" + "=" * 70)
    print("PIPELINE TERMINÉ")
    print("=" * 70)
    print(f"Outputs dans: {OUTPUT_DIR}")
    print("Prochaines étapes:")
    print("  1. Importer STEP dans Gmsh")
    print("  2. Générer maillage CFD")
    print("  3. Exporter vers OpenFOAM via foampilot")

    return 0


if __name__ == "__main__":
    sys.exit(main())
