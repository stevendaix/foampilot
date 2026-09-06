from __future__ import annotations

from collections import Counter
from typing import Any
import numpy as np
import pyvista as pv


def audit_surface(mesh: pv.PolyData, *, tolerance: float = 1e-9) -> dict[str, Any]:
    """Run deterministic topology/metric checks used by medical_build regression tests."""
    surface = mesh.extract_surface().triangulate().clean(tolerance=tolerance)
    faces = surface.faces.reshape(-1, 4)[:, 1:]
    edges = Counter()
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges[tuple(sorted((int(a), int(b))))] += 1
    edge_hist = Counter(edges.values())
    open_edges = sum(v for k, v in edges.items() if v == 1)
    non_manifold_edges = sum(v for k, v in edges.items() if v > 2)
    canonical = [tuple(sorted(map(int, tri))) for tri in faces]
    duplicates = len(canonical) - len(set(canonical))
    volume = float(abs(surface.volume)) if surface.n_cells else 0.0
    return {
        'points': int(surface.n_points), 'triangles': int(len(faces)),
        'closed': bool(open_edges == 0), 'open_edges': int(open_edges),
        'non_manifold_edges': int(non_manifold_edges), 'duplicate_triangles': int(duplicates),
        'volume': volume, 'area': float(surface.area),
        'edge_histogram': {str(k): int(v) for k, v in sorted(edge_hist.items())},
        'quality_ok': bool(open_edges == 0 and non_manifold_edges == 0 and duplicates == 0 and volume > 0.0),
    }


def audit_file(path: str) -> dict[str, Any]:
    return audit_surface(pv.read(path))
