import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import vtk

logger = logging.getLogger(__name__)


@dataclass
class Centerline:
    points: np.ndarray
    radii: np.ndarray
    abscissas: np.ndarray
    tangents: np.ndarray
    curvature: np.ndarray
    torsion: np.ndarray
    tortuosity: float
    frenet_tangents: np.ndarray
    parallel_transport_normals: np.ndarray
    parallel_transport_binormals: np.ndarray


def compute_centerline_geometry(points: np.ndarray, radii: np.ndarray) -> Centerline:
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 2:
        return Centerline(
            points=pts,
            radii=np.asarray(radii, dtype=float),
            abscissas=np.zeros(n, dtype=float),
            tangents=np.zeros((n, 3), dtype=float),
            curvature=np.zeros(n, dtype=float),
            torsion=np.zeros(n, dtype=float),
            tortuosity=0.0,
            frenet_tangents=np.zeros((n, 3), dtype=float),
            parallel_transport_normals=np.zeros((n, 3), dtype=float),
            parallel_transport_binormals=np.zeros((n, 3), dtype=float),
        )

    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    abscissas = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = abscissas[-1]

    tangents = np.zeros((n, 3), dtype=float)
    tangents[0] = diffs[0] / (seg_lengths[0] + 1e-12)
    tangents[-1] = diffs[-1] / (seg_lengths[-1] + 1e-12)
    for i in range(1, n - 1):
        t = diffs[i - 1] + diffs[i]
        norm = np.linalg.norm(t) + 1e-12
        tangents[i] = t / norm

    curvature = np.zeros(n, dtype=float)
    for i in range(1, n - 1):
        t0 = tangents[i - 1]
        t1 = tangents[i + 1]
        k = np.linalg.norm(t1 - t0) / (0.5 * (abscissas[i + 1] - abscissas[i - 1]) + 1e-12)
        curvature[i] = k

    torsion = np.zeros(n, dtype=float)

    normals = np.zeros((n, 3), dtype=float)
    binormals = np.zeros((n, 3), dtype=float)
    t_prev = tangents[0].copy()
    n_prev = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(t_prev, n_prev)) > 0.9:
        n_prev = np.array([1.0, 0.0, 0.0])
    n_prev = n_prev - np.dot(n_prev, t_prev) * t_prev
    n_norm = np.linalg.norm(n_prev)
    if n_norm > 1e-12:
        n_prev /= n_norm
    else:
        n_prev = np.array([0.0, 1.0, 0.0])
    normals[0] = n_prev
    binormals[0] = np.cross(t_prev, n_prev)

    for i in range(1, n):
        t = tangents[i]
        b = np.cross(t_prev, t)
        b_norm = np.linalg.norm(b)
        if b_norm > 1e-12:
            b /= b_norm
        else:
            b = binormals[i - 1]
        n = np.cross(b, t)
        n_norm = np.linalg.norm(n)
        if n_norm > 1e-12:
            n /= n_norm
        else:
            n = normals[i - 1]
        normals[i] = n
        binormals[i] = b
        t_prev = t

    tortuosity = float(total_length / max(np.linalg.norm(pts[-1] - pts[0]), 1e-12))

    return Centerline(
        points=pts,
        radii=np.asarray(radii, dtype=float),
        abscissas=abscissas,
        tangents=tangents,
        curvature=curvature,
        torsion=torsion,
        tortuosity=tortuosity,
        frenet_tangents=tangents,
        parallel_transport_normals=normals,
        parallel_transport_binormals=binormals,
    )
