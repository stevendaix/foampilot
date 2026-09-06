import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    prange = range


def circumscribed_sphere_numpy(tetra_points: np.ndarray) -> Tuple[np.ndarray, float]:
    tets = np.asarray(tetra_points, dtype=float)
    n = len(tets)
    if n == 0:
        return np.zeros(3), 0.0
    centers = np.zeros((n, 3), dtype=float)
    radii = np.zeros(n, dtype=float)
    for k in range(n):
        p = tets[k]
        if len(p) < 4:
            continue
        p0, p1, p2, p3 = p[0], p[1], p[2], p[3]
        A = 2.0 * np.vstack([p1 - p0, p2 - p0, p3 - p0])
        b = np.array([
            np.dot(p1, p1) - np.dot(p0, p0),
            np.dot(p2, p2) - np.dot(p0, p0),
            np.dot(p3, p3) - np.dot(p0, p0),
        ])
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            c = p.mean(axis=0)
        centers[k] = c
        radii[k] = float(np.max(np.linalg.norm(p - c, axis=1)))
    return centers, radii


if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def circumscribed_sphere_numba(tetra_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tets = tetra_points
        n = len(tets)
        centers = np.zeros((n, 3), dtype=np.float64)
        radii = np.zeros(n, dtype=np.float64)
        for k in prange(n):
            p = tets[k]
            if len(p) < 4:
                continue
            p0 = p[0]
            p1 = p[1]
            p2 = p[2]
            p3 = p[3]
            A = 2.0 * np.vstack((p1 - p0, p2 - p0, p3 - p0))
            b = np.array([
                np.dot(p1, p1) - np.dot(p0, p0),
                np.dot(p2, p2) - np.dot(p0, p0),
                np.dot(p3, p3) - np.dot(p0, p0),
            ])
            try:
                c = np.linalg.solve(A, b)
            except Exception:
                c = np.mean(p, axis=0)
            centers[k] = c
            r = 0.0
            for j in range(4):
                d = np.sqrt(np.sum((p[j] - c) ** 2))
                if d > r:
                    r = d
            radii[k] = r
        return centers, radii
else:
    circumscribed_sphere_numba = None


def voxel_mask_sample_numpy(mask: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return mask[tuple(indices.T)]


if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def voxel_mask_sample_numba(mask: np.ndarray, indices: np.ndarray) -> np.ndarray:
        n = indices.shape[0]
        result = np.zeros(n, dtype=np.bool_)
        for i in prange(n):
            idx = indices[i]
            result[i] = mask[idx[0], idx[1], idx[2]]
        return result
else:
    voxel_mask_sample_numba = None


def prepare_delaunay_faces_numpy(tets: np.ndarray) -> np.ndarray:
    tets_sorted = np.sort(tets, axis=1)
    faces = np.stack([
        tets_sorted[:, [0, 1, 2]],
        tets_sorted[:, [0, 1, 3]],
        tets_sorted[:, [0, 2, 3]],
        tets_sorted[:, [1, 2, 3]],
    ], axis=1).reshape(-1, 3)
    return np.sort(faces, axis=1)


if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def prepare_delaunay_faces_numba(tets: np.ndarray) -> np.ndarray:
        n = tets.shape[0]
        faces = np.empty((n * 4, 3), dtype=np.int64)
        for k in prange(n):
            t = np.sort(tets[k])
            faces[k * 4] = np.sort(np.array([t[0], t[1], t[2]]))
            faces[k * 4 + 1] = np.sort(np.array([t[0], t[1], t[3]]))
            faces[k * 4 + 2] = np.sort(np.array([t[0], t[2], t[3]]))
            faces[k * 4 + 3] = np.sort(np.array([t[1], t[2], t[3]]))
        return faces
else:
    prepare_delaunay_faces_numba = None


def edge_costs_numpy(points: np.ndarray, radii: np.ndarray, edges: np.ndarray, radius_floor: float = 1e-6) -> np.ndarray:
    xi = np.array([-0.7745966692, 0.0, 0.7745966692])
    wi = np.array([0.5555555556, 0.8888888889, 0.5555555556])
    result = np.empty(edges.shape[0], dtype=np.float64)
    for k in range(edges.shape[0]):
        i, j = edges[k]
        p0, p1 = points[i], points[j]
        r0, r1 = radii[i], radii[j]
        length = np.linalg.norm(p1 - p0)
        total = 0.0
        for x, w in zip(xi, wi):
            a = 0.5 * (x + 1.0)
            r = (1.0 - a) * r0 + a * r1
            total += w / max(r, radius_floor)
        result[k] = 0.5 * length * total
    return result


if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def edge_costs_numba(points: np.ndarray, radii: np.ndarray, edges: np.ndarray, radius_floor: float = 1e-6) -> np.ndarray:
        xi = np.array([-0.7745966692, 0.0, 0.7745966692])
        wi = np.array([0.5555555556, 0.8888888889, 0.5555555556])
        result = np.empty(edges.shape[0], dtype=np.float64)
        for k in prange(edges.shape[0]):
            i, j = edges[k]
            p0, p1 = points[i], points[j]
            r0, r1 = radii[i], radii[j]
            length = np.sqrt(np.sum((p1 - p0) ** 2))
            total = 0.0
            for idx in range(3):
                a = 0.5 * (xi[idx] + 1.0)
                r = (1.0 - a) * r0 + a * r1
                total += wi[idx] / max(r, radius_floor)
            result[k] = 0.5 * length * total
        return result
else:
    edge_costs_numba = None
