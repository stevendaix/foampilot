import logging
from typing import Any, List, Optional

import numpy as np
import vtk

logger = logging.getLogger(__name__)


def resample_centerline(points, radii, abscissas, tangents, curvature, torsion, step_length: float = 1.0) -> Any:
    from .vmtkcenterlinegeometry_local import compute_centerline_geometry

    pts = np.asarray(points, dtype=float)
    rad = np.asarray(radii, dtype=float)
    absc = np.asarray(abscissas, dtype=float)
    if len(pts) < 2:
        return compute_centerline_geometry(pts, rad)

    total = absc[-1]
    step = max(step_length, 1e-6)
    n_steps = max(2, int(total / step) + 1)
    new_absc = np.linspace(0, total, n_steps)

    new_pts = np.zeros((n_steps, 3), dtype=float)
    new_pts[0] = pts[0]
    new_pts[-1] = pts[-1]
    for i in range(1, n_steps - 1):
        s = new_absc[i]
        idx = int(np.searchsorted(absc, s))
        idx = min(idx, len(absc) - 1)
        if idx == 0:
            t = 0.0
        else:
            t = (s - absc[idx - 1]) / max(absc[idx] - absc[idx - 1], 1e-12)
        t = max(0.0, min(1.0, t))
        new_pts[i] = (1 - t) * pts[idx - 1] + t * pts[idx]

    new_radii = np.interp(new_absc, absc, rad)
    new_pts = _taubin_smooth(new_pts, n_iter=10, alpha=0.5, beta=0.5)
    new_pts[0] = pts[0]
    new_pts[-1] = pts[-1]

    result = compute_centerline_geometry(new_pts, new_radii)
    logger.info("Resampled centerline: %d points, step=%.3f", n_steps, step)
    return result


def _taubin_smooth(pts: np.ndarray, n_iter: int = 10, alpha: float = 0.5, beta: float = 0.5) -> np.ndarray:
    x = pts.copy()
    n = len(x)
    for _ in range(n_iter):
        for i in range(n):
            neighbors = []
            for j in range(max(0, i - 1), min(n, i + 2)):
                if j != i:
                    neighbors.append(x[j] - x[i])
            if neighbors:
                mean = np.mean(neighbors, axis=0)
                x[i] += alpha * mean
        for i in range(n):
            neighbors = []
            for j in range(max(0, i - 1), min(n, i + 2)):
                if j != i:
                    neighbors.append(x[j] - x[i])
            if neighbors:
                mean = np.mean(neighbors, axis=0)
                x[i] -= beta * mean
    return x
