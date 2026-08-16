import logging
from typing import List, Optional

import numpy as np
from geomdl import BSpline
from geomdl.knotvector import generate
from scipy.interpolate import splprep

logger = logging.getLogger(__name__)


def _order_contour(points_2d: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_2d, dtype=float)
    if pts.shape[0] < 3:
        raise ValueError("Not enough 2D points to order")
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order = np.argsort(angles)
    return pts[order]


def _resample_contour(points_2d: np.ndarray, n_ctrl: int = 12) -> np.ndarray:
    pts = _order_contour(points_2d)
    pts = np.vstack((pts, pts[0]))
    diffs = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cumulative = np.insert(np.cumsum(seg_len), 0, 0.0)
    total = cumulative[-1]
    if total < 1e-9:
        raise ValueError("Degenerate contour")
    uniform_t = np.linspace(0, total, n_ctrl)
    x = np.interp(uniform_t, cumulative, pts[:, 0])
    y = np.interp(uniform_t, cumulative, pts[:, 1])
    return np.column_stack((x, y))


class BSplineFitter:
    def __init__(self, degree: int = 3, n_ctrl: int = 12, use_least_squares: bool = False):
        self.degree = degree
        self.n_ctrl = n_ctrl
        self.use_least_squares = use_least_squares

    def fit_section(self, section_2d: np.ndarray) -> BSpline.Curve:
        pts = np.asarray(section_2d, dtype=float)
        if pts.shape[0] < 3:
            raise ValueError("Not enough 2D points to fit a B-spline")

        if self.use_least_squares and pts.shape[0] >= 4:
            try:
                return self._fit_least_squares(pts)
            except Exception as exc:
                logger.debug("Least-squares fit failed, falling back to resampling: %s", exc)

        ctrl = _resample_contour(pts, n_ctrl=self.n_ctrl)
        curve = BSpline.Curve()
        curve.degree = self.degree
        curve.ctrlpts = ctrl.tolist()
        curve.knotvector = generate(self.degree, len(ctrl))
        return curve

    def _fit_least_squares(self, pts: np.ndarray) -> BSpline.Curve:
        ordered = _order_contour(pts)
        ordered = np.vstack((ordered, ordered[0]))
        x = ordered[:, 0]
        y = ordered[:, 1]

        n_ctrl = min(self.n_ctrl, len(ordered))
        tck, _ = splprep([x, y], s=0, k=self.degree, nest=n_ctrl + self.degree + 1)

        curve = BSpline.Curve()
        curve.degree = self.degree
        curve.ctrlpts = np.column_stack(tck[1]).tolist()
        curve.knotvector = tck[0].tolist()
        return curve

    def fit_sections(self, sections_2d: List[np.ndarray]) -> List[BSpline.Curve]:
        return [self.fit_section(s) for s in sections_2d]
