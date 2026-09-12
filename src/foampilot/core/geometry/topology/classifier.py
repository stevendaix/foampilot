import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .open_profile import BoundaryRole, OpenProfile

logger = logging.getLogger(__name__)


class OpenProfileClassifier:
    def __init__(self) -> None:
        self._weights = {
            "area": 0.15,
            "circularity": 0.15,
            "planarity": 0.2,
            "normal_alignment": 0.25,
            "position": 0.25,
        }

    def classify(
        self,
        profiles: List[OpenProfile],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[OpenProfile]:
        context = context or {}
        if len(profiles) == 0:
            return profiles
        if len(profiles) == 1:
            p = profiles[0]
            p.role = BoundaryRole.INLET
            p.confidence = 0.6
            return profiles
        axis = context.get("axis")
        origin = context.get("origin")
        if axis is not None and origin is not None:
            return self.classify_by_axis(profiles, axis, origin)
        if len(profiles) == 2:
            return self.classify_by_area(profiles)
        return self.classify_by_area(profiles)

    def classify_by_area(self, profiles: List[OpenProfile]) -> List[OpenProfile]:
        if not profiles:
            return profiles
        sorted_profiles = sorted(profiles, key=lambda p: p.area, reverse=True)
        total_area = sum(p.area for p in profiles)
        for p in profiles:
            if p.area <= 0:
                p.role = BoundaryRole.UNKNOWN
                p.confidence = 0.0
                continue
            area_ratio = p.area / total_area
            if p is sorted_profiles[0]:
                p.role = BoundaryRole.INLET
                p.confidence = min(0.3 + area_ratio * 0.5 + p.circularity * 0.2, 1.0)
            else:
                p.role = BoundaryRole.OUTLET
                p.confidence = min(0.2 + area_ratio * 0.4 + p.circularity * 0.2, 0.9)
            if p.area < 1e-6:
                p.confidence *= 0.3
            if p.planarity < 0.1:
                p.confidence *= 0.5
        return profiles

    def classify_by_axis(
        self,
        profiles: List[OpenProfile],
        axis: np.ndarray,
        origin: np.ndarray,
    ) -> List[OpenProfile]:
        if not profiles:
            return profiles
        axis = np.asarray(axis, dtype=float)
        origin = np.asarray(origin, dtype=float)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        projections = []
        for p in profiles:
            rel = p.centroid - origin
            projections.append(float(np.dot(rel, axis)))
        if not projections:
            return profiles
        min_idx = int(np.argmin(projections))
        max_idx = int(np.argmax(projections))
        for idx, p in enumerate(profiles):
            if idx == min_idx:
                p.role = BoundaryRole.INLET
                p.confidence = 0.7
            elif idx == max_idx:
                p.role = BoundaryRole.OUTLET
                p.confidence = 0.7
            else:
                p.role = BoundaryRole.UNKNOWN
                p.confidence = 0.3
        return profiles

    def request_user_selection(self, profiles: List[OpenProfile]) -> List[OpenProfile]:
        if not profiles:
            return profiles
        print(f"Detected {len(profiles)} open profiles:")
        for p in profiles:
            print(
                f"  [{p.id}] area={p.area:.6f}, centroid={p.centroid}, "
                f"normal={p.normal}, circularity={p.circularity:.3f}, "
                f"planarity={p.planarity:.3f}"
            )
        if len(profiles) == 2:
            profiles[0].role = BoundaryRole.INLET
            profiles[0].confidence = 0.9
            profiles[1].role = BoundaryRole.OUTLET
            profiles[1].confidence = 0.9
        else:
            for p in profiles:
                p.role = BoundaryRole.UNKNOWN
                p.confidence = 0.0
        return profiles
