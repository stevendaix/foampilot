import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv

from .open_profile import BoundaryRole, OpenProfile

logger = logging.getLogger(__name__)


def _newell_normal(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.array([0.0, 0.0, 1.0])
    normal = np.zeros(3)
    for i in range(len(points)):
        curr = points[i]
        nxt = points[(i + 1) % len(points)]
        normal[0] += (curr[1] - nxt[1]) * (curr[2] + nxt[2])
        normal[1] += (curr[2] - nxt[2]) * (curr[0] + nxt[0])
        normal[2] += (curr[0] - nxt[0]) * (curr[1] + nxt[1])
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return normal / norm


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i, 0], points[i, 1]
        x2, y2 = points[(i + 1) % n, 0], points[(i + 1) % n, 1]
        area += x1 * y2 - x2 * y1
    return 0.5 * abs(area)


def _planarity_score(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    centered = points - points.mean(axis=0)
    _, s, _ = np.linalg.svd(centered)
    if s.size < 3:
        return 0.0
    return float(s[2] / (s[0] + s[1] + s[2] + 1e-12))


def _circularity(area: float, perimeter: float) -> float:
    if perimeter <= 0.0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter * perimeter))


class SurfaceTopologyAnalyzer:
    def __init__(self, mesh: pv.PolyData) -> None:
        self.mesh = mesh
        self._edges: Optional[np.ndarray] = None
        self._boundary_edges: Optional[np.ndarray] = None

    def extract_surface(self) -> pv.PolyData:
        surface = self.mesh.extract_geometry()
        if surface.n_points == 0:
            surface = self.mesh
        cleaner = surface.clean()
        tri = cleaner.triangulate()
        return tri

    def find_boundary_edges(self, surface: Optional[pv.PolyData] = None) -> np.ndarray:
        if surface is None:
            surface = self.extract_surface()
        edges = surface.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        points = edges.points
        lines = edges.lines
        edge_array = []
        i = 0
        while i < len(lines):
            n = lines[i]
            edge_array.append([lines[i + 1], lines[i + 2]])
            i += 3
        self._boundary_edges = np.array(edge_array, dtype=np.int64)
        return self._boundary_edges

    def find_non_manifold_edges(self, surface: Optional[pv.PolyData] = None) -> np.ndarray:
        if surface is None:
            surface = self.extract_surface()
        edges = surface.extract_feature_edges(
            boundary_edges=False,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=True,
        )
        points = edges.points
        lines = edges.lines
        edge_array = []
        i = 0
        while i < len(lines):
            n = lines[i]
            edge_array.append([lines[i + 1], lines[i + 2]])
            i += 3
        return np.array(edge_array, dtype=np.int64)

    def find_connected_components(self, surface: Optional[pv.PolyData] = None) -> List[set]:
        if surface is None:
            surface = self.extract_surface()
        components = []
        visited = set()
        adjacency = {}
        for i in range(surface.n_points):
            adjacency[i] = set()
        faces = surface.faces.reshape(-1, 4)[:, 1:]
        for face in faces:
            for j in range(3):
                a = int(face[j])
                b = int(face[(j + 1) % 3])
                adjacency[a].add(b)
                adjacency[b].add(a)
        for start in range(surface.n_points):
            if start in visited:
                continue
            stack = [start]
            comp = set()
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                comp.add(node)
                for nb in adjacency[node]:
                    if nb not in visited:
                        stack.append(nb)
            components.append(comp)
        return components

    def check_orientation(self, surface: Optional[pv.PolyData] = None) -> Dict[str, Any]:
        if surface is None:
            surface = self.extract_surface()
        faces = surface.faces.reshape(-1, 4)[:, 1:]
        points = surface.points
        normals = surface.point_normals
        inconsistent = 0
        total = len(faces)
        for face in faces:
            face_pts = points[face]
            computed = _newell_normal(face_pts)
            avg_normal = np.mean(normals[face], axis=0)
            avg_norm = np.linalg.norm(avg_normal)
            if avg_norm > 1e-9:
                avg_normal = avg_normal / avg_norm
            else:
                avg_normal = computed
            if np.dot(computed, avg_normal) < 0.0:
                inconsistent += 1
        return {
            "total_faces": total,
            "inconsistent_faces": inconsistent,
            "consistency_ratio": 1.0 - inconsistent / max(total, 1),
        }

    def _chain_boundary_loops(self, boundary_edges: np.ndarray) -> List[np.ndarray]:
        if boundary_edges.size == 0:
            return []
        adjacency = {}
        for edge in boundary_edges:
            a, b = int(edge[0]), int(edge[1])
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
        visited_edges = set()
        loops = []
        for start_edge in boundary_edges:
            a, b = int(start_edge[0]), int(start_edge[1])
            if (a, b) in visited_edges or (b, a) in visited_edges:
                continue
            loop = [a, b]
            visited_edges.add((a, b))
            visited_edges.add((b, a))
            current = b
            while True:
                neighbors = adjacency.get(current, [])
                next_node = None
                for nb in neighbors:
                    if (current, nb) in visited_edges:
                        continue
                    next_node = nb
                    break
                if next_node is None:
                    break
                loop.append(next_node)
                visited_edges.add((current, next_node))
                visited_edges.add((next_node, current))
                current = next_node
                if current == a:
                    break
            if len(loop) >= 3 and loop[0] == loop[-1]:
                loops.append(np.array(loop[:-1], dtype=np.int64))
        return loops

    def find_open_profiles(
        self, surface: Optional[pv.PolyData] = None, centerlines: Optional[np.ndarray] = None
    ) -> List[OpenProfile]:
        if surface is None:
            surface = self.extract_surface()
        boundary_edges = self.find_boundary_edges(surface)
        loops = self._chain_boundary_loops(boundary_edges)
        if not loops:
            logger.info("No boundary loops detected")
            return []
        point_to_faces: Dict[int, set] = {}
        faces = surface.faces.reshape(-1, 4)[:, 1:]
        for face_idx, face in enumerate(faces):
            for pt in face:
                pt = int(pt)
                point_to_faces.setdefault(pt, set()).add(face_idx)
        profiles = []
        for profile_id, loop in enumerate(loops):
            vertex_ids = set(int(v) for v in loop)
            edge_ids = set()
            adjacent_face_ids = set()
            for i in range(len(loop)):
                a = int(loop[i])
                b = int(loop[(i + 1) % len(loop)])
                edge_ids.add((min(a, b), max(a, b)))
            for vid in vertex_ids:
                adjacent_face_ids.update(point_to_faces.get(vid, set()))
            loop_points = surface.points[loop]
            centroid = loop_points.mean(axis=0)
            normal = _newell_normal(loop_points)
            area = _polygon_area(loop_points)
            perimeter = float(np.sum(np.linalg.norm(np.diff(np.vstack([loop_points, loop_points[0]]), axis=0), axis=1)))
            planarity = _planarity_score(loop_points)
            circularity = _circularity(area, perimeter)
            eq_radius = float(np.sqrt(area / np.pi)) if area > 0.0 else 0.0
            profiles.append(
                OpenProfile(
                    id=profile_id,
                    vertex_ids=vertex_ids,
                    edge_ids=edge_ids,
                    adjacent_face_ids=adjacent_face_ids,
                    centroid=centroid,
                    normal=normal,
                    area=area,
                    perimeter=perimeter,
                    equivalent_radius=eq_radius,
                    planarity=planarity,
                    circularity=circularity,
                )
            )
        logger.info("Detected %d open profiles", len(profiles))
        if centerlines is not None and len(centerlines) >= 2 and len(profiles) >= 2:
            self._classify_with_centerlines(profiles, centerlines)
        return profiles

    def _classify_with_centerlines(self, profiles: List[OpenProfile], centerlines: np.ndarray) -> None:
        if len(profiles) < 2 or len(centerlines) < 2:
            return
        endpoints = np.array([centerlines[0], centerlines[-1]])
        centroids = np.array([p.centroid for p in profiles], dtype=float)
        dists = np.linalg.norm(centroids[:, None, :] - endpoints[None, :, :], axis=2)
        closest = np.argmin(dists, axis=1)
        endpoint_idx = np.argsort(np.linalg.norm(endpoints[0] - endpoints[1]))
        for idx, p in enumerate(profiles):
            if closest[idx] == endpoint_idx[0]:
                p.role = BoundaryRole.INLET
                p.confidence = 0.75
                p.metadata.setdefault("classification_method", "centerline")
            elif closest[idx] == endpoint_idx[1]:
                p.role = BoundaryRole.OUTLET
                p.confidence = 0.75
                p.metadata.setdefault("classification_method", "centerline")
            else:
                p.role = BoundaryRole.UNKNOWN
                p.confidence = 0.3
