import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import vtk

from .pypes import vmtkBaseScript

logger = logging.getLogger(__name__)


@dataclass
class SurfaceModel:
    polydata: vtk.vtkPolyData
    quality_report: Dict[str, Any] = field(default_factory=dict)
    num_points: int = 0
    num_triangles: int = 0
    boundary_edges: int = 0
    non_manifold_edges: int = 0
    connected_components: int = 0
    signed_volume: float = 0.0
    bbox_diameter: float = 0.0


def read_polydata(path) -> vtk.vtkPolyData:
    path = str(path)
    if path.lower().endswith(".stl"):
        reader = vtk.vtkSTLReader()
    elif path.lower().endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()
    elif path.lower().endswith(".ply"):
        reader = vtk.vtkPLYReader()
    else:
        raise ValueError(f"Surface inconnue: {path}")
    reader.SetFileName(path)
    reader.Update()
    result = vtk.vtkPolyData()
    result.DeepCopy(reader.GetOutput())
    return result


def mha_to_surface(path, threshold) -> vtk.vtkPolyData:
    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(str(path))
    reader.Update()

    binary = vtk.vtkImageThreshold()
    binary.SetInputConnection(reader.GetOutputPort())
    binary.ThresholdBetween(float(threshold), 1.0e12)
    binary.SetInValue(1)
    binary.SetOutValue(0)
    binary.SetOutputScalarTypeToUnsignedChar()

    cubes = vtk.vtkFlyingEdges3D()
    cubes.SetInputConnection(binary.GetOutputPort())
    cubes.SetValue(0, 0.5)
    cubes.Update()

    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(cubes.GetOutputPort())
    triangles.Update()
    return triangles.GetOutput()


def _vtk_to_numpy(polydata: vtk.vtkPolyData) -> Tuple[np.ndarray, np.ndarray]:
    n_pts = polydata.GetNumberOfPoints()
    points = np.array([polydata.GetPoint(i) for i in range(n_pts)], dtype=np.float64)

    faces = []
    polys = polydata.GetPolys()
    polys.InitTraversal()
    pt_ids = vtk.vtkIdList()
    while polys.GetNextCell(pt_ids):
        if pt_ids.GetNumberOfIds() >= 3:
            faces.append([pt_ids.GetId(0), pt_ids.GetId(1), pt_ids.GetId(2)])
    return points, np.array(faces, dtype=np.int64)


def _chain_boundary_loops(boundary_edges: np.ndarray, points: Optional[np.ndarray] = None, merge_tol: float = 1e-6) -> List[np.ndarray]:
    if boundary_edges.size == 0:
        return []

    if points is not None:
        node_map = {}
        unique_nodes = []
        for i in range(len(points)):
            key = tuple(np.round(points[i] / merge_tol).astype(int))
            if key not in node_map:
                node_map[key] = len(unique_nodes)
                unique_nodes.append(i)
        edges = [(node_map[tuple(np.round(points[a] / merge_tol).astype(int))],
                  node_map[tuple(np.round(points[b] / merge_tol).astype(int))])
                 for a, b in boundary_edges]
    else:
        edges = [(int(a), int(b)) for a, b in boundary_edges]

    adjacency: Dict[int, List[int]] = {}
    for a, b in edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    visited_edges = set()
    loops = []
    for start_edge in edges:
        a, b = start_edge
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


def _compute_quality_report(polydata: vtk.vtkPolyData) -> Dict[str, any]:
    report: Dict[str, any] = {}

    n_pts = polydata.GetNumberOfPoints()
    n_cells = polydata.GetNumberOfCells()

    faces = []
    polys = polydata.GetPolys()
    polys.InitTraversal()
    pt_ids = vtk.vtkIdList()
    while polys.GetNextCell(pt_ids):
        if pt_ids.GetNumberOfIds() >= 3:
            faces.append([pt_ids.GetId(0), pt_ids.GetId(1), pt_ids.GetId(2)])
    faces_arr = np.array(faces, dtype=np.int64)

    report["num_points"] = n_pts
    report["num_triangles"] = len(faces_arr)

    boundary_edges_filter = vtk.vtkFeatureEdges()
    boundary_edges_filter.SetInputData(polydata)
    boundary_edges_filter.BoundaryEdgesOn()
    boundary_edges_filter.FeatureEdgesOff()
    boundary_edges_filter.ManifoldEdgesOff()
    boundary_edges_filter.NonManifoldEdgesOff()
    boundary_edges_filter.Update()
    b_output = boundary_edges_filter.GetOutput()
    b_lines = b_output.GetLines()
    boundary_count = 0
    if b_lines is not None:
        b_lines.InitTraversal()
        b_id_list = vtk.vtkIdList()
        while b_lines.GetNextCell(b_id_list):
            if b_id_list.GetNumberOfIds() == 2:
                boundary_count += 1
    report["boundary_edges"] = boundary_count

    non_manifold_filter = vtk.vtkFeatureEdges()
    non_manifold_filter.SetInputData(polydata)
    non_manifold_filter.BoundaryEdgesOff()
    non_manifold_filter.FeatureEdgesOff()
    non_manifold_filter.ManifoldEdgesOff()
    non_manifold_filter.NonManifoldEdgesOn()
    non_manifold_filter.Update()
    nm_output = non_manifold_filter.GetOutput()
    nm_lines = nm_output.GetLines()
    nm_count = 0
    if nm_lines is not None:
        nm_lines.InitTraversal()
        nm_id_list = vtk.vtkIdList()
        while nm_lines.GetNextCell(nm_id_list):
            if nm_id_list.GetNumberOfIds() == 2:
                nm_count += 1
    report["non_manifold_edges"] = nm_count

    if len(faces_arr) > 0:
        adjacency: Dict[int, set] = {}
        for i in range(n_pts):
            adjacency[i] = set()
        for face in faces_arr:
            for j in range(3):
                a = int(face[j])
                b = int(face[(j + 1) % 3])
                adjacency[a].add(b)
                adjacency[b].add(a)
        visited = set()
        components = 0
        for start in range(n_pts):
            if start in visited:
                continue
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                for nb in adjacency[node]:
                    if nb not in visited:
                        stack.append(nb)
            components += 1
        report["connected_components"] = components
    else:
        report["connected_components"] = 0

    if len(faces_arr) > 0:
        pts = np.array([polydata.GetPoint(i) for i in range(n_pts)], dtype=np.float64)
        tri_pts = pts[faces_arr]
        vols = np.sum(tri_pts[:, 0] * np.cross(tri_pts[:, 1], tri_pts[:, 2]), axis=1)
        report["signed_volume"] = float(np.sum(vols)) / 6.0
    else:
        report["signed_volume"] = 0.0

    if n_pts > 0:
        bbox_min = np.array([polydata.GetBounds()[i] for i in range(0, 6, 2)])
        bbox_max = np.array([polydata.GetBounds()[i] for i in range(1, 6, 2)])
        diag = np.linalg.norm(bbox_max - bbox_min)
        report["bbox_diameter"] = float(diag)
    else:
        report["bbox_diameter"] = 0.0

    return report


def preprocess_surface(surface, smooth: bool = False, flip_normals: bool = False) -> SurfaceModel:
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(surface)
    clean.Update()

    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(clean.GetOutputPort())
    triangles.PassLinesOff()
    triangles.PassVertsOff()
    triangles.Update()

    current = triangles.GetOutput()
    if smooth:
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(current)
        smoother.SetNumberOfIterations(10)
        smoother.SetPassBand(0.08)
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOff()
        smoother.Update()
        current = smoother.GetOutput()

    copy = vtk.vtkPolyData()
    copy.ShallowCopy(current)
    copy.GetPointData().SetNormals(None)
    copy.GetCellData().SetNormals(None)

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(copy)
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.ComputePointNormalsOn()
    normals.SetFlipNormals(bool(flip_normals))
    normals.Update()

    result = vtk.vtkPolyData()
    result.DeepCopy(normals.GetOutput())

    quality_report = _compute_quality_report(result)
    model = SurfaceModel(
        polydata=result,
        quality_report=quality_report,
        num_points=quality_report["num_points"],
        num_triangles=quality_report["num_triangles"],
        boundary_edges=quality_report["boundary_edges"],
        non_manifold_edges=quality_report["non_manifold_edges"],
        connected_components=quality_report["connected_components"],
        signed_volume=quality_report["signed_volume"],
        bbox_diameter=quality_report["bbox_diameter"],
    )
    return model


class vmtkSurfacePreprocess(vmtkBaseScript):
    def __init__(self):
        super().__init__()
        self.Surface: Optional[vtk.vtkPolyData] = None
        self.InputFileName: str = ""
        self.Smooth: bool = False
        self.FlipNormals: bool = False
        self.Output: Optional[SurfaceModel] = None

    def Execute(self):
        if self.Surface is None and self.InputFileName:
            self.Surface = read_polydata(self.InputFileName)
        if self.Surface is None:
            self.PrintError("Error: No input surface.")
            return
        self.Output = preprocess_surface(self.Surface, smooth=self.Smooth, flip_normals=self.FlipNormals)
        self.PrintLog(
            f"Preprocessed surface: {self.Output.num_points} points, "
            f"{self.Output.num_triangles} triangles, "
            f"{self.Output.boundary_edges} boundary edges, "
            f"{self.Output.non_manifold_edges} non-manifold edges, "
            f"{self.Output.connected_components} components, "
            f"volume={self.Output.signed_volume:.3f}, "
            f"bbox_diameter={self.Output.bbox_diameter:.3f}"
        )
