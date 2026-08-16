import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pyvista as pv

from foampilot.postprocess.openfoam_direct import OpenFOAMDirectReader, CHTDirectReader

logger = logging.getLogger(__name__)

_BC_COLORS: Dict[str, str] = {
    "fixedValue": "#ff0000",
    "zeroGradient": "#0000ff",
    "wall": "#808080",
    "wallFunction": "#ff8000",
    "symmetry": "#00ff00",
    "symmetryPlane": "#00ff00",
    "inletOutlet": "#800080",
    "other": "#ffff00",
}


@dataclass
class PatchInfo:
    name: str
    patch_type: str
    n_faces: int
    start_face: int
    n_cells: int
    bounds: Tuple[float, float, float, float, float, float]
    area: float
    field_bcs: Dict[str, Dict[str, Union[str, List[float]]]]


def _compute_patch_area(face_points: np.ndarray) -> float:
    if len(face_points) < 3:
        return 0.0
    area = 0.0
    for i in range(1, len(face_points) - 1):
        v1 = face_points[i] - face_points[0]
        v2 = face_points[i + 1] - face_points[0]
        cross = np.cross(v1, v2)
        area += 0.5 * np.linalg.norm(cross)
    return float(area)


def _compute_bounds(points: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    xmin, ymin, zmin = points.min(axis=0)
    xmax, ymax, zmax = points.max(axis=0)
    return (float(xmin), float(xmax), float(ymin), float(ymax), float(zmin), float(zmax))


def _classify_patch_type(patch_type: str) -> str:
    normalized = patch_type.lower()
    if "patch" in normalized:
        return "patch"
    if "wall" in normalized:
        return "wall"
    if "symmetry" in normalized:
        return "symmetry"
    if "cyclic" in normalized:
        return "cyclic"
    if "processor" in normalized:
        return "processor"
    if "empty" in normalized:
        return "empty"
    return "other"


def _classify_field_bc_type(bc_type: str) -> str:
    normalized = bc_type.lower()
    if "fixedvalue" in normalized:
        return "fixedValue"
    if "zerogradient" in normalized:
        return "zeroGradient"
    if "wall" in normalized:
        if "function" in normalized:
            return "wallFunction"
        return "wall"
    if "symmetry" in normalized:
        return "symmetry"
    if "inletoutlet" in normalized:
        return "inletOutlet"
    if "calculated" in normalized:
        return "calculated"
    return "other"


def _read_field_boundary_conditions(filepath: Path) -> Dict[str, Dict[str, str]]:
    """Parse boundaryField section from an OpenFOAM field file."""
    with open(filepath, "r") as f:
        lines = f.read().split("\n")
    boundary_bcs: Dict[str, Dict[str, str]] = {}
    in_bf = False
    current_bc = None
    in_bc = False
    for line in lines:
        stripped = line.strip()
        if stripped == "boundaryField":
            in_bf = True
            continue
        if not in_bf:
            continue
        if stripped == "}":
            break
        if "{" in stripped and not stripped.startswith("type"):
            current_bc = stripped.split("{")[0].strip()
            in_bc = True
            boundary_bcs[current_bc] = {}
            continue
        if in_bc and "}" in stripped:
            current_bc = None
            in_bc = False
            continue
        if in_bc and ";" in stripped:
            parts = stripped.split(";")[0].strip().split()
            if len(parts) >= 2:
                boundary_bcs[current_bc][parts[0]] = parts[1]
    return boundary_bcs


def _build_polydata_from_faces(
    points: np.ndarray,
    faces: List[np.ndarray],
    face_indices: np.ndarray,
) -> pv.PolyData:
    """Build a PyVista PolyData directly from OpenFOAM face indices."""
    all_point_indices = []
    for fi in face_indices:
        if 0 <= fi < len(faces):
            all_point_indices.extend(faces[fi])
    unique_indices = np.unique(all_point_indices)
    point_map = {old: new for new, old in enumerate(unique_indices)}
    new_points = points[unique_indices]
    faces_connectivity = []
    for fi in face_indices:
        if 0 <= fi < len(faces):
            face_pts = faces[fi]
            n = len(face_pts)
            faces_connectivity.append(n)
            faces_connectivity.extend(point_map[int(p)] for p in face_pts)
    return pv.PolyData(new_points, faces_connectivity)


class BoundaryViewer:
    def __init__(self, reader: OpenFOAMDirectReader, fields: Optional[List[str]] = None):
        self.reader = reader
        self.fields = fields or []

    def list_patches(self) -> List[str]:
        """Return sorted list of boundary patch names."""
        self._ensure_mesh_loaded()
        return sorted(self.reader.boundary_patches.keys())

    def get_patch_faces(self, name: str) -> np.ndarray:
        """Return the OpenFOAM face indices belonging to a patch.

        Parameters
        ----------
        name : str
            Patch name.

        Returns
        -------
        np.ndarray
            Face indices for the patch.
        """
        self._ensure_mesh_loaded()
        patches = self.reader.boundary_patches
        if name not in patches:
            raise KeyError(f"Patch '{name}' not found")
        info = patches[name]
        start_face = int(info.get("startFace", 0))
        n_faces = int(info.get("nFaces", 0))
        return np.arange(start_face, start_face + n_faces, dtype=int)

    def get_patch_info(self, name: str) -> PatchInfo:
        """Return geometry and metadata for a boundary patch.

        Parameters
        ----------
        name : str
            Patch name.

        Returns
        -------
        PatchInfo
            Structured patch information.
        """
        self._ensure_mesh_loaded()
        patches = self.reader.boundary_patches
        if name not in patches:
            raise KeyError(f"Patch '{name}' not found")
        info = patches[name]
        patch_type = info.get("type", "unknown")
        n_faces = int(info.get("nFaces", 0))
        start_face = int(info.get("startFace", 0))
        face_indices = np.arange(start_face, start_face + n_faces, dtype=int)
        owner = self.reader._owner
        cell_indices = owner[face_indices]
        n_cells = int(np.unique(cell_indices).size)
        face_pts = self.reader._faces
        points = self.reader._points
        patch_points = []
        total_area = 0.0
        for fi in face_indices:
            if 0 <= fi < len(face_pts):
                pts = points[face_pts[fi]]
                patch_points.append(pts)
                total_area += _compute_patch_area(pts)
        if patch_points:
            all_patch_pts = np.vstack(patch_points)
            bounds = _compute_bounds(all_patch_pts)
        else:
            bounds = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        field_bcs: Dict[str, Dict[str, Union[str, List[float]]]] = {}
        for field_name in self.fields:
            try:
                time_dirs = self.reader.get_time_steps()
                time_step = time_dirs[-1] if time_dirs else "0"
                region_dir = self.reader.region
                search_paths = []
                if region_dir:
                    search_paths.append(self.reader.case_path / time_step / region_dir / field_name)
                search_paths.append(self.reader.case_path / time_step / field_name)
                field_path = None
                for p in search_paths:
                    if p.exists():
                        field_path = p
                        break
                    gz_path = p.with_suffix(p.suffix + ".gz")
                    if gz_path.exists():
                        field_path = gz_path
                        break
                if field_path and field_path.exists():
                    boundary_bcs = _read_field_boundary_conditions(field_path)
                    if name in boundary_bcs:
                        bc = boundary_bcs[name]
                        bc_type = bc.get("type", "unknown")
                        bc_info: Dict[str, Union[str, List[float]]] = {
                            "type": _classify_field_bc_type(bc_type),
                        }
                        if "value" in bc:
                            val_str = bc["value"]
                            if val_str.startswith("(") and val_str.endswith(")"):
                                vals = [float(v) for v in val_str[1:-1].split()]
                                bc_info["value"] = vals
                            else:
                                try:
                                    bc_info["value"] = [float(val_str)]
                                except ValueError:
                                    bc_info["value"] = val_str
                        field_bcs[field_name] = bc_info
            except Exception:
                pass
        return PatchInfo(
            name=name,
            patch_type=patch_type,
            n_faces=n_faces,
            start_face=start_face,
            n_cells=n_cells,
            bounds=bounds,
            area=total_area,
            field_bcs=field_bcs,
        )

    def get_patch_mesh(self, name: str) -> pv.PolyData:
        """Extract the surface mesh for a single boundary patch.

        Parameters
        ----------
        name : str
            Patch name.

        Returns
        -------
        pv.PolyData
            Surface mesh for the patch.
        """
        self._ensure_mesh_loaded()
        face_indices = self.get_patch_faces(name)
        if len(face_indices) == 0:
            empty = pv.PolyData()
            empty["patch"] = [name]
            return empty
        return _build_polydata_from_faces(
            self.reader._points, self.reader._faces, face_indices
        )

    def get_boundary_only(self) -> pv.PolyData:
        """Extract all boundary faces as a single surface mesh.

        Returns
        -------
        pv.PolyData
            Surface mesh containing all boundary faces with
            ``patch_id`` and ``patch_name`` in ``cell_data``.
        """
        self._ensure_mesh_loaded()
        patches = self.reader.boundary_patches
        all_faces = []
        patch_ids = []
        patch_names = []
        for idx, (name, info) in enumerate(patches.items()):
            start_face = int(info.get("startFace", 0))
            n_faces = int(info.get("nFaces", 0))
            face_indices = np.arange(start_face, start_face + n_faces, dtype=int)
            all_faces.extend(face_indices)
            patch_ids.extend([idx] * n_faces)
            patch_names.extend([name] * n_faces)
        if not all_faces:
            return pv.PolyData()
        surface = _build_polydata_from_faces(
            self.reader._points, self.reader._faces, np.array(all_faces, dtype=int)
        )
        surface.cell_data["patch_id"] = np.array(patch_ids, dtype=int)
        surface.cell_data["patch_name"] = patch_names
        return surface

    def get_bc_type_mesh(self) -> pv.PolyData:
        """Extract boundary faces colored by boundary condition type.

        Returns
        -------
        pv.PolyData
            Surface mesh with ``bc_type_id`` in ``cell_data``.
        """
        self._ensure_mesh_loaded()
        patches = self.reader.boundary_patches
        all_faces = []
        bc_type_ids = []
        bc_type_names = []
        for idx, (name, info) in enumerate(patches.items()):
            start_face = int(info.get("startFace", 0))
            n_faces = int(info.get("nFaces", 0))
            face_indices = np.arange(start_face, start_face + n_faces, dtype=int)
            all_faces.extend(face_indices)
            bc_type = _classify_field_bc_type(info.get("type", "other"))
            bc_type_id = list(_BC_COLORS.keys()).index(bc_type) if bc_type in _BC_COLORS else len(_BC_COLORS) - 1
            bc_type_ids.extend([bc_type_id] * n_faces)
            bc_type_names.extend([bc_type] * n_faces)
        if not all_faces:
            return pv.PolyData()
        surface = _build_polydata_from_faces(
            self.reader._points, self.reader._faces, np.array(all_faces, dtype=int)
        )
        surface.cell_data["bc_type_id"] = np.array(bc_type_ids, dtype=int)
        surface.cell_data["bc_type_name"] = bc_type_names
        return surface

    def _ensure_mesh_loaded(self) -> None:
        _ = self.reader.mesh

    def plot(
        self,
        off_screen: Optional[bool] = None,
        screenshot: Optional[str] = None,
    ) -> pv.Plotter:
        """Build a 2x2 PyVista dashboard for boundary inspection.

        Parameters
        ----------
        off_screen : bool, optional
            Render off-screen.
        screenshot : str, optional
            Path to save screenshot.

        Returns
        -------
        pv.Plotter
            The plotter instance.
        """
        self._ensure_mesh_loaded()
        plotter = pv.Plotter(shape=(2, 2), off_screen=off_screen or False)
        plotter.set_background("white")
        plotter.subplot(0, 0)
        plotter.add_text("3D Global View", font_size=10)
        mesh = self.reader.mesh
        plotter.add_mesh(
            mesh.extract_surface(),
            color="lightgray",
            opacity=0.3,
            show_edges=True,
            edge_color="black",
        )
        patches = self.reader.boundary_patches
        patch_list = list(patches.keys())
        for idx, name in enumerate(patch_list):
            try:
                patch_mesh = self.get_patch_mesh(name)
                color = _BC_COLORS.get(_classify_field_bc_type(patches[name].get("type", "other")), "lightblue")
                plotter.add_mesh(patch_mesh, color=color, show_edges=False)
            except Exception:
                pass
        plotter.subplot(0, 1)
        plotter.add_text("Boundary Only", font_size=10)
        boundary_mesh = self.get_boundary_only()
        plotter.add_mesh(boundary_mesh, color="lightblue", show_edges=True)
        plotter.subplot(1, 0)
        plotter.add_text("BC Types", font_size=10)
        bc_mesh = self.get_bc_type_mesh()
        if bc_mesh.n_cells > 0 and "bc_type_name" in bc_mesh.cell_data:
            bc_names = bc_mesh.cell_data["bc_type_name"]
            for bc_type, color in _BC_COLORS.items():
                mask = bc_names == bc_type
                if np.any(mask):
                    plotter.add_mesh(
                        bc_mesh.extract_cells(np.where(mask)[0]),
                        color=color,
                        show_edges=True,
                        label=bc_type,
                    )
        else:
            plotter.add_mesh(bc_mesh, color="lightgray", show_edges=True)
        plotter.subplot(1, 1)
        plotter.add_text("Patches", font_size=10)
        y_offset = 0.9
        for name in self.list_patches():
            info = self.reader.boundary_patches.get(name, {})
            bc_type = info.get("type", "unknown")
            text = f"{name}: {bc_type}"
            plotter.add_text(
                text,
                position=(0.05, y_offset),
                font_size=8,
                viewport=True,
            )
            y_offset -= 0.05
        plotter.link_views()
        if screenshot:
            plotter.screenshot(screenshot)
            logger.info("Screenshot saved to %s", screenshot)
        else:
            plotter.show()
        return plotter
