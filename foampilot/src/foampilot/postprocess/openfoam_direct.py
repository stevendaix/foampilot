import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pyvista as pv

logger = logging.getLogger(__name__)


def _parse_foam_header(lines: list) -> dict:
    header = {}
    in_header = False
    for line in lines:
        stripped = line.strip()
        if stripped == "FoamFile":
            in_header = True
            continue
        if in_header:
            if stripped == "}":
                break
            if "{" in stripped and not stripped.startswith("//"):
                key = stripped.split("{")[0].strip()
                header[key] = {}
                current = key
                continue
            if ";" in stripped:
                parts = stripped.split(";")
                val = parts[0].strip().strip('"')
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                if current in header:
                    header[current] = val
                else:
                    for k in list(header.keys()):
                        if isinstance(header[k], dict):
                            header[k][stripped.split(";")[0].strip()] = val
    return header


def _read_points(filepath: Path) -> np.ndarray:
    with open(filepath, "r") as f:
        content = f.read()
    lines = content.split("\n")
    n_points = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.isdigit():
            n_points = int(stripped)
            data_start = i + 1
            break
    if n_points is None:
        raise ValueError(f"Could not find number of points in {filepath}")
    data_lines = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if stripped == ")":
            break
        if stripped and not stripped.startswith("//"):
            data_lines.append(stripped)
    data_str = " ".join(data_lines)
    data_str = data_str.replace("(", "").replace(")", "")
    values = [float(x) for x in data_str.split()]
    points = np.array(values, dtype=float).reshape(-1, 3)
    return points


def _read_faces(filepath: Path) -> List[np.ndarray]:
    with open(filepath, "r") as f:
        content = f.read()
    lines = content.split("\n")
    n_faces = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.isdigit():
            n_faces = int(stripped)
            data_start = i + 1
            break
    if n_faces is None:
        raise ValueError(f"Could not find number of faces in {filepath}")
    faces = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if stripped == ")":
            break
        if not stripped or stripped.startswith("//"):
            continue
        if stripped.startswith("("):
            stripped = stripped[1:]
        if stripped.endswith(")"):
            stripped = stripped[:-1]
        stripped = stripped.strip()
        if not stripped:
            continue
        lparen = stripped.find("(")
        if lparen > 0:
            n_pts = int(stripped[:lparen])
            pts_str = stripped[lparen + 1:]
        else:
            parts = stripped.split(None, 1)
            n_pts = int(parts[0])
            pts_str = parts[1] if len(parts) > 1 else ""
        pts = [int(x) for x in pts_str.split()]
        faces.append(np.array(pts, dtype=int))
    return faces


def _read_label_list(filepath: Path) -> np.ndarray:
    with open(filepath, "r") as f:
        content = f.read()
    lines = content.split("\n")
    n_vals = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.isdigit():
            n_vals = int(stripped)
            data_start = i + 1
            break
    if n_vals is None:
        raise ValueError(f"Could not find number of values in {filepath}")
    vals = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if stripped == ")":
            break
        if stripped and not stripped.startswith("//"):
            inner = stripped.strip("()")
            for v in inner.split():
                vals.append(int(v))
    return np.array(vals, dtype=int)


def _read_boundary(filepath: Path) -> Dict[str, dict]:
    with open(filepath, "r") as f:
        content = f.read()
    lines = content.split("\n")
    patches = {}
    current_patch = None
    in_patches = False
    for line in lines:
        stripped = line.strip()
        if stripped == ")":
            break
        if not stripped or stripped.startswith("//"):
            continue
        if stripped == "(":
            in_patches = True
            continue
        if not in_patches:
            continue
        if stripped == "}":
            current_patch = None
            continue
        if stripped == "{":
            if current_patch is not None and current_patch not in patches:
                patches[current_patch] = {}
            continue
        if "{" in stripped:
            current_patch = stripped.split("{")[0].strip()
            continue
        if current_patch is None and ";" not in stripped:
            current_patch = stripped
            continue
        if current_patch and ";" in stripped:
            key_val = stripped.split(";")[0].strip()
            if key_val:
                parts = key_val.split()
                if len(parts) >= 2:
                    key = parts[0]
                    val = parts[1]
                    try:
                        val = int(val)
                    except ValueError:
                        pass
                    patches[current_patch][key] = val
    return patches


def _read_field(filepath: Path) -> Tuple[np.ndarray, bool]:
    with open(filepath, "r") as f:
        content = f.read()
    lines = content.split("\n")
    header = {}
    in_header = False
    current_header_key = None
    for line in lines:
        stripped = line.strip()
        if stripped == "FoamFile":
            in_header = True
            current_header_key = "FoamFile"
            continue
        if in_header:
            if stripped == "}":
                break
            if "{" in stripped:
                key = stripped.split("{")[0].strip()
                if not key and current_header_key:
                    key = current_header_key
                header[key] = {}
                current_header_key = key
                continue
            if ";" in stripped:
                parts = stripped.split(";")
                raw_val = parts[0].strip()
                key_val = raw_val.split(None, 1)
                key = key_val[0] if key_val else raw_val
                val = key_val[1] if len(key_val) > 1 else raw_val
                val = val.strip().strip('"')
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                assigned = False
                for k in list(header.keys()):
                    if isinstance(header[k], dict) and not assigned:
                        header[k][key] = val
                        assigned = True
                if not assigned:
                    header[key] = val

    field_class = header.get("FoamFile", {}).get("class", header.get("class", ""))
    is_point_field = field_class.startswith("point")
    is_vector = "Vector" in field_class
    is_scalar = "Scalar" in field_class

    internal_field = None
    boundary_field = {}
    in_bf = False
    current_bc = None
    in_bc = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("internalField"):
            rest = stripped.split("internalField", 1)[1].strip().rstrip(";")
            parts = rest.split(None, 1)
            if len(parts) == 1:
                try:
                    internal_field = float(parts[0])
                except ValueError:
                    internal_field = parts[0]
            elif len(parts) == 2:
                val_str = parts[1].strip()
                if val_str.startswith("(") and val_str.endswith(")"):
                    val_str = val_str[1:-1]
                    try:
                        vals = [float(v) for v in val_str.split()]
                        internal_field = np.array(vals, dtype=float)
                    except ValueError:
                        internal_field = val_str
                else:
                    try:
                        internal_field = float(val_str)
                    except ValueError:
                        internal_field = val_str
            continue
        if stripped == "boundaryField":
            in_bf = True
            continue
        if in_bf:
            if stripped == "}":
                break
            if "{" in stripped and not stripped.startswith("type"):
                current_bc = stripped.split("{")[0].strip()
                in_bc = True
                boundary_field[current_bc] = {}
                continue
            if in_bc and "}" in stripped:
                current_bc = None
                in_bc = False
                continue
            if in_bc and ";" in stripped:
                parts = stripped.split(";")[0].strip().split()
                if len(parts) >= 2:
                    boundary_field[current_bc][parts[0]] = parts[1]

    data_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("internalField"):
            for j in range(i + 1, len(lines)):
                if lines[j].strip().startswith("("):
                    data_start = j
                    break
            break

    if data_start is None:
        if internal_field is not None:
            if isinstance(internal_field, np.ndarray):
                return internal_field, is_point_field
            return np.full(1, internal_field, dtype=float), is_point_field
        return np.array([]), is_point_field

    data_lines = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if stripped == ")":
            break
        if stripped and not stripped.startswith("//"):
            data_lines.append(stripped)

    data_str = " ".join(data_lines)
    data_str = data_str.replace("(", "").replace(")", "")
    values = [float(x) for x in data_str.split()]

    if is_vector:
        values = np.array(values, dtype=float).reshape(-1, 3)
    else:
        values = np.array(values, dtype=float)

    return values, is_point_field


def _build_cells_from_faces(
    faces: List[np.ndarray],
    owner: np.ndarray,
    neighbour: Optional[np.ndarray],
    n_cells: int,
) -> Tuple[np.ndarray, np.ndarray]:
    cell_faces = [[] for _ in range(n_cells)]
    for face_idx, own in enumerate(owner):
        cell_faces[own].append(face_idx)
    if neighbour is not None:
        for face_idx, nei in enumerate(neighbour):
            cell_faces[nei].append(face_idx)

    cells = []
    cell_types = []
    for cf in cell_faces:
        cf_sorted = sorted(cf, key=lambda fi: len(faces[fi]), reverse=True)
        n_face_pts = sum(len(faces[fi]) for fi in cf_sorted)
        cell_conn = [n_face_pts]
        for fi in cf_sorted:
            for pt in faces[fi]:
                cell_conn.append(int(pt))
        cells.extend(cell_conn)
        cell_types.append(pv.CellType.POLYGON)

    cells = np.array(cells, dtype=int)
    cell_types = np.array(cell_types, dtype=int)
    return cells, cell_types


def _detect_regions(case_path: Path) -> List[str]:
    regions = []
    seen = set()
    time_dirs = _get_time_dirs(case_path)
    if not time_dirs:
        time_dirs = ["0"]
    for td in time_dirs:
        td_path = case_path / td
        if not td_path.exists() or not td_path.is_dir():
            continue
        for entry in sorted(td_path.iterdir()):
            if entry.is_dir() and (entry / "polyMesh").exists():
                if entry.name not in seen:
                    seen.add(entry.name)
                    regions.append(entry.name)
    constant_path = case_path / "constant"
    if constant_path.exists() and constant_path.is_dir():
        for entry in sorted(constant_path.iterdir()):
            if entry.is_dir() and (entry / "polyMesh").exists():
                if entry.name not in seen:
                    seen.add(entry.name)
                    regions.append(entry.name)
    return regions


def _get_region_mesh_path(case_path: Path, region: str) -> Path:
    mesh_path = case_path / "constant" / region / "polyMesh"
    if mesh_path.exists():
        return mesh_path
    mesh_path = case_path / "constant" / "polyMesh"
    if mesh_path.exists():
        return mesh_path
    raise FileNotFoundError(
        f"No polyMesh directory found for region '{region}' in {case_path}"
    )


def _get_time_dirs(case_path: Path) -> List[str]:
    time_dirs = []
    for entry in sorted(case_path.iterdir()):
        if entry.is_dir() and entry.name.replace(".", "", 1).replace("-", "", 1).isdigit():
            time_dirs.append(entry.name)
    return time_dirs


def _get_latest_time(case_path: Path) -> str:
    time_dirs = _get_time_dirs(case_path)
    if not time_dirs:
        return "0"
    return time_dirs[-1]


class OpenFOAMDirectReader:
    """Direct reader for OpenFOAM cases into PyVista objects.

    Parses OpenFOAM polyMesh and field files directly without
    intermediate conversion (foamToVTK). Supports single-region
    and multi-region (CHT) cases.

    Parameters
    ----------
    case_path : str or Path
        Path to the OpenFOAM case directory.
    region : str, optional
        Region name for multi-region cases. If ``None``, the
        main region (``constant/polyMesh``) is used.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        region: Optional[str] = None,
    ):
        self.case_path = Path(case_path)
        self.region = region
        self._mesh_path: Optional[Path] = None
        self._points: Optional[np.ndarray] = None
        self._faces: Optional[List[np.ndarray]] = None
        self._owner: Optional[np.ndarray] = None
        self._neighbour: Optional[np.ndarray] = None
        self._boundary: Optional[Dict[str, dict]] = None
        self._n_cells: Optional[int] = None
        self._n_points: Optional[int] = None
        self._mesh: Optional[pv.UnstructuredGrid] = None
        self._field_cache: Dict[str, np.ndarray] = {}
        self._field_is_point: Dict[str, bool] = {}

    def _ensure_mesh_loaded(self) -> None:
        if self._mesh is not None:
            return
        if self.region is None:
            self._mesh_path = self.case_path / "constant" / "polyMesh"
        else:
            self._mesh_path = self.case_path / "constant" / self.region / "polyMesh"

        if not self._mesh_path.exists():
            raise FileNotFoundError(
                f"polyMesh directory not found at {self._mesh_path}"
            )

        points_file = self._mesh_path / "points"
        faces_file = self._mesh_path / "faces"
        owner_file = self._mesh_path / "owner"
        neighbour_file = self._mesh_path / "neighbour"
        boundary_file = self._mesh_path / "boundary"

        if not points_file.exists():
            raise FileNotFoundError(f"points file not found: {points_file}")
        if not faces_file.exists():
            raise FileNotFoundError(f"faces file not found: {faces_file}")
        if not owner_file.exists():
            raise FileNotFoundError(f"owner file not found: {owner_file}")

        self._points = _read_points(points_file)
        self._faces = _read_faces(faces_file)
        self._owner = _read_label_list(owner_file)
        self._n_cells = int(self._owner.max()) + 1 if len(self._owner) > 0 else 0

        if neighbour_file.exists():
            self._neighbour = _read_label_list(neighbour_file)
        else:
            self._neighbour = None

        if boundary_file.exists():
            self._boundary = _read_boundary(boundary_file)
        else:
            self._boundary = {}

        self._n_points = len(self._points)

        cells, cell_types = _build_cells_from_faces(
            self._faces, self._owner, self._neighbour, self._n_cells
        )

        self._mesh = pv.UnstructuredGrid(cells, cell_types, self._points)
        logger.info(
            "Loaded mesh: %d points, %d cells, region='%s'",
            self._n_points,
            self._n_cells,
            self.region or "main",
        )

    @property
    def mesh(self) -> pv.UnstructuredGrid:
        """Return the PyVista UnstructuredGrid for the loaded region."""
        self._ensure_mesh_loaded()
        return self._mesh

    @property
    def points(self) -> np.ndarray:
        """Return the mesh points array."""
        self._ensure_mesh_loaded()
        return self._points

    @property
    def boundary_patches(self) -> Dict[str, dict]:
        """Return the boundary patch definitions."""
        self._ensure_mesh_loaded()
        return self._boundary or {}

    @property
    def region_names(self) -> List[str]:
        """Return available region names for multi-region cases."""
        return _detect_regions(self.case_path)

    def get_time_steps(self) -> List[str]:
        """Return sorted list of available time directories."""
        return _get_time_dirs(self.case_path)

    def get_latest_time(self) -> str:
        """Return the latest time directory name."""
        return _get_latest_time(self.case_path)

    def read_field(
        self,
        field_name: str,
        time_step: Optional[str] = None,
        region: Optional[str] = None,
    ) -> np.ndarray:
        """Read a scalar or vector field from a time directory.

        Parameters
        ----------
        field_name : str
            Name of the field (e.g. ``'U'``, ``'p'``, ``'T'``).
        time_step : str, optional
            Time directory name (e.g. ``'0'``, ``'0.1'``).
            Defaults to ``'0'`` or the latest time if the field
            is not found at ``'0'``.
        region : str, optional
            Region subdirectory name. If ``None``, uses the
            reader's ``region`` attribute.

        Returns
        -------
        np.ndarray
            Field values (shape ``(n,)`` for scalars, ``(n, 3)`` for vectors).
        """
        region_dir = region or self.region
        time_step = time_step or "0"

        cache_key = f"{region_dir}:{time_step}:{field_name}"
        if cache_key in self._field_cache:
            return self._field_cache[cache_key]

        search_paths = []
        if region_dir:
            search_paths.append(self.case_path / time_step / region_dir / field_name)
        search_paths.append(self.case_path / time_step / field_name)
        if region_dir is None:
            for reg in self.region_names:
                search_paths.append(self.case_path / time_step / reg / field_name)

        field_path = None
        for p in search_paths:
            if p.exists():
                field_path = p
                break
            gz_path = p.with_suffix(p.suffix + ".gz")
            if gz_path.exists():
                field_path = gz_path
                break

        if field_path is None:
            raise FileNotFoundError(
                f"Field file not found for '{field_name}' at time '{time_step}' "
                f"in region '{region_dir}'. Searched: {search_paths}"
            )

        values, is_point_field = _read_field(field_path)
        self._field_cache[cache_key] = values
        self._field_is_point[cache_key] = is_point_field
        return values

    def attach_field(
        self,
        field_name: str,
        time_step: Optional[str] = None,
        region: Optional[str] = None,
        as_point_data: bool = True,
    ) -> pv.UnstructuredGrid:
        """Read a field and attach it to the mesh as point or cell data.

        Parameters
        ----------
        field_name : str
            Name of the field.
        time_step : str, optional
            Time directory name.
        region : str, optional
            Region subdirectory name.
        as_point_data : bool, optional
            If ``True`` (default), attach as point data. If ``False``,
            attach as cell data.

        Returns
        -------
        pv.UnstructuredGrid
            The mesh with the field attached.
        """
        mesh = self.mesh
        region_dir = region or self.region
        time_step = time_step or "0"
        cache_key = f"{region_dir}:{time_step}:{field_name}"
        values = self.read_field(field_name, time_step=time_step, region=region)
        is_point = self._field_is_point.get(cache_key, as_point_data)

        target = mesh.point_data if is_point else mesh.cell_data
        if values.ndim == 1 and len(values) == (mesh.n_points if is_point else mesh.n_cells):
            target[field_name] = values
        elif values.ndim == 2 and values.shape[0] == (mesh.n_points if is_point else mesh.n_cells):
            target[field_name] = values
        elif is_point and values.ndim == 1 and len(values) == mesh.n_cells:
            logger.warning(
                "Field '%s' has %d values (n_cells=%d), "
                "attaching as cell data instead of point data.",
                field_name, len(values), mesh.n_cells,
            )
            mesh.cell_data[field_name] = values
        elif not is_point and values.ndim == 1 and len(values) == mesh.n_points:
            logger.warning(
                "Field '%s' has %d values (n_points=%d), "
                "attaching as point data instead of cell data.",
                field_name, len(values), mesh.n_points,
            )
            mesh.point_data[field_name] = values
        else:
            target[field_name] = values

        return mesh

    def to_pyvista(
        self,
        fields: Optional[List[str]] = None,
        time_step: Optional[str] = None,
        as_point_data: bool = True,
    ) -> pv.UnstructuredGrid:
        """Build a complete PyVista mesh with attached fields.

        Parameters
        ----------
        fields : list of str, optional
            Field names to load and attach. If ``None``, no fields
            are attached beyond the mesh geometry.
        time_step : str, optional
            Time directory name.
        as_point_data : bool, optional
            Attach fields as point data (``True``) or cell data
            (``False``).

        Returns
        -------
        pv.UnstructuredGrid
            The mesh with attached field data.
        """
        mesh = self.mesh
        if fields:
            for field_name in fields:
                try:
                    mesh = self.attach_field(
                        field_name,
                        time_step=time_step,
                        as_point_data=as_point_data,
                    )
                except FileNotFoundError as e:
                    logger.warning("Skipping field '%s': %s", field_name, e)
        return mesh

    def to_multiblock(
        self,
        fields: Optional[List[str]] = None,
        time_step: Optional[str] = None,
    ) -> pv.MultiBlock:
        """Build a PyVista MultiBlock with all regions.

        For single-region cases, returns a MultiBlock with one block.
        For multi-region CHT cases, each region becomes a separate block.

        Parameters
        ----------
        fields : list of str, optional
            Field names to load and attach.
        time_step : str, optional
            Time directory name.

        Returns
        -------
        pv.MultiBlock
            MultiBlock containing meshes for each region.
        """
        mb = pv.MultiBlock()
        regions = self.region_names if self.region is None else [self.region]
        for reg in regions:
            reader = OpenFOAMDirectReader(
                case_path=self.case_path,
                region=reg if reg != "main" else None,
            )
            mesh = reader.to_pyvista(
                fields=fields,
                time_step=time_step,
            )
            mb.append(mesh, name=reg)
        return mb

    def plot(
        self,
        scalars: Optional[str] = None,
        time_step: Optional[str] = None,
        show_edges: bool = False,
        opacity: float = 1.0,
        cmap: str = "coolwarm",
        off_screen: bool = False,
        screenshot: Optional[str] = None,
        **kwargs,
    ) -> pv.Plotter:
        """Visualize the mesh with optional scalar field.

        Parameters
        ----------
        scalars : str, optional
            Field name to use for coloring.
        time_step : str, optional
            Time directory name.
        show_edges : bool, optional
            Show mesh edges.
        opacity : float, optional
            Mesh opacity.
        cmap : str, optional
            Colormap name.
        off_screen : bool, optional
            Render off-screen (for screenshots).
        screenshot : str, optional
            Path to save screenshot.

        Returns
        -------
        pv.Plotter
            The PyVista plotter object.
        """
        mesh = self.to_pyvista(fields=[scalars] if scalars else None, time_step=time_step)
        pl = pv.Plotter(off_screen=off_screen)
        pl.add_mesh(
            mesh,
            scalars=scalars,
            show_edges=show_edges,
            opacity=opacity,
            cmap=cmap,
            **kwargs,
        )
        if screenshot:
            pl.screenshot(screenshot)
            logger.info("Screenshot saved to %s", screenshot)
        else:
            pl.show()
        return pl


class CHTDirectReader:
    """Direct reader for conjugate heat transfer (CHT) OpenFOAM cases.

    Automatically detects fluid and solid regions and loads them
    into separate PyVista meshes with their respective fields.

    Parameters
    ----------
    case_path : str or Path
        Path to the OpenFOAM case directory.
    """

    def __init__(self, case_path: Union[str, Path]):
        self.case_path = Path(case_path)
        self._direct_readers: Dict[str, OpenFOAMDirectReader] = {}
        self._regions: Dict[str, str] = {}
        self._detect_regions()

    def _detect_regions(self) -> None:
        regions = _detect_regions(self.case_path)
        time_dirs = _get_time_dirs(self.case_path)
        if not time_dirs:
            time_dirs = ["0"]
        latest_time = time_dirs[-1] if time_dirs else "0"

        for reg in regions:
            has_velocity = False
            for td in time_dirs:
                u_file = self.case_path / td / reg / "U"
                if u_file.exists() or (u_file.with_suffix(".gz")).exists():
                    has_velocity = True
                    break
            self._regions[reg] = "fluid" if has_velocity else "solid"

        main_mesh = self.case_path / "constant" / "polyMesh"
        if main_mesh.exists() and not self._regions:
            self._regions["main"] = "fluid"

    def _get_reader(self, region: Optional[str] = None) -> OpenFOAMDirectReader:
        reg_key = region or "main"
        if reg_key not in self._direct_readers:
            reader = OpenFOAMDirectReader(
                case_path=self.case_path,
                region=region,
            )
            self._direct_readers[reg_key] = reader
        return self._direct_readers[reg_key]

    @property
    def region_names(self) -> List[str]:
        """Return available region names."""
        return list(self._regions.keys())

    @property
    def regions(self) -> Dict[str, str]:
        """Return region classification (fluid/solid)."""
        return self._regions

    def get_mesh(
        self,
        region: Optional[str] = None,
        fields: Optional[List[str]] = None,
        time_step: Optional[str] = None,
    ) -> pv.UnstructuredGrid:
        """Get a PyVista mesh for a region with optional fields.

        Parameters
        ----------
        region : str, optional
            Region name. If ``None``, uses the first detected region.
        fields : list of str, optional
            Field names to attach.
        time_step : str, optional
            Time directory name.

        Returns
        -------
        pv.UnstructuredGrid
            The mesh with attached field data.
        """
        reader = self._get_reader(region)
        return reader.to_pyvista(fields=fields, time_step=time_step)

    def get_all_meshes(
        self,
        fields: Optional[List[str]] = None,
        time_step: Optional[str] = None,
    ) -> pv.MultiBlock:
        """Get meshes for all regions as a MultiBlock.

        Parameters
        ----------
        fields : list of str, optional
            Field names to attach to each region mesh.
        time_step : str, optional
            Time directory name.

        Returns
        -------
        pv.MultiBlock
            MultiBlock with one block per region.
        """
        return self._get_reader().to_multiblock(
            fields=fields, time_step=time_step,
        )

    def get_interface_temperatures(
        self,
        interface_name: str,
        time_step: Optional[str] = None,
    ) -> Dict[str, float]:
        """Extract interface temperatures from fluid and solid regions.

        Parameters
        ----------
        interface_name : str
            Name of the interface patch.
        time_step : str, optional
            Time directory name.

        Returns
        -------
        dict
            Dictionary with ``'fluid_T'``, ``'solid_T'``, and
            ``'T_interface'`` (average).
        """
        result = {}
        for reg_name, reg_type in self._regions.items():
            reader = self._get_reader(reg_name if reg_name != "main" else None)
            try:
                T = reader.read_field("T", time_step=time_step)
                result[f"{reg_name}_T"] = float(np.mean(T))
            except (FileNotFoundError, KeyError):
                logger.warning(
                    "Temperature field 'T' not found in region '%s'", reg_name
                )
        if len(result) == 2:
            result["T_interface"] = float(np.mean(list(result.values())))
        return result

    def plot(
        self,
        scalars: Optional[str] = None,
        time_step: Optional[str] = None,
        region: Optional[str] = None,
        show_edges: bool = False,
        opacity: float = 1.0,
        cmap: str = "coolwarm",
        off_screen: bool = False,
        screenshot: Optional[str] = None,
        **kwargs,
    ) -> pv.Plotter:
        """Visualize a region mesh with optional scalar field.

        Parameters
        ----------
        scalars : str, optional
            Field name for coloring.
        time_step : str, optional
            Time directory name.
        region : str, optional
            Region name.
        show_edges : bool, optional
            Show mesh edges.
        opacity : float, optional
            Mesh opacity.
        cmap : str, optional
            Colormap name.
        off_screen : bool, optional
            Render off-screen.
        screenshot : str, optional
            Path to save screenshot.

        Returns
        -------
        pv.Plotter
            The PyVista plotter object.
        """
        return self.get_mesh(
            region=region, fields=[scalars] if scalars else None, time_step=time_step,
        ).plot(
            scalars=scalars,
            show_edges=show_edges,
            opacity=opacity,
            cmap=cmap,
            off_screen=off_screen,
            screenshot=screenshot,
            **kwargs,
        )


def read_openfoam(
    case_path: Union[str, Path],
    region: Optional[str] = None,
    fields: Optional[List[str]] = None,
    time_step: Optional[str] = None,
) -> pv.UnstructuredGrid:
    """Convenience function to read an OpenFOAM case directly into PyVista.

    Parameters
    ----------
    case_path : str or Path
        Path to the OpenFOAM case directory.
    region : str, optional
        Region name for multi-region cases.
    fields : list of str, optional
        Field names to attach to the mesh.
    time_step : str, optional
        Time directory name.

    Returns
    -------
    pv.UnstructuredGrid
        The mesh with attached field data.
    """
    reader = OpenFOAMDirectReader(
        case_path=case_path,
        region=region,
    )
    return reader.to_pyvista(fields=fields, time_step=time_step)


def read_cht_openfoam(
    case_path: Union[str, Path],
    fields: Optional[List[str]] = None,
    time_step: Optional[str] = None,
) -> pv.MultiBlock:
    """Convenience function to read a CHT OpenFOAM case directly into PyVista.

    Parameters
    ----------
    case_path : str or Path
        Path to the OpenFOAM case directory.
    fields : list of str, optional
        Field names to attach to each region mesh.
    time_step : str, optional
        Time directory name.

    Returns
    -------
    pv.MultiBlock
        MultiBlock with one block per region.
    """
    reader = CHTDirectReader(case_path=case_path)
    return reader.get_all_meshes(fields=fields, time_step=time_step)
