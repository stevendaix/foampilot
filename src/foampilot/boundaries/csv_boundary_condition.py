"""
Time-varying boundary conditions from CSV or pandas DataFrame.

Uses OpenFOAM's built-in ``table`` Function1 with ``format csv`` to read
time-varying data from a CSV file.  This works with any BC type that accepts
a Function1-typed parameter (e.g. ``uniformFixedValue``, ``fixedValue``,
``externalTemperature``, ``externalWallHeatFluxTemperature``, …).

Two access patterns are provided:

* **High-level API** — :func:`set_csv_condition` on the :class:`~foampilot.boundaries.Boundary`
  object.
* **Low-level helpers** — :class:`CsvTimeSeries`, :func:`write_csv_table`,
  :func:`make_uniform_fixed_value_bc`, :func:`make_uniform_fixed_value_vector_bc`.

Supported modes
---------------
* **Scalar field** (temperature, heat-transfer coefficient, …) — uniform value
  that varies with time.
* **Vector field** (velocity, …) — uniform vector that varies with time.
* **Steady-state** — the CSV is still accepted; only the first (or a user-chosen)
  row is used as a constant value.
* **Transient** — OpenFOAM linearly interpolates between the two nearest time
  entries at each time step.

Example
-------
::

    from foampilot import Solver
    from foampilot.boundaries import set_csv_condition

    solver = Solver(case_path)
    solver.transient = True
    solver.energy_activated = True

    # Write a CSV file and attach it to patch "inlet", field "T"
    set_csv_condition(
        solver.boundary,
        patch_name="inlet",
        field="T",
        data="inlet_temperature.csv",
        time_column="time_s",
        value_column="T_inlet_K",
        header_lines=1,
        separator=",",
    )
"""

from __future__ import annotations

import logging
from typing import Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Dict, Any, List

try:
    from scipy.interpolate import griddata
    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CsvTimeSeries
# ---------------------------------------------------------------------------

class CsvTimeSeries:
    """Load and query time-varying data from a CSV file or DataFrame.

    Parameters
    ----------
    data : str, Path or pandas.DataFrame
        Path to a CSV file, or an already-loaded DataFrame.
    time_column : str or int
        Column name or positional index for the independent variable (time).
    value_columns : list of str or int, optional
        Column names or indices for the dependent variable(s).  For a scalar
        field provide one column; for a vector field provide three.
    header_lines : int
        Number of header lines to skip when reading a CSV file.
    separator : str
        Field separator (default ``","``).
    """

    def __init__(
        self,
        data: Union[str, Path, pd.DataFrame],
        time_column: Union[str, int] = 0,
        value_columns: Optional[List[Union[str, int]]] = None,
        header_lines: int = 0,
        separator: str = ",",
    ) -> None:
        if isinstance(data, (str, Path)):
            self.df = pd.read_csv(data, sep=separator, skiprows=header_lines, header=None)
            self._source_path = Path(data)
        else:
            self.df = data.copy()
            self._source_path = None

        self.time_column = time_column
        self.value_columns = value_columns if value_columns is not None else [self.df.columns[1]]

        if isinstance(self.time_column, int):
            self._time_col_name = self.df.columns[self.time_column]
        else:
            self._time_col_name = self.time_column

        self._value_col_names: List[str] = []
        for vc in self.value_columns:
            if isinstance(vc, int):
                self._value_col_names.append(self.df.columns[vc])
            else:
                self._value_col_names.append(vc)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def times(self) -> np.ndarray:
        """Return the time values as a NumPy array."""
        return self.df[self._time_col_name].values

    @property
    def values(self) -> np.ndarray:
        """Return the value(s) as a NumPy array.

        For a scalar series this is a 1-D array; for a vector series it is a
        *(N, 3)* array.
        """
        if len(self._value_col_names) == 1:
            return self.df[self._value_col_names[0]].values
        return self.df[self._value_col_names].values

    def is_vector(self) -> bool:
        """Return *True* if the series represents a vector (3 components)."""
        return len(self._value_col_names) == 3

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_value_at_time(self, time: float, method: str = "linear") -> Any:
        """Interpolate the value at *time*.

        Parameters
        ----------
        time : float
            Time at which to evaluate the series.
        method : str
            Interpolation method (``"linear"`` or ``"spline"``).

        Returns
        -------
        float or list of float
            Interpolated value.
        """
        if self.is_vector():
            return [float(np.interp(time, self.times, self.df[c].values)) for c in self._value_col_names]
        return float(np.interp(time, self.times, self.values))

    def get_initial_value(self) -> Any:
        """Return the value at the first time entry."""
        t0 = float(self.times[0])
        return self.get_value_at_time(t0)

    def get_final_value(self) -> Any:
        """Return the value at the last time entry."""
        t_end = float(self.times[-1])
        return self.get_value_at_time(t_end)


# ---------------------------------------------------------------------------
# CSV file writer
# ---------------------------------------------------------------------------

def write_csv_table(
    case_path: Union[str, Path],
    csv_data: Union[str, Path, pd.DataFrame],
    time_column: Union[str, int] = 0,
    value_columns: Optional[List[Union[str, int]]] = None,
    header_lines: int = 0,
    separator: str = ",",
    filename: Optional[str] = None,
) -> Path:
    """Write a CSV file in OpenFOAM-compatible format inside ``constant/``.

    The CSV header is stripped so that OpenFOAM's ``CsvTableReader`` can
    parse it directly.

    Parameters
    ----------
    case_path : str or Path
        Root directory of the OpenFOAM case.
    csv_data : str, Path or pandas.DataFrame
        Source data.
    time_column : str or int
        Column name or index for the time column.
    value_columns : list of str or int, optional
        Value column(s).  Defaults to all columns except the time column.
    header_lines : int
        Number of header lines to skip in the source CSV.
    separator : str
        Column separator.
    filename : str, optional
        Destination filename inside ``constant/``.

    Returns
    -------
    Path
        Absolute path to the written CSV file.
    """
    case_path = Path(case_path)
    constant_dir = case_path / "constant"
    constant_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(csv_data, (str, Path)):
        src = Path(csv_data)
        # Read without header inference; column names will be 0, 1, 2, ...
        df = pd.read_csv(src, sep=separator, skiprows=header_lines, header=None)
        if filename is None:
            filename = src.name
    else:
        df = csv_data.copy()
        if filename is None:
            filename = "table_data.csv"

    # Select and order columns: time first, then values
    if isinstance(time_column, int):
        tc_name = df.columns[time_column]
    else:
        tc_name = time_column

    if value_columns:
        vc_names = [df.columns[v] if isinstance(v, int) else v for v in value_columns]
    else:
        vc_names = [c for c in df.columns if c != tc_name]

    ordered_cols = [tc_name] + vc_names
    df_out = df[ordered_cols]

    dst = constant_dir / filename
    df_out.to_csv(dst, sep=separator, index=False, header=False)
    logger.debug("Wrote OpenFOAM CSV table to %s", dst)
    return dst


# ---------------------------------------------------------------------------
# BC dict generators
# ---------------------------------------------------------------------------

def make_uniform_fixed_value_bc(
    csv_path: Union[str, Path],
    time_column: Union[str, int] = 0,
    value_column: Union[str, int] = 1,
    header_lines: int = 0,
    separator: str = ",",
    out_of_bounds: str = "clamp",
    interpolation_scheme: str = "linear",
    default_value: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate a ``uniformFixedValue`` BC dict for a **scalar** field.

    The BC reads its value from a CSV table using OpenFOAM's ``table``
    Function1.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file (relative to the case root or absolute).
        Typically ``constant/<filename>`` when written by
        :func:`write_csv_table`.
    time_column : str or int
        Column name or index for time.
    value_column : str or int
        Column name or index for the scalar value.
    header_lines : int
        Number of header lines to skip.
    separator : str
        CSV separator character.
    out_of_bounds : str
        Behaviour outside the table range (``clamp``, ``error``, ``warn``,
        ``zero``, ``repeat``).
    interpolation_scheme : str
        ``linear`` or ``spline``.
    default_value : float, optional
        Initial value for the ``value`` entry.

    Returns
    -------
    dict
        OpenFOAM boundary-condition dictionary.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_absolute() and not str(csv_path).startswith("constant/"):
        csv_path = f"constant/{csv_path}"

    return {
        "type": "uniformFixedValue",
        "uniformValue": {
            "type": "table",
            "format": "csv",
            "nHeaderLine": header_lines,
            "columns": (0, 1),
            "file": f'"{csv_path}"',
            "separator": f'"{separator}"',
            "mergeSeparators": False,
            "interpolationScheme": interpolation_scheme,
        },
        "value": f"uniform {default_value if default_value is not None else 0}",
    }


def make_uniform_fixed_value_vector_bc(
    csv_path: Union[str, Path],
    time_column: Union[str, int] = 0,
    value_columns: Optional[List[Union[str, int]]] = None,
    header_lines: int = 0,
    separator: str = ",",
    out_of_bounds: str = "clamp",
    interpolation_scheme: str = "linear",
    default_value: str = "(0 0 0)",
) -> Dict[str, Any]:
    """Generate a ``uniformFixedValue`` BC dict for a **vector** field.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file (relative to the case root or absolute).
        Typically ``constant/<filename>`` when written by
        :func:`write_csv_table`.
    time_column : str or int
        Column name or index for time.
    value_columns : list of str or int, optional
        Three column names/indices for the *(x, y, z)* components.
    header_lines : int
        Number of header lines to skip.
    separator : str
        CSV separator character.
    out_of_bounds : str
        Behaviour outside the table range.
    interpolation_scheme : str
        ``linear`` or ``spline``.
    default_value : str
        Initial value for the ``value`` entry (e.g. ``"(0 0 0)"``).

    Returns
    -------
    dict
        OpenFOAM boundary-condition dictionary.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_absolute() and not str(csv_path).startswith("constant/"):
        csv_path = f"constant/{csv_path}"

    if value_columns is None:
        value_columns = [1, 2, 3]

    return {
        "type": "uniformFixedValue",
        "uniformValue": {
            "type": "table",
            "format": "csv",
            "nHeaderLine": header_lines,
            "columns": (0, (1, 2, 3)),
            "file": f'"{csv_path}"',
            "separator": f'"{separator}"',
            "mergeSeparators": False,
            "interpolationScheme": interpolation_scheme,
        },
        "value": f"uniform {default_value}",
    }


# ---------------------------------------------------------------------------
# High-level helper
# ---------------------------------------------------------------------------

def set_csv_condition(
    boundary,
    patch_name: str,
    field: str,
    data: Union[str, Path, pd.DataFrame],
    time_column: Union[str, int] = 0,
    value_column: Optional[Union[str, int]] = None,
    value_columns: Optional[List[Union[str, int]]] = None,
    header_lines: int = 0,
    separator: str = ",",
    out_of_bounds: str = "clamp",
    interpolation_scheme: str = "linear",
    default_value: Optional[Union[float, str]] = None,
    csv_filename: Optional[str] = None,
) -> None:
    """Attach a time-varying CSV boundary condition to a patch/field.

    This is the recommended high-level entry point.  It writes the CSV data
    to ``<case>/constant/`` and registers the appropriate BC dictionary on
    the :class:`~foampilot.boundaries.Boundary` object.

    Parameters
    ----------
    boundary : foampilot.boundaries.Boundary
        The boundary manager attached to the solver.
    patch_name : str
        Name of the patch.
    field : str
        Name of the field (e.g. ``"T"``, ``"U"``).
    data : str, Path or pandas.DataFrame
        CSV file path or DataFrame.  The DataFrame must contain at least a
        time column and one (scalar) or three (vector) value columns.
    time_column : str or int
        Column name or index for the time variable.
    value_column : str or int, optional
        Column name or index for a **scalar** value.
    value_columns : list of str or int, optional
        Column names/indices for a **vector** value (three columns).
    header_lines : int
        Number of header lines to skip in the CSV.
    separator : str
        CSV separator.
    out_of_bounds : str
        OpenFOAM ``outOfBounds`` behaviour.
    interpolation_scheme : str
        ``linear`` or ``spline``.
    default_value : float or str, optional
        Initial value written to the ``value`` entry.  If omitted, the first
        value from the data is used.
    csv_filename : str, optional
        Filename inside ``constant/``.  Defaults to the source filename or
        ``"table_data.csv"``.

    Notes
    -----
    * For **scalar** fields pass ``value_column`` (a single column).
    * For **vector** fields pass ``value_columns`` (a list of three columns).
    """
    case_path = boundary.parent.case_path

    # Determine if scalar or vector
    is_vector = value_columns is not None and len(value_columns) == 3

    # Write the CSV table to constant/
    if csv_filename is None:
        if isinstance(data, (str, Path)):
            csv_filename = Path(data).name
        else:
            csv_filename = "table_data.csv"

    csv_dst = write_csv_table(
        case_path=case_path,
        csv_data=data,
        time_column=time_column,
        value_columns=value_columns if is_vector else ([value_column] if value_column is not None else None),
        header_lines=header_lines,
        separator=separator,
        filename=csv_filename,
    )

    # Build the BC dict
    if is_vector:
        bc_dict = make_uniform_fixed_value_vector_bc(
            csv_path=csv_dst.name,
            time_column=0,
            value_columns=list(range(1, 4)),
            header_lines=0,
            separator=separator,
            out_of_bounds=out_of_bounds,
            interpolation_scheme=interpolation_scheme,
            default_value=str(default_value) if default_value is not None else "(0 0 0)",
        )
    else:
        # Infer value column if not provided
        if value_column is None:
            series = CsvTimeSeries(data, time_column=time_column, header_lines=header_lines, separator=separator)
            if default_value is None:
                default_value = series.get_initial_value()

        bc_dict = make_uniform_fixed_value_bc(
            csv_path=csv_dst.name,
            time_column=0,
            value_column=1,
            header_lines=0,
            separator=separator,
            out_of_bounds=out_of_bounds,
            interpolation_scheme=interpolation_scheme,
            default_value=default_value if default_value is not None else 0,
        )

    # Register on the boundary manager
    boundary.set_raw_condition(patch_name, field, bc_dict)
    logger.info(
        "CSV BC registered: patch=%s, field=%s, csv=%s",
        patch_name, field, csv_dst.name,
    )


# ---------------------------------------------------------------------------
# Spatial CSV helpers
# ---------------------------------------------------------------------------

_FIELD_DEFAULTS = {
    "T": ("volScalarField", "[0 0 0 0 0 0 0]"),
    "U": ("volVectorField", "[0 1 -1 0 0 0 0]"),
    "p": ("volScalarField", "[0 2 -2 0 0 0 0]"),
    "k": ("volScalarField", "[0 2 -2 0 0 0 0]"),
    "epsilon": ("volScalarField", "[0 2 -3 0 0 0 0]"),
    "omega": ("volScalarField", "[0 0 -1 0 0 0 0]"),
    "nut": ("volScalarField", "[0 2 -1 0 0 0 0]"),
    "nuTilda": ("volScalarField", "[0 2 -1 0 0 0 0]"),
}


def _pivot_long_to_wide(
    df: pd.DataFrame,
    time_column: Union[str, int],
    face_id_column: Union[str, int],
    value_column: Union[str, int],
) -> pd.DataFrame:
    """Pivot a long-format CSV into wide format.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    time_column : str or int
        Column name or index for time.
    face_id_column : str or int
        Column name or index for face ID.
    value_column : str or int
        Column name or index for value.

    Returns
    -------
    pandas.DataFrame
        Wide-format DataFrame with time as first column and one column per face.
    """
    tc = df.columns[time_column] if isinstance(time_column, int) else time_column
    fc = df.columns[face_id_column] if isinstance(face_id_column, int) else face_id_column
    vc = df.columns[value_column] if isinstance(value_column, int) else value_column

    pivot = df.pivot(index=tc, columns=fc, values=vc)
    pivot = pivot.sort_index()
    pivot.columns.name = None
    pivot.index.name = None
    pivot = pivot.reset_index(drop=False)
    return pivot


def _format_nonuniform_scalar(values: np.ndarray) -> str:
    """Format a 1-D array as an OpenFOAM nonuniform scalar list."""
    vals = " ".join(f"{v:.15g}" for v in values)
    return f"nonuniform List<scalar> {len(values)}({vals})"


def _format_nonuniform_vector(values: List[Tuple[float, float, float]]) -> str:
    """Format a list of 3-D tuples as an OpenFOAM nonuniform vector list."""
    vals = " ".join(f"({v[0]:.15g} {v[1]:.15g} {v[2]:.15g})" for v in values)
    return f"nonuniform List<vector> {len(values)}({vals})"


def _write_spatial_field(
    file_path: Path,
    field_name: str,
    dimensions: str,
    values: Union[np.ndarray, List],
    is_vector: bool = False,
) -> None:
    """Write an OpenFOAM field file with nonuniform boundary values.

    Parameters
    ----------
    file_path : Path
        Destination path for the field file.
    field_name : str
        Name of the field (e.g. ``"T"``).
    dimensions : str
        OpenFOAM dimensions string.
    values : array-like
        1-D array for scalar, or list of 3-tuples for vector.
    is_vector : bool
        ``True`` if the field is a vector field.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    field_class = "volVectorField" if is_vector else "volScalarField"

    with open(file_path, "w") as f:
        f.write("FoamFile\n{\n")
        f.write("    version     2.0;\n")
        f.write("    format      ascii;\n")
        f.write(f"    class       {field_class};\n")
        f.write(f"    object      {field_name};\n")
        f.write("}\n\n")
        f.write(f"dimensions      {dimensions};\n")
        f.write("internalField   uniform 0;\n\n")
        f.write("boundaryField\n{\n")
        f.write("    __PATCH__\n    {\n")
        f.write("        type fixedValue;\n")
        if is_vector:
            f.write(f"        value {_format_nonuniform_vector(values)};\n")
        else:
            f.write(f"        value {_format_nonuniform_scalar(values)};\n")
        f.write("    }\n")
        f.write("}\n")
        f.write("// ************************************************************************* //\n")


def _read_openfoam_mesh(case_path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Read points and patch faces from an OpenFOAM polyMesh.

    Parameters
    ----------
    case_path : Path
        Root directory of the OpenFOAM case.

    Returns
    -------
    points : numpy.ndarray
        Array of shape ``(nPoints, 3)``.
    patch_faces : dict
        Mapping from patch name to array of face indices.
    """
    points_file = case_path / "constant" / "polyMesh" / "points"
    faces_file = case_path / "constant" / "polyMesh" / "faces"
    boundary_file = case_path / "constant" / "polyMesh" / "boundary"

    if not points_file.exists() or not faces_file.exists():
        raise FileNotFoundError(
            f"OpenFOAM mesh not found in {case_path / 'constant' / 'polyMesh'}"
        )

    points = _read_of_points(points_file)
    faces = _read_of_faces(faces_file)

    patch_faces: Dict[str, np.ndarray] = {}
    if boundary_file.exists():
        patch_faces = _read_of_boundary(boundary_file, len(faces))

    return points, patch_faces


def _read_of_points(file_path: Path) -> np.ndarray:
    """Parse an OpenFOAM ``points`` file."""
    lines = file_path.read_text().splitlines()
    pts = []
    in_points = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("FoamFile"):
            continue
        if line == "{":
            continue
        if line == "(":
            in_points = True
            continue
        if line == ")":
            break
        if in_points and line.startswith("("):
            vals = line.strip("()").split()
            if len(vals) == 3:
                pts.append([float(v) for v in vals])
    return np.array(pts)


def _read_of_faces(file_path: Path) -> List[np.ndarray]:
    """Parse an OpenFOAM ``faces`` file."""
    lines = file_path.read_text().splitlines()
    faces = []
    in_faces = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("FoamFile") or line.startswith("*"):
            continue
        if line == "{":
            continue
        if line == "(":
            in_faces = True
            continue
        if line == ")":
            break
        if in_faces:
            if line.startswith("("):
                vals = line.strip("()").split()
            else:
                parts = line.split("(")
                if len(parts) == 2:
                    vals = parts[1].strip(")").split()
                else:
                    continue
            if vals:
                faces.append(np.array([int(v) for v in vals]))
    return faces


def _compute_face_centres(
    points: np.ndarray,
    faces: List[np.ndarray],
    face_indices: np.ndarray,
) -> np.ndarray:
    """Compute face centres for given face indices."""
    centres = []
    for idx in face_indices:
        face = faces[idx]
        pts = points[face]
        centres.append(pts.mean(axis=0))
    return np.array(centres)


def _read_of_boundary(file_path: Path, n_faces: int) -> Dict[str, np.ndarray]:
    """Parse an OpenFOAM ``boundary`` file and return patch face ranges."""
    text = file_path.read_text()
    patches = {}
    import re
    patch_pattern = re.compile(
        r'(\w+)\s*\{\s*type\s+\w+.*?nFaces\s+(\d+).*?startFace\s+(\d+)',
        re.DOTALL | re.IGNORECASE
    )
    for match in patch_pattern.finditer(text):
        patch_name = match.group(1)
        n_faces_patch = int(match.group(2))
        start_face = int(match.group(3))
        patches[patch_name] = np.arange(start_face, start_face + n_faces_patch)
    return patches


def _read_of_boundary_types(file_path: Path) -> Dict[str, str]:
    """Parse an OpenFOAM ``boundary`` file and return patch types."""
    text = file_path.read_text()
    patches = {}
    import re
    patch_pattern = re.compile(
        r'(\w+)\s*\{\s*type\s+(\w+)',
        re.DOTALL | re.IGNORECASE
    )
    for match in patch_pattern.finditer(text):
        patch_name = match.group(1)
        patch_type = match.group(2)
        patches[patch_name] = patch_type
    return patches


_DEFAULT_BC_FOR_TYPE = {
    "patch": "fixedValue",
    "wall": "zeroGradient",
    "empty": "empty",
    "cyclic": "cyclic",
    "symmetryPlane": "symmetry",
    "wedge": "wedge",
    "processorCyclic": "processorCyclic",
    "nonConformalCyclic": "nonConformalCyclic",
}


def _get_default_bc_for_patch_type(patch_type: str) -> str:
    return _DEFAULT_BC_FOR_TYPE.get(patch_type, "zeroGradient")


def set_spatial_csv_condition(
    boundary,
    patch_name: str,
    field: str,
    data: Union[str, Path, pd.DataFrame],
    time_column: Union[str, int] = 0,
    spatial_columns: Optional[List[Union[str, int]]] = None,
    face_id_column: Optional[Union[str, int]] = None,
    value_column: Optional[Union[str, int]] = None,
    header_lines: int = 0,
    separator: str = ",",
    default_value: Optional[Union[float, str]] = None,
    interpolation_method: str = "linear",
) -> None:
    """Attach a time-varying **spatial** CSV boundary condition to a patch.

    Reads a CSV file or DataFrame containing either:

    - **Wide format**: one row per time, one column per spatial point.
      Provide ``spatial_columns``.
    - **Long format**: one row per face per time, with explicit face ID.
      Provide ``face_id_column`` and ``value_column``.
    - **Point cloud format**: columns ``time, x, y, z, value`` (or without
      time for steady-state). The source points are interpolated onto the
      patch face centres of the OpenFOAM mesh using ``scipy.interpolate.griddata``.

    Parameters
    ----------
    boundary : foampilot.boundaries.Boundary
        The boundary manager attached to the solver.
    patch_name : str
        Name of the patch.
    field : str
        Name of the field (e.g. ``"T"``, ``"U"``).
    data : str, Path or pandas.DataFrame
        CSV file path or DataFrame.
    time_column : str or int
        Column name or index for time.
    spatial_columns : list of str or int, optional
        Column names/indices for spatial values (wide format).
    face_id_column : str or int, optional
        Column name/index for face ID (long format).
    value_column : str or int, optional
        Column name/index for value (long format).
    header_lines : int
        Number of header lines to skip in the CSV.
    separator : str
        CSV separator character.
    default_value : float or str, optional
        Default value used when a time snapshot is missing.
    interpolation_method : str
        Interpolation method for spatial interpolation: ``"linear"``,
        ``"nearest"``, or ``"cubic"`` (requires SciPy).

    Examples
    --------
    Wide format (point cloud with x,y,z)::

        set_spatial_csv_condition(
            boundary=solver.boundary,
            patch_name="inlet",
            field="T",
            data="inlet_temperature_spatial.csv",
            time_column=0,
            spatial_columns=[1, 2, 3],
        )

    Long format with face IDs::

        set_spatial_csv_condition(
            boundary=solver.boundary,
            patch_name="inlet",
            field="T",
            data="inlet_temperature_long.csv",
            time_column="time_s",
            face_id_column="face_id",
            value_column="T_K",
        )
    """
    if not HAS_SCIPY:
        raise ImportError(
            "set_spatial_csv_condition requires scipy. "
            "Install it with: pip install scipy"
        )

    case_path = boundary.parent.case_path

    # Read the CSV
    if isinstance(data, (str, Path)):
        df = pd.read_csv(
            Path(data),
            sep=separator,
            skiprows=header_lines,
            header=None,
        )
    else:
        df = data.copy()

    # Read OpenFOAM mesh
    points, patch_faces = _read_openfoam_mesh(case_path)

    if patch_name not in patch_faces:
        raise ValueError(
            f"Patch '{patch_name}' not found in mesh. "
            f"Available patches: {list(patch_faces.keys())}"
        )

    face_indices = patch_faces[patch_name]
    face_centres = _compute_face_centres(points, _read_of_faces(case_path / "constant" / "polyMesh" / "faces"), face_indices)

    # Determine format and interpolate
    if spatial_columns is not None:
        # Point cloud format: time, x, y, z, value(s)
        tc = df.columns[time_column] if isinstance(time_column, int) else time_column
        sc = [df.columns[c] if isinstance(c, int) else c for c in spatial_columns]

        times = sorted(df[tc].unique())
        field_files = {}

        for t in times:
            subset = df[df[tc] == t]
            src_points = subset[sc[:3]].values if len(sc) >= 3 else subset[sc[:2]].values
            src_values = subset[sc[-1]].values

            if src_points.shape[1] == 2:
                src_points = np.hstack([src_points, np.zeros((len(src_points), 1))])

            if len(src_points) > 0:
                interp_values = griddata(
                    src_points, src_values, face_centres, method=interpolation_method
                )
                if np.any(np.isnan(interp_values)):
                    if default_value is not None:
                        interp_values = np.where(np.isnan(interp_values), float(default_value), interp_values)
                    else:
                        raise ValueError(
                            f"Interpolation produced NaN values at t={t}. "
                            "Consider using a larger domain or different interpolation method."
                        )
            else:
                interp_values = np.full(len(face_centres), float(default_value) if default_value is not None else 0.0)

            field_files[t] = interp_values

    elif face_id_column is not None and value_column is not None:
        # Long format with face IDs
        tc = df.columns[time_column] if isinstance(time_column, int) else time_column
        fc = df.columns[face_id_column] if isinstance(face_id_column, int) else face_id_column
        vc = df.columns[value_column] if isinstance(value_column, int) else value_column

        times = sorted(df[tc].unique())
        field_files = {}

        for t in times:
            subset = df[df[tc] == t]
            face_ids = subset[fc].values
            values = subset[vc].values

            interp_values = np.full(len(face_centres), np.nan)
            valid_mask = (face_ids >= 0) & (face_ids < len(interp_values))
            interp_values[face_ids[valid_mask]] = values[valid_mask]

            if np.any(np.isnan(interp_values)):
                if default_value is not None:
                    interp_values = np.where(np.isnan(interp_values), float(default_value), interp_values)
                else:
                    raise ValueError(
                        f"Missing face values at t={t} for {np.sum(np.isnan(interp_values))} faces. "
                        "Consider providing a default_value."
                    )

            field_files[t] = interp_values
    else:
        raise ValueError(
            "Either spatial_columns or face_id_column+value_column must be provided"
        )

    # Determine field metadata
    field_name_upper = field.upper()
    is_vector = field_name_upper in ("U",)
    field_class, dimensions = _FIELD_DEFAULTS.get(
        field_name_upper,
        ("volScalarField", "[0 0 0 0 0 0 0]"),
    )
    if is_vector:
        field_class = "volVectorField"
        dimensions = "[0 1 -1 0 0 0 0]"

    # Write the initial spatial BC to 0/<field> as well, so that OpenFOAM
    # reads the nonuniform values from the initial time directory.
    if field_files:
        initial_time = sorted(field_files.keys())[0]
        initial_values = field_files[initial_time]
        _write_spatial_field_from_template(
            file_path=case_path / "0" / field,
            patch_name=patch_name,
            field_name=field,
            case_path=case_path,
            values=initial_values,
            is_vector=is_vector,
        )

    # Write field files for each time
    for t, values in field_files.items():
        time_dir = case_path / str(t)
        time_dir.mkdir(parents=True, exist_ok=True)
        field_file = time_dir / field

        _write_spatial_field_from_template(
            file_path=field_file,
            patch_name=patch_name,
            field_name=field,
            case_path=case_path,
            values=values,
            is_vector=is_vector,
        )

    # Register a fixedValue BC on the patch.
    bc_dict = {
        "type": "fixedValue",
        "value": "uniform 0",
    }

    boundary.set_raw_condition(patch_name, field, bc_dict)
    logger.info(
        "Spatial CSV BC registered: patch=%s, field=%s, times=%s",
        patch_name, field, list(field_files.keys()),
    )


def _write_spatial_field_from_template(
    file_path: Path,
    patch_name: str,
    field_name: str,
    case_path: Path,
    values: Union[np.ndarray, List],
    is_vector: bool = False,
) -> None:
    """Write an OpenFOAM field file by copying the 0/<field> template and
    replacing the target patch's boundary value with nonuniform data.

    This preserves all other patches' boundary conditions from the template.
    Missing patches are added with default BCs based on their mesh type.

    Parameters
    ----------
    file_path : Path
        Destination path for the field file.
    patch_name : str
        Name of the boundary patch to update.
    field_name : str
        Name of the field (e.g. ``"T"``).
    case_path : Path
        Root directory of the OpenFOAM case.
    values : array-like
        1-D array for scalar, or list of 3-tuples for vector.
    is_vector : bool
        ``True`` if the field is a vector field.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    template_path = case_path / "0" / field_name
    boundary_file = case_path / "constant" / "polyMesh" / "boundary"
    mesh_patches = {}
    if boundary_file.exists():
        mesh_patches = _read_of_boundary_types(boundary_file)

    if is_vector:
        value_str = f"value {_format_nonuniform_vector(values)};"
    else:
        value_str = f"value {_format_nonuniform_scalar(values)};"

    if template_path.exists():
        content = template_path.read_text()
        lines = content.split("\n")

        new_lines = []
        i = 0
        in_boundary_field = False
        in_target_patch = False
        target_patch_indent = 0
        found_target = False
        found_value = False

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if stripped == "boundaryField":
                in_boundary_field = True
                new_lines.append(line)
                i += 1
                continue

            if in_boundary_field and not found_target:
                # Match both quoted ("inlet") and unquoted (inlet) patch names
                check_stripped = stripped.strip('"\'')
                if check_stripped == patch_name:
                    if stripped.endswith("{"):
                        in_target_patch = True
                        found_target = True
                        target_patch_indent = len(line) - len(line.lstrip())
                        new_lines.append(line)
                        i += 1
                        continue
                    else:
                        in_target_patch = True
                        found_target = True
                        target_patch_indent = len(line) - len(line.lstrip())
                        new_lines.append(line)
                        i += 1
                        continue

            if in_target_patch:
                if stripped.startswith("value") and not found_value:
                    new_lines.append(" " * (target_patch_indent + 4) + value_str)
                    found_value = True
                    i += 1
                    continue
                if stripped == "}" or (stripped.startswith("}") and len(stripped) == 1):
                    new_lines.append(line)
                    in_target_patch = False
                    i += 1
                    continue
                else:
                    new_lines.append(line)
                    i += 1
                    continue

            new_lines.append(line)
            i += 1

        content = "\n".join(new_lines)

        if mesh_patches:
            for pname, ptype in mesh_patches.items():
                if pname == patch_name:
                    continue
                if f"    {pname}\n" in content or f"\n{pname}\n" in content:
                    continue
                default_bc = _get_default_bc_for_patch_type(ptype)
                if default_bc == "fixedValue":
                    patch_block = f"\n    {pname}\n    {{\n        type {default_bc};\n        value uniform 0;\n    }}"
                else:
                    patch_block = f"\n    {pname}\n    {{\n        type {default_bc};\n    }}"
                content = content.replace(
                    "boundaryField\n{",
                    "boundaryField\n{" + patch_block,
                    1,
                )

        file_path.write_text(content)
    else:
        field_class = "volVectorField" if is_vector else "volScalarField"
        dimensions = "[0 1 -1 0 0 0 0]" if is_vector else "[0 0 0 0 0 0 0]"
        with open(file_path, "w") as f:
            f.write("FoamFile\n{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write(f"    class       {field_class};\n")
            f.write(f"    object      {field_name};\n")
            f.write("}\n\n")
            f.write(f"dimensions      {dimensions};\n")
            f.write("internalField   uniform 0;\n\n")
            f.write("boundaryField\n{\n")
            for pname, ptype in mesh_patches.items():
                if pname == patch_name:
                    f.write(f"    {pname}\n    {{\n")
                    f.write("        type fixedValue;\n")
                    if is_vector:
                        f.write(f"        value {_format_nonuniform_vector(values)};\n")
                    else:
                        f.write(f"        value {_format_nonuniform_scalar(values)};\n")
                    f.write("    }\n")
                else:
                    default_bc = _get_default_bc_for_patch_type(ptype)
                    f.write(f"    {pname}\n    {{\n")
                    f.write(f"        type {default_bc};\n")
                    f.write("    }\n")
            f.write("}\n")
            f.write("// ************************************************************************* //\n")
