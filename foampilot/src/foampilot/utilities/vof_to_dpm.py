"""Conservative VOF-to-DPM fragment extraction utilities.

The module operates on already-read finite-volume arrays.  It deliberately
keeps OpenFOAM I/O separate from the extraction algorithm so the same logic can
be used by case generators, post-processing workflows and unit tests.

A fragment volume is computed as ``sum(alpha_i * V_i)``.  The conversion does
not renormalise alpha above the threshold: the threshold selects cells, while
alpha remains the physical liquid volume fraction.  This distinction is
important for mass conservation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import pi
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class VofFragment:
    """One connected liquid fragment represented by one DPM parcel."""

    cell_indices: tuple[int, ...]
    volume: float
    centroid: tuple[float, float, float]
    velocity: tuple[float, float, float]

    @property
    def equivalent_diameter(self) -> float:
        """Return the diameter of a sphere with the fragment volume."""

        return float((6.0 * self.volume / pi) ** (1.0 / 3.0))


class OpenFoamFormatError(ValueError):
    """Raised when an OpenFOAM ASCII file cannot be interpreted safely."""


class OpenFoamAsciiReader:
    """Read the ASCII finite-volume files needed by VOF-to-DPM extraction.

    The reader supports the standard OpenFOAM `uniform` and `nonuniform List`
    internal-field encodings and the ASCII `C`, `V`, `owner` and `neighbour`
    mesh files. Binary files are rejected explicitly instead of being guessed.
    """

    _comment_re = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)

    @classmethod
    def _tokens(cls, path: str | Path) -> list[str]:
        file_path = Path(path)
        text = file_path.read_text(encoding="utf-8")
        if re.search(r"\bformat\s+binary\s*;", text):
            raise OpenFoamFormatError(f"Binary OpenFOAM file is not supported: {file_path}")
        text = cls._comment_re.sub(" ", text)
        return re.findall(r"\(|\)|\{|\}|;|[^\s(){};]+", text)

    @staticmethod
    def _vector(tokens: list[str], index: int) -> tuple[tuple[float, float, float], int]:
        if tokens[index] != "(":
            raise OpenFoamFormatError("Expected a vector opening parenthesis")
        try:
            vector = tuple(float(tokens[index + offset]) for offset in (1, 2, 3))
        except (IndexError, ValueError) as error:
            raise OpenFoamFormatError("Invalid vector value") from error
        if tokens[index + 4] != ")":
            raise OpenFoamFormatError("Invalid vector closing parenthesis")
        return vector, index + 5

    @classmethod
    def field(cls, path: str | Path) -> np.ndarray:
        """Read a scalar or vector internalField as a NumPy array."""

        tokens = cls._tokens(path)
        try:
            field_index = tokens.index("internalField")
        except ValueError as error:
            raise OpenFoamFormatError(f"No internalField in {path}") from error
        index = field_index + 1
        if tokens[index] == "uniform":
            index += 1
            if tokens[index] == "(":
                value, _ = cls._vector(tokens, index)
                return np.asarray([value], dtype=float)
            try:
                return np.asarray([float(tokens[index])], dtype=float)
            except (IndexError, ValueError) as error:
                raise OpenFoamFormatError(f"Invalid uniform field in {path}") from error
        if tokens[index] != "nonuniform":
            raise OpenFoamFormatError(f"Unsupported internalField encoding in {path}")
        index += 1
        if index < len(tokens) and tokens[index].startswith("List<"):
            index += 1
        try:
            count = int(tokens[index])
        except (IndexError, ValueError) as error:
            raise OpenFoamFormatError(f"Invalid nonuniform count in {path}") from error
        index += 1
        if tokens[index] != "(":
            raise OpenFoamFormatError(f"Missing nonuniform list in {path}")
        index += 1
        values: list[object] = []
        for _ in range(count):
            if index < len(tokens) and tokens[index] == "(":
                value, index = cls._vector(tokens, index)
            else:
                try:
                    value = float(tokens[index])
                except (IndexError, ValueError) as error:
                    raise OpenFoamFormatError(f"Invalid nonuniform value in {path}") from error
                index += 1
            values.append(value)
        if index >= len(tokens) or tokens[index] != ")":
            raise OpenFoamFormatError(f"Unclosed nonuniform list in {path}")
        array = np.asarray(values, dtype=float)
        return array

    @classmethod
    def integer_list(cls, path: str | Path) -> np.ndarray:
        """Read an OpenFOAM ASCII label list such as owner or neighbour."""

        tokens = cls._tokens(path)
        candidates = [
            (index, ")") for index, token in enumerate(tokens) if token == "("
        ] + [
            (index, "}") for index, token in enumerate(tokens) if token == "{"
        ]
        if not candidates:
            raise OpenFoamFormatError(f"No label list in {path}")
        start, closing = max(candidates, key=lambda item: item[0])
        start += 1
        end = start
        while end < len(tokens) and tokens[end] != closing:
            end += 1
        if end == len(tokens):
            raise OpenFoamFormatError(f"Unclosed label list in {path}")
        try:
            return np.asarray([int(token) for token in tokens[start:end]], dtype=int)
        except ValueError as error:
            raise OpenFoamFormatError(f"Invalid label list in {path}") from error


class OpenFoamCaseReader:
    """Load the VOF fields and cell connectivity needed by the converter.

    OpenFOAM 13 writes cell volumes as ``Vc`` in the selected time directory
    when using the standard ``writeCellVolumes`` function object; a precomputed
    ``constant/polyMesh/V`` is also accepted for externally prepared cases.
    """

    def __init__(self, case_directory: str | Path, time_directory: str = "0") -> None:
        self.case_directory = Path(case_directory)
        self.time_directory = self.case_directory / time_directory
        self.mesh_directory = self.case_directory / "constant" / "polyMesh"

    def read(
        self,
        alpha_name: str = "alpha.liquid",
        velocity_name: str | None = "U",
    ) -> dict[str, object]:
        """Read alpha, optional U, C, V and internal-cell connectivity."""

        alpha = OpenFoamAsciiReader.field(self.time_directory / alpha_name)
        centre_path = self.time_directory / "C"
        if not centre_path.exists():
            centre_path = self.mesh_directory / "C"
        centres = OpenFoamAsciiReader.field(centre_path)
        volume_path = self.mesh_directory / "V"
        if not volume_path.exists():
            volume_path = self.time_directory / "Vc"
        volumes = OpenFoamAsciiReader.field(volume_path)
        owner = OpenFoamAsciiReader.integer_list(self.mesh_directory / "owner")
        neighbour = OpenFoamAsciiReader.integer_list(self.mesh_directory / "neighbour")
        # owner contains boundary faces too; neighbour contains internal faces only.
        if owner.size < neighbour.size:
            raise OpenFoamFormatError("owner has fewer faces than neighbour")
        owner = owner[: neighbour.size]
        if alpha.ndim != 1 or centres.ndim != 2 or centres.shape[1] != 3 or volumes.ndim != 1:
            raise OpenFoamFormatError("Unexpected field dimensions in OpenFOAM case")
        if alpha.size != centres.shape[0] or alpha.size != volumes.size:
            raise OpenFoamFormatError("alpha, C and V do not have the same cell count")
        neighbours = [[] for _ in range(alpha.size)]
        for owner_cell, neighbour_cell in zip(owner, neighbour):
            if not 0 <= owner_cell < alpha.size or not 0 <= neighbour_cell < alpha.size:
                raise OpenFoamFormatError("owner/neighbour references an invalid cell")
            neighbours[int(owner_cell)].append(int(neighbour_cell))
            neighbours[int(neighbour_cell)].append(int(owner_cell))
        velocity = None
        if velocity_name is not None:
            velocity = OpenFoamAsciiReader.field(self.time_directory / velocity_name)
            if velocity.shape != centres.shape:
                raise OpenFoamFormatError("U and C do not have the same shape")
        return {
            "alpha": alpha,
            "cell_centres": centres,
            "cell_volumes": volumes,
            "neighbours": neighbours,
            "velocity": velocity,
        }


class VofToDpmConverter:
    """Extract connected VOF fragments while preserving liquid volume.

    Parameters
    ----------
    alpha_threshold:
        Cells with ``alpha >= alpha_threshold`` are eligible for conversion.
        The alpha value itself is retained in the volume integral.
    min_volume:
        Fragments below this physical volume are rejected.  Leave at zero to
        avoid silently discarding liquid.
    min_cells:
        Minimum number of eligible cells in a fragment.
    strict:
        If true, reject invalid alpha, geometry, neighbour or velocity data
        instead of silently repairing it.
    """

    def __init__(
        self,
        alpha_threshold: float = 0.5,
        min_volume: float = 0.0,
        min_cells: int = 1,
        strict: bool = True,
    ) -> None:
        if not 0.0 <= alpha_threshold <= 1.0:
            raise ValueError("alpha_threshold must be between 0 and 1")
        if min_volume < 0.0:
            raise ValueError("min_volume must be non-negative")
        if min_cells < 1:
            raise ValueError("min_cells must be at least one")
        self.alpha_threshold = float(alpha_threshold)
        self.min_volume = float(min_volume)
        self.min_cells = int(min_cells)
        self.strict = bool(strict)

    @staticmethod
    def _array(name: str, value: object, ndim: int) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.ndim != ndim:
            raise ValueError(f"{name} must have {ndim} dimensions")
        return array

    def extract(
        self,
        alpha: Sequence[float],
        cell_centres: Sequence[Sequence[float]],
        cell_volumes: Sequence[float],
        neighbours: Sequence[Iterable[int]],
        velocity: Sequence[Sequence[float]] | None = None,
    ) -> list[VofFragment]:
        """Extract connected fragments from cell-centred VOF data.

        ``neighbours[i]`` contains the cell indices sharing a face with cell
        ``i``.  A fragment's volume and centroid are weighted by ``alpha*V``;
        its velocity is the corresponding liquid-volume-weighted mean of the
        supplied cell velocity.
        """

        alpha_array = self._array("alpha", alpha, 1)
        centres = self._array("cell_centres", cell_centres, 2)
        volumes = self._array("cell_volumes", cell_volumes, 1)
        n_cells = alpha_array.size
        if centres.shape != (n_cells, 3):
            raise ValueError("cell_centres must have shape (nCells, 3)")
        if volumes.size != n_cells:
            raise ValueError("cell_volumes must have one value per cell")
        if len(neighbours) != n_cells:
            raise ValueError("neighbours must have one entry per cell")
        if velocity is None:
            velocity_array = np.zeros((n_cells, 3), dtype=float)
        else:
            velocity_array = self._array("velocity", velocity, 2)
            if velocity_array.shape != (n_cells, 3):
                raise ValueError("velocity must have shape (nCells, 3)")

        if np.any(~np.isfinite(alpha_array)) or np.any(~np.isfinite(volumes)):
            raise ValueError("alpha and cell_volumes must be finite")
        if np.any(volumes <= 0.0):
            raise ValueError("cell_volumes must be strictly positive")
        if np.any((alpha_array < -1e-12) | (alpha_array > 1.0 + 1e-12)):
            raise ValueError("alpha values must lie in [0, 1]")
        if not self.strict:
            alpha_array = np.clip(alpha_array, 0.0, 1.0)

        eligible = alpha_array >= self.alpha_threshold
        visited = np.zeros(n_cells, dtype=bool)
        fragments: list[VofFragment] = []

        for seed in np.flatnonzero(eligible):
            if visited[seed]:
                continue
            stack = [int(seed)]
            visited[seed] = True
            component: list[int] = []
            while stack:
                cell = stack.pop()
                component.append(cell)
                for neighbour in neighbours[cell]:
                    neighbour_index = int(neighbour)
                    if not 0 <= neighbour_index < n_cells:
                        raise ValueError("neighbours contains an invalid cell index")
                    if eligible[neighbour_index] and not visited[neighbour_index]:
                        visited[neighbour_index] = True
                        stack.append(neighbour_index)

            if len(component) < self.min_cells:
                continue
            indices = np.asarray(component, dtype=int)
            weights = alpha_array[indices] * volumes[indices]
            liquid_volume = float(np.sum(weights))
            if liquid_volume < self.min_volume:
                continue
            if liquid_volume <= 0.0:
                continue
            centroid = np.sum(centres[indices] * weights[:, None], axis=0) / liquid_volume
            mean_velocity = (
                np.sum(velocity_array[indices] * weights[:, None], axis=0)
                / liquid_volume
            )
            fragments.append(
                VofFragment(
                    cell_indices=tuple(sorted(component)),
                    volume=liquid_volume,
                    centroid=tuple(float(x) for x in centroid),
                    velocity=tuple(float(x) for x in mean_velocity),
                )
            )

        fragments.sort(key=lambda fragment: fragment.cell_indices[0])
        return fragments

    def extract_case(
        self,
        case_directory: str | Path,
        time_directory: str = "0",
        alpha_name: str = "alpha.liquid",
        velocity_name: str | None = "U",
    ) -> list[VofFragment]:
        """Read an ASCII OpenFOAM case and extract its VOF fragments."""

        fields = OpenFoamCaseReader(case_directory, time_directory).read(
            alpha_name=alpha_name,
            velocity_name=velocity_name,
        )
        return self.extract(
            alpha=fields["alpha"],
            cell_centres=fields["cell_centres"],
            cell_volumes=fields["cell_volumes"],
            neighbours=fields["neighbours"],
            velocity=fields["velocity"],
        )

    @staticmethod
    def total_volume(fragments: Sequence[VofFragment]) -> float:
        """Return the total liquid volume represented by fragments."""

        return float(sum(fragment.volume for fragment in fragments))

    def write_openfoam_outputs(
        self,
        fragments: Sequence[VofFragment],
        output_directory: str | Path,
        cloud_name: str = "vofToDpmCloud",
    ) -> dict[str, Path]:
        """Write positions, fragment properties and a machine-readable report."""

        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        positions_path = directory / f"{cloud_name}Positions"
        properties_path = directory / f"{cloud_name}Fragments"
        report_path = directory / f"{cloud_name}Report.json"

        with positions_path.open("w", encoding="utf-8") as stream:
            stream.write(
                "FoamFile\n{\n"
                "    format ascii;\n"
                "    class vectorField;\n"
                f'    location "constant";\n    object {cloud_name}Positions;\n'
                "}\n\n(\n"
            )
            for fragment in fragments:
                stream.write("(" + " ".join(f"{value:.16g}" for value in fragment.centroid) + ")\n")
            stream.write(")\n")

        with properties_path.open("w", encoding="utf-8") as stream:
            stream.write(
                "FoamFile\n{\n"
                "    format ascii;\n"
                "    class dictionary;\n"
                f'    location "constant";\n    object {cloud_name}Fragments;\n'
                "}\n\nfragments\n(\n"
            )
            for index, fragment in enumerate(fragments):
                stream.write(
                    f"    {{ index {index}; volume {fragment.volume:.16g}; "
                    f"diameter {fragment.equivalent_diameter:.16g}; "
                    f"centroid ({' '.join(f'{x:.16g}' for x in fragment.centroid)}); "
                    f"velocity ({' '.join(f'{x:.16g}' for x in fragment.velocity)}); }}\n"
                )
            stream.write(");\n")

        report = {
            "alphaThreshold": self.alpha_threshold,
            "minVolume": self.min_volume,
            "minCells": self.min_cells,
            "fragmentCount": len(fragments),
            "liquidVolume": self.total_volume(fragments),
            "fragments": [
                {
                    "cellIndices": list(fragment.cell_indices),
                    "volume": fragment.volume,
                    "centroid": list(fragment.centroid),
                    "velocity": list(fragment.velocity),
                    "equivalentDiameter": fragment.equivalent_diameter,
                }
                for fragment in fragments
            ],
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return {
            "positions": positions_path,
            "fragments": properties_path,
            "report": report_path,
        }
