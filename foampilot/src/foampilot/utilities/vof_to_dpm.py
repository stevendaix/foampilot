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
