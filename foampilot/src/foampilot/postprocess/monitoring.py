"""Engineering-oriented monitors for OpenFOAM post-processing outputs.

The monitor deliberately consumes the existing :class:`FoamPostProcessing`
interface. It does not assume a particular solver and works with scalar and
vector point/cell fields loaded from VTK files.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonitorPoint:
    """A named spatial probe expressed in mesh coordinates."""

    name: str
    coordinates: tuple[float, float, float]


class CFDMonitor:
    """Extract repeatable engineering monitors from a VTK-backed case.

    Parameters
    ----------
    postprocessor:
        An existing ``FoamPostProcessing`` instance.
    """

    def __init__(self, postprocessor: Any):
        self.postprocessor = postprocessor

    @staticmethod
    def _values(mesh: Any, field: str, association: str = "auto", magnitude: bool = False) -> np.ndarray:
        if association not in {"auto", "point", "cell"}:
            raise ValueError("association must be 'auto', 'point' or 'cell'")
        point = field in mesh.point_data
        cell = field in mesh.cell_data
        if association == "point" and not point:
            raise KeyError(f"Point field {field!r} is not available")
        if association == "cell" and not cell:
            raise KeyError(f"Cell field {field!r} is not available")
        if association == "auto":
            if point:
                association = "point"
            elif cell:
                association = "cell"
            else:
                raise KeyError(f"Field {field!r} is not available as point or cell data")
        values = np.asarray(mesh.point_data[field] if association == "point" else mesh.cell_data[field])
        if values.ndim > 1 and magnitude:
            values = np.linalg.norm(values, axis=1)
        return values

    def _mesh(self, time_step: Any, region: str = "cell") -> Any:
        structure = self.postprocessor.load_time_step(time_step)
        if region == "cell":
            return structure["cell"]
        try:
            return structure["boundaries"][region]
        except KeyError as exc:
            raise KeyError(f"Region {region!r} is not available at time {time_step}") from exc

    @staticmethod
    def statistics(values: np.ndarray) -> dict[str, float]:
        """Return robust engineering statistics for scalar or vector values."""
        arr = np.asarray(values, dtype=float)
        if arr.ndim > 1:
            arr = np.linalg.norm(arr, axis=1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"n_samples": 0, **{key: float("nan") for key in ("mean", "std", "min", "max", "p05", "p50", "p95", "rms")}}
        return {
            "n_samples": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p05": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "rms": float(np.sqrt(np.mean(arr**2))),
        }

    def track_region(
        self,
        field: str,
        region: str = "cell",
        time_steps: Optional[Iterable[Any]] = None,
        association: str = "auto",
        magnitude: bool = False,
    ) -> pd.DataFrame:
        """Build a time series of statistics for a region and field."""
        steps = list(self.postprocessor.get_all_time_steps() if time_steps is None else time_steps)
        records = []
        for step in steps:
            values = self._values(self._mesh(step, region), field, association, magnitude)
            records.append({"time": step, "region": region, "field": field, **self.statistics(values)})
        return pd.DataFrame.from_records(records).set_index("time") if records else pd.DataFrame()

    def track_point(
        self,
        point: Sequence[float],
        field: str,
        time_steps: Optional[Iterable[Any]] = None,
        association: str = "point",
        magnitude: bool = False,
    ) -> pd.DataFrame:
        """Track the nearest point value over time."""
        if association != "point":
            raise ValueError("Point probes currently require point association")
        steps = list(self.postprocessor.get_all_time_steps() if time_steps is None else time_steps)
        records = []
        target = np.asarray(point, dtype=float)
        for step in steps:
            mesh = self._mesh(step)
            values = self._values(mesh, field, association, magnitude)
            index = mesh.find_closest_point(target)
            value = values[index]
            if np.ndim(value) > 0:
                value = float(np.linalg.norm(value)) if magnitude else value.tolist()
            records.append({"time": step, "field": field, "x": target[0], "y": target[1], "z": target[2], "value": value})
        return pd.DataFrame.from_records(records).set_index("time") if records else pd.DataFrame()

    def summary(
        self,
        fields: Sequence[str],
        time_step: Any = None,
        region: str = "cell",
        association: str = "auto",
        magnitudes: Optional[Sequence[str]] = None,
    ) -> dict[str, Any]:
        """Return a JSON-ready latest-time summary for selected fields."""
        if time_step is None:
            steps = list(self.postprocessor.get_all_time_steps())
            if not steps:
                raise FileNotFoundError("No time steps are available")
            time_step = steps[-1]
        magnitudes = set(magnitudes or ())
        mesh = self._mesh(time_step, region)
        return {
            "time": time_step,
            "region": region,
            "fields": {
                field: self.statistics(self._values(mesh, field, association, field in magnitudes))
                for field in fields
            },
        }

    @staticmethod
    def export_csv(frame: pd.DataFrame, filename: str | Path) -> Path:
        """Write a monitor table and return its resolved path."""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path)
        return path

    @staticmethod
    def export_json(data: dict[str, Any], filename: str | Path) -> Path:
        """Write a JSON summary with NumPy/Pandas scalar compatibility."""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)))
        return path


__all__ = ["CFDMonitor", "MonitorPoint"]
