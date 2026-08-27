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
        """Track a spatially stable point- or cell-associated value over time."""
        if association not in {"point", "cell"}:
            raise ValueError("Probe association must be 'point' or 'cell'")
        steps = list(self.postprocessor.get_all_time_steps() if time_steps is None else time_steps)
        records = []
        target = np.asarray(point, dtype=float)
        for step in steps:
            mesh = self._mesh(step)
            values = self._values(mesh, field, association, magnitude)
            if association == "point":
                index = mesh.find_closest_point(target)
            else:
                index = mesh.find_containing_cell(target)
                if index < 0:
                    centers = mesh.cell_centers().points
                    index = int(np.argmin(np.sum((centers - target) ** 2, axis=1)))
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


def integrate_surface_forces(
    normals: np.ndarray,
    areas: np.ndarray,
    pressure: np.ndarray,
    wall_shear: Optional[np.ndarray] = None,
    *,
    rho: float,
    reference_velocity: float,
    reference_area: float,
    drag_direction: Sequence[float] = (1.0, 0.0, 0.0),
    lift_direction: Sequence[float] = (0.0, 1.0, 0.0),
    pressure_reference: float = 0.0,
) -> dict[str, float]:
    """Integrate pressure and viscous forces over a surface.

    ``normals`` are outward unit normals and ``wall_shear`` is the viscous
    traction vector in Pa. The pressure traction convention is ``-p*n``.
    ``lift_direction`` and ``drag_direction`` are normalized internally and
    must be orthogonal for unambiguous coefficients.
    """
    n = np.asarray(normals, dtype=float)
    a = np.asarray(areas, dtype=float).reshape(-1)
    p = np.asarray(pressure, dtype=float).reshape(-1) - pressure_reference
    if n.ndim != 2 or n.shape[1] != 3 or len(n) != len(a) or len(p) != len(a):
        raise ValueError("normals, areas and pressure must describe the same surface cells")
    if np.any(a < 0) or rho <= 0 or reference_velocity <= 0 or reference_area <= 0:
        raise ValueError("areas must be non-negative and reference quantities positive")
    shear = np.zeros_like(n) if wall_shear is None else np.asarray(wall_shear, dtype=float)
    if shear.shape != n.shape:
        raise ValueError("wall_shear must have shape (n_cells, 3)")
    drag = np.asarray(drag_direction, dtype=float)
    lift = np.asarray(lift_direction, dtype=float)
    drag_norm = np.linalg.norm(drag)
    lift_norm = np.linalg.norm(lift)
    if drag_norm == 0 or lift_norm == 0:
        raise ValueError("force directions must be non-zero")
    drag = drag / drag_norm
    lift = lift / lift_norm
    if abs(float(np.dot(drag, lift))) > 1e-8:
        raise ValueError("drag_direction and lift_direction must be orthogonal")
    pressure_force = np.sum((-p[:, None] * n) * a[:, None], axis=0)
    viscous_force = np.sum(shear * a[:, None], axis=0)
    total_force = pressure_force + viscous_force
    dynamic_pressure = 0.5 * rho * reference_velocity**2
    return {
        "pressure_force_x": float(pressure_force[0]),
        "pressure_force_y": float(pressure_force[1]),
        "pressure_force_z": float(pressure_force[2]),
        "viscous_force_x": float(viscous_force[0]),
        "viscous_force_y": float(viscous_force[1]),
        "viscous_force_z": float(viscous_force[2]),
        "force_x": float(total_force[0]),
        "force_y": float(total_force[1]),
        "force_z": float(total_force[2]),
        "drag": float(np.dot(total_force, drag)),
        "lift": float(np.dot(total_force, lift)),
        "Cd": float(np.dot(total_force, drag) / (dynamic_pressure * reference_area)),
        "Cl": float(np.dot(total_force, lift) / (dynamic_pressure * reference_area)),
    }


def compute_y_plus(
    wall_distance: np.ndarray,
    wall_shear_magnitude: np.ndarray,
    *,
    rho: float,
    kinematic_viscosity: float,
) -> np.ndarray:
    """Compute ``y+ = y*u_tau/nu`` from wall distance and shear stress."""
    y = np.asarray(wall_distance, dtype=float)
    tau = np.asarray(wall_shear_magnitude, dtype=float)
    if y.shape != tau.shape:
        raise ValueError("wall_distance and wall_shear_magnitude must have the same shape")
    if rho <= 0 or kinematic_viscosity <= 0 or np.any(y < 0) or np.any(tau < 0):
        raise ValueError("rho, viscosity, distance and shear stress must be non-negative/positive")
    friction_velocity = np.sqrt(tau / rho)
    return y * friction_velocity / kinematic_viscosity


__all__.extend(["integrate_surface_forces", "compute_y_plus"])



def integrate_mass_flux(
    normals: np.ndarray,
    areas: np.ndarray,
    velocity: np.ndarray,
    *,
    density: float | np.ndarray = 1.0,
) -> dict[str, float]:
    """Integrate outward mass and volumetric flux over a patch.

    Normals must be outward unit normals and ``velocity`` is in m/s. A scalar
    density or one density value per face may be supplied. The sign follows
    ``rho * U dot n * dA``: positive is outward, negative is inward.
    """
    n = np.asarray(normals, dtype=float)
    a = np.asarray(areas, dtype=float).reshape(-1)
    u = np.asarray(velocity, dtype=float)
    if n.ndim != 2 or n.shape[1] != 3 or u.shape != n.shape or len(a) != len(n):
        raise ValueError("normals, areas and velocity must describe the same faces")
    if np.any(a < 0):
        raise ValueError("areas must be non-negative")
    rho = np.asarray(density, dtype=float)
    if rho.ndim == 0:
        if float(rho) <= 0:
            raise ValueError("density must be positive")
    elif rho.shape != (len(a),) or np.any(rho <= 0):
        raise ValueError("density must be a positive scalar or one value per face")
    normal_velocity = np.einsum("ij,ij->i", u, n)
    volumetric_flux = float(np.sum(normal_velocity * a))
    mass_flux = float(np.sum(rho * normal_velocity * a))
    return {
        "volumetric_flux": volumetric_flux,
        "mass_flux": mass_flux,
        "inflow_mass": float(np.sum(np.maximum(-rho * normal_velocity, 0.0) * a)),
        "outflow_mass": float(np.sum(np.maximum(rho * normal_velocity, 0.0) * a)),
    }


def mass_balance(patches: dict[str, dict[str, np.ndarray]], *, density: float | np.ndarray = 1.0) -> dict[str, Any]:
    """Aggregate signed mass fluxes from several named boundary patches."""
    per_patch = {}
    total = 0.0
    for name, values in patches.items():
        result = integrate_mass_flux(
            values["normals"], values["areas"], values["velocity"],
            density=values.get("density", density),
        )
        per_patch[name] = result
        total += result["mass_flux"]
    return {"net_mass_flux": float(total), "patches": per_patch}


__all__.extend(["integrate_mass_flux", "mass_balance"])
