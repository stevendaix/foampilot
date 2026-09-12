"""Typed, JSON-ready result objects shared by CFD post-processing reports."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field as dataclass_field
from typing import Any, Mapping

import numpy as np


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


@dataclass
class ResultMetadata:
    """Provenance shared by every engineering result."""

    case: str | None = None
    time: float | str | None = None
    region: str | None = None
    patch: str | None = None
    field: str | None = None
    association: str | None = None
    units: str | None = None
    method: str | None = None
    source: str | None = None
    warnings: list[str] = dataclass_field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass
class EngineeringResult:
    """Generic payload used by reports and dashboards."""

    metadata: ResultMetadata = dataclass_field(default_factory=ResultMetadata)
    values: dict[str, Any] = dataclass_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"metadata": self.metadata.to_dict(), "values": _json_value(self.values)}


@dataclass
class MassBalanceResult(EngineeringResult):
    """Signed mass-flow balance with per-patch contributions."""

    values: dict[str, Any] = dataclass_field(default_factory=lambda: {"net_mass_flux": 0.0, "patches": {}})


@dataclass
class TimeSeriesResult(EngineeringResult):
    """A named temporal series with explicit units and provenance."""

    values: dict[str, Any] = dataclass_field(default_factory=lambda: {"time": [], "data": []})


__all__ = ["ResultMetadata", "EngineeringResult", "MassBalanceResult", "TimeSeriesResult"]
