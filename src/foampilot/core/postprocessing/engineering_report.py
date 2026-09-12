"""Engineering CFD report assembled from post-processing results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .results import EngineeringResult


class EngineeringReport:
    """Collect CFD post-processing outputs into one auditable artifact."""

    def __init__(self, *, case: str | None = None, solver: str | None = None, openfoam_version: str | None = None):
        self.metadata = {
            "case": case,
            "solver": solver,
            "openfoam_version": openfoam_version,
        }
        self.results: dict[str, Any] = {}
        self.warnings: list[str] = []

    def add(self, name: str, result: EngineeringResult | dict[str, Any]) -> None:
        """Add a named result while preserving its metadata and values."""
        if isinstance(result, EngineeringResult):
            self.results[name] = result.to_dict()
        elif isinstance(result, dict):
            self.results[name] = result
        else:
            raise TypeError("result must be an EngineeringResult or dictionary")

    def warn(self, message: str) -> None:
        self.warnings.append(str(message))

    def to_dict(self) -> dict[str, Any]:
        return {"metadata": self.metadata, "results": self.results, "warnings": self.warnings}

    def export_json(self, filename: str | Path) -> Path:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return path

    def export_markdown(self, filename: str | Path) -> Path:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# CFD Engineering Report", "", "## Metadata", ""]
        for key, value in self.metadata.items():
            lines.append(f"- **{key}**: {value if value is not None else 'not specified'}")
        lines.extend(["", "## Results", ""])
        for name, result in self.results.items():
            lines.extend([f"### {name}", "", "```json", json.dumps(result, indent=2, default=str), "```", ""])
        if self.warnings:
            lines.extend(["## Warnings", "", *[f"- {warning}" for warning in self.warnings], ""])
        path.write_text("\n".join(lines), encoding="utf-8")
        return path


__all__ = ["EngineeringReport"]
