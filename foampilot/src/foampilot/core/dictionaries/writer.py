"""Generic OpenFOAM dictionary writing primitives."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def _render_value(value: Any, indent: int = 0) -> str:
    prefix = "    " * indent
    if isinstance(value, Mapping):
        lines = ["{"]
        for key, item in value.items():
            rendered = _render_value(item, indent + 1)
            if "\n" in rendered:
                lines.append(f"{'    ' * (indent + 1)}{key}\n{'    ' * (indent + 1)}{rendered}")
            else:
                lines.append(f"{'    ' * (indent + 1)}{key} {rendered};")
        lines.append(f"{prefix}}}")
        return "\n".join(lines)
    if isinstance(value, (list, tuple)):
        return "(" + " ".join(_render_value(item, indent) for item in value) + ")"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class DictionaryWriter:
    """Write a small, explicit OpenFOAM dictionary."""

    def __init__(self, object_name: str, attributes: Mapping[str, Any] | None = None) -> None:
        if not object_name or "/" in object_name or "\\" in object_name:
            raise ValueError("object_name must be a plain dictionary filename")
        self.object_name = object_name
        self.attributes = dict(attributes or {})

    def render(self) -> str:
        lines = [
            "FoamFile",
            "{",
            "    version     2.0;",
            "    format      ascii;",
            "    class       dictionary;",
            f"    object      {self.object_name};",
            "}",
            "",
        ]
        for key, value in self.attributes.items():
            rendered = _render_value(value)
            if "\n" in rendered:
                lines.extend([str(key), rendered])
            else:
                lines.append(f"{key} {rendered};")
        return "\n".join(lines) + "\n"

    def write(self, destination: str | Path) -> Path:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(), encoding="utf-8")
        return path
