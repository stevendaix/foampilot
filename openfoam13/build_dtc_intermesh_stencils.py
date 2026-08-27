from __future__ import annotations

import json
from pathlib import Path
import re
import numpy as np


def read_scalar_field(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\(",
        text,
    )
    if match is None:
        raise ValueError(f"{path}: no nonuniform scalar internalField found")
    count = int(match.group(1))
    left = match.end()
    right = text.index(")", left)
    values = np.fromstring(text[left:right], sep=" ")
    if len(values) != count:
        raise ValueError(f"{path}: expected {count} values, got {len(values)}")
    return values


def read_centres(case: Path) -> np.ndarray:
    directory = case / "0"
    if not (directory / "Ccx").is_file():
        directory = case / "constant"
    return np.column_stack(
        [read_scalar_field(directory / name) for name in ("Ccx", "Ccy", "Ccz")]
    )


def build_mapping(background: np.ndarray, hull: np.ndarray, n_donors: int = 4):
    span = np.maximum(background.max(axis=0) - background.min(axis=0), 1e-12)
    scale = float(np.cbrt(np.prod(span) / max(len(background), 1)))
    cell_size = max(scale, 1e-3)
    origin = background.min(axis=0)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, point in enumerate(background):
        key = tuple(np.floor((point - origin) / cell_size).astype(int))
        buckets.setdefault(key, []).append(index)

    rows = []
    for index, point in enumerate(hull):
        key = np.floor((point - origin) / cell_size).astype(int)
        candidates: list[int] = []
        radius = 0
        while len(candidates) < n_donors and radius <= 4:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        candidates.extend(buckets.get(tuple(key + (dx, dy, dz)), []))
            radius += 1
        if len(candidates) < n_donors:
            raise RuntimeError(f"not enough background candidates for hull cell {index}")
        candidate_array = np.asarray(candidates, dtype=np.int64)
        distances = np.linalg.norm(background[candidate_array] - point, axis=1)
        order = np.argsort(distances)[:n_donors]
        selected = candidate_array[order]
        selected_distances = distances[order]
        if selected_distances[0] == 0:
            weights = np.zeros(n_donors)
            weights[0] = 1.0
        else:
            inverse = 1.0 / np.maximum(selected_distances, 1e-15)
            weights = inverse / inverse.sum()
        rows.append(
            {
                "acceptor": int(index),
                "donorIndices": [int(value) for value in selected],
                "weights": [float(value) for value in weights],
            }
        )
    return rows


if __name__ == "__main__":
    root = Path("/home/ubuntu/foampilot-audit/openfoam13/DTCMoving_Overset_Foundation13")
    background = read_centres(root / "background")
    hull = read_centres(root / "hull")
    rows = build_mapping(background, hull)
    output = root / "marineInterMeshStencils.json"
    output.write_text(
        json.dumps(
            {
                "format": "marineInterMeshStencils-v1",
                "acceptorMesh": "hull",
                "donorMesh": "background",
                "nDonors": 4,
                "backgroundCellCount": len(background),
                "hullCellCount": len(hull),
                "stencils": rows,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    print(f"wrote {output} with {len(rows)} stencils")
