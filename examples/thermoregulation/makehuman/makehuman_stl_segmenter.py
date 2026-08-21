#!/usr/bin/env python3
"""Classify a MakeHuman body STL into 17 JOS-3 surface patches."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import trimesh

SEGMENTS = [
    "Head", "Neck", "Chest", "Back", "Pelvis",
    "LShoulder", "LArm", "LHand", "RShoulder", "RArm", "RHand",
    "LThigh", "LLeg", "LFoot", "RThigh", "RLeg", "RFoot",
]


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Maillage non triangulaire : {path}")
    mask = mesh.nondegenerate_faces()
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()
    return mesh


def normalize(mesh: trimesh.Trimesh):
    vertices = mesh.vertices.astype(float)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    vertical = int(np.argmax(span))
    remaining = [axis for axis in range(3) if axis != vertical]
    lateral = remaining[int(np.argmax(span[remaining]))]
    depth = remaining[0] if remaining[1] == lateral else remaining[1]
    p = np.zeros_like(vertices)
    p[:, 0] = (vertices[:, lateral] - (lo[lateral] + hi[lateral]) / 2) / span[lateral]
    p[:, 1] = (vertices[:, depth] - (lo[depth] + hi[depth]) / 2) / span[depth]
    p[:, 2] = (vertices[:, vertical] - lo[vertical]) / span[vertical]
    return p, lo, hi, {"vertical": "xyz"[vertical], "lateral": "xyz"[lateral], "depth": "xyz"[depth]}


def classify(centroids: np.ndarray) -> np.ndarray:
    x, y, z = centroids.T
    result = np.full(len(centroids), "Pelvis", dtype=object)
    result[z >= 0.84] = "Head"
    result[(z >= 0.76) & (z < 0.84)] = "Neck"
    result[(z >= 0.56) & (z < 0.76) & (y >= 0)] = "Chest"
    result[(z >= 0.56) & (z < 0.76) & (y < 0)] = "Back"
    result[(z >= 0.42) & (z < 0.56)] = "Pelvis"

    upper = (z >= 0.47) & (z < 0.72) & (np.abs(x) > 0.16)
    lower = (z >= 0.27) & (z < 0.47) & (np.abs(x) > 0.10)
    feet = (z < 0.13) & (np.abs(x) > 0.08)
    hands = (z >= 0.28) & (z < 0.58) & (np.abs(x) > 0.36)
    result[upper & (x < 0)] = "LShoulder"
    result[upper & (x >= 0)] = "RShoulder"
    result[lower & (x < 0)] = "LThigh"
    result[lower & (x >= 0)] = "RThigh"
    result[feet & (x < 0)] = "LFoot"
    result[feet & (x >= 0)] = "RFoot"
    result[hands & (x < 0)] = "LHand"
    result[hands & (x >= 0)] = "RHand"
    arm = (np.abs(x) > 0.22) & ~hands & ~feet & (z >= 0.13) & (z < 0.58)
    result[arm & (x < 0)] = "LArm"
    result[arm & (x >= 0)] = "RArm"
    leg = (np.abs(x) > 0.08) & ~feet & (z < 0.42)
    result[leg & (x < 0)] = "LLeg"
    result[leg & (x >= 0)] = "RLeg"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_stl", type=Path)
    parser.add_argument("--out", type=Path, default=Path("jos3_zones"))
    parser.add_argument("--export-global", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    patch_dir = args.out / "patches"
    patch_dir.mkdir(exist_ok=True)

    mesh = load_mesh(args.input_stl)
    normalized, lo, hi, axes = normalize(mesh)
    centroids = normalized[mesh.faces].mean(axis=1)
    labels = classify(centroids)

    manifest = {"source": str(args.input_stl), "segments": [], "normalized_axes": axes}
    rows = []
    for idx, segment in enumerate(SEGMENTS):
        face_ids = np.flatnonzero(labels == segment)
        patch = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[face_ids], process=True)
        patch.remove_unreferenced_vertices()
        filename = f"skin_{segment}.stl"
        patch.export(patch_dir / filename)
        manifest["segments"].append({"id": idx, "jos3_name": segment, "stl": f"patches/{filename}", "triangles": int(len(face_ids))})
        rows.extend({"face_id": int(face_id), "zone_id": idx, "jos3_name": segment} for face_id in face_ids)

    if args.export_global:
        mesh.export(args.out / "body_global_clean.stl")
        manifest["global_surface"] = "body_global_clean.stl"
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (args.out / "zone_mapping.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["face_id", "zone_id", "jos3_name"])
        writer.writeheader(); writer.writerows(rows)
    quality = {
        "triangles": int(len(mesh.faces)), "vertices": int(len(mesh.vertices)),
        "watertight": bool(mesh.is_watertight), "winding_consistent": bool(mesh.is_winding_consistent),
        "bounds_min": lo.tolist(), "bounds_max": hi.tolist(), "face_counts": {s: int(np.sum(labels == s)) for s in SEGMENTS},
        "warning": "Les patches sont des sous-surfaces; valider le STL global/STEP comme volume CFD.",
    }
    (args.out / "quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(json.dumps({"input": str(args.input_stl), "triangles": int(len(mesh.faces)), "zones": 17, "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
