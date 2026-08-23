#!/usr/bin/env python3
"""Export the current MakeHuman body through the local server socket."""
from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import numpy as np
import trimesh


def call(host: str, port: int, function: str, *, binary: bool = False):
    request = {"function": function, "error": "", "params": {}, "data": None}
    with socket.create_connection((host, port), timeout=30) as client:
        client.sendall(json.dumps(request).encode("utf-8"))
        chunks = []
        while True:
            chunk = client.recv(1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    payload = b"".join(chunks)
    if binary:
        return payload
    response = json.loads(payload.decode("utf-8"))
    if response.get("error"):
        raise RuntimeError(response["error"])
    return response["data"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--out", type=Path, default=Path("makehuman_output"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    metadata = call(args.host, args.port, "getBodyMeshInfo")
    vertices = np.frombuffer(
        call(args.host, args.port, "getBodyVerticesBinary", binary=True),
        dtype=np.dtype(metadata["verticesTypeCode"]),
    ).reshape(tuple(metadata["verticesShape"]))
    faces = np.frombuffer(
        call(args.host, args.port, "getBodyFacesBinary", binary=True),
        dtype=np.dtype(metadata["facesTypeCode"]),
    ).reshape(tuple(metadata["facesShape"]))[:, :3]

    full = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    full.remove_unreferenced_vertices()
    full.export(args.out / "makehuman_base.stl")
    full.export(args.out / "makehuman_base.obj")

    body_ranges = None
    for group in metadata.get("faceGroups", []):
        if group.get("name") == "body" and group.get("fgStartStops"):
            body_ranges = group["fgStartStops"]
            break
    if body_ranges is None:
        raise RuntimeError("Le groupe de faces MakeHuman 'body' est absent")

    # MakeHuman stores inclusive [start, stop] ranges. Do not assume that
    # the body group is one contiguous prefix of the face array.
    body_face_ids = sorted({
        face_id
        for start, stop in body_ranges
        for face_id in range(int(start), int(stop) + 1)
        if 0 <= int(face_id) < len(faces)
    })
    body_faces = faces[np.asarray(body_face_ids, dtype=np.int64)]
    body = trimesh.Trimesh(vertices=vertices, faces=body_faces, process=False)
    body.merge_vertices(digits_vertex=8)
    if hasattr(body, "nondegenerate_faces"):
        body.update_faces(body.nondegenerate_faces())
    body.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(body)
    body.export(args.out / "makehuman_body_only.stl")
    body.export(args.out / "makehuman_body_only.obj")

    report = {
        "vertices": int(len(body.vertices)),
        "triangles": int(len(body.faces)),
        "body_face_ranges": body_ranges,
        "body_face_count_selected": int(len(body_face_ids)),
        "watertight": bool(body.is_watertight),
        "winding_consistent": bool(body.is_winding_consistent),
        "bounds_min": body.bounds[0].tolist(),
        "bounds_max": body.bounds[1].tolist(),
        "source_groups": metadata.get("faceGroups", []),
    }
    (args.out / "makehuman_export_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
