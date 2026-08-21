from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

case = Path(sys.argv[1]).resolve()
comms = case / "comms"
comms.mkdir(parents=True, exist_ok=True)
patch_faces = comms / "patchFaces"
if not patch_faces.exists():
    raise FileNotFoundError(patch_faces)
n_faces = int(patch_faces.read_text().splitlines()[1])
constant_temperature_k = float(sys.argv[2]) if len(sys.argv) > 2 else 307.75
last_mtime = 0
steps = 0
(comms / "data.in").write_text(
    "\n".join(f"{constant_temperature_k:.16g} 0 1" for _ in range(n_faces)) + "\n",
    encoding="utf-8",
)
(comms / "OpenFOAM.lock").touch()
while True:
    deadline = time.monotonic() + 30
    data_out = comms / "data.out"
    while not data_out.exists() or data_out.stat().st_mtime_ns <= last_mtime:
        if time.monotonic() > deadline:
            break
        time.sleep(0.02)
    if not data_out.exists() or data_out.stat().st_mtime_ns <= last_mtime:
        break
    size0 = data_out.stat().st_size
    time.sleep(0.1)
    if not data_out.exists() or data_out.stat().st_size != size0:
        continue
    last_mtime = data_out.stat().st_mtime_ns
    rows = []
    for line in data_out.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        tokens = line.replace("(", " ").replace(")", " ").split()
        if len(tokens) >= 4:
            rows.append(tokens)
    if len(rows) != n_faces:
        raise ValueError(f"data.out: {len(rows)} faces, attendu {n_faces}")
    (comms / "data.in").write_text(
        "\n".join(f"{constant_temperature_k:.16g} 0 1" for _ in range(n_faces)) + "\n",
        encoding="utf-8",
    )
    (comms / "OpenFOAM.lock").touch()
    steps += 1
    print(f"step: faces={n_faces} Tconstant={constant_temperature_k:.6g} K", flush=True)
print(f"constant coupler terminé: steps={steps}", flush=True)
