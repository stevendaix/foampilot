from __future__ import annotations
import json
from pathlib import Path
import trimesh


def main() -> None:
    root = Path(__file__).resolve().parent / "outputs" / "reconstructed_local_deformation"
    report = json.loads((root / "reconstruction_comparison.json").read_text())
    rows = []
    for branch in report["reference"]["branches"]:
        bid = branch["branch_id"]
        ref = trimesh.load(root / "reference" / f"branch_{bid:02d}.stl", force="mesh")
        deformed = trimesh.load(root / "deformed" / f"branch_{bid:02d}.stl", force="mesh")
        rows.append({
            "branch_id": bid,
            "reference_volume": float(abs(ref.volume)),
            "deformed_volume": float(abs(deformed.volume)),
            "relative_change_percent": float(100.0 * (abs(deformed.volume) - abs(ref.volume)) / max(abs(ref.volume), 1e-12)),
            "reference_area": float(ref.area),
            "deformed_area": float(deformed.area),
            "reference_watertight": bool(ref.is_watertight),
            "deformed_watertight": bool(deformed.is_watertight),
        })
    output = {"branches": rows}
    (root / "reconstructed_volume_comparison.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
