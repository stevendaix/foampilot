#!/usr/bin/env python3
"""Test the new topology detection module on patient58 mesh."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "foampilot" / "src"))

import numpy as np
import pyvista as pv
from foampilot.geometry.topology import OpenProfileClassifier, SurfaceTopologyAnalyzer
from foampilot.postprocess import OpenFOAMDirectReader

CASE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "examples" / "coa" / "patient58_cfd_example"


def main() -> int:
    reader = OpenFOAMDirectReader(case_path=str(CASE_DIR))
    mesh = reader.to_pyvista(as_point_data=False)
    surface = mesh.extract_geometry()
    analyzer = SurfaceTopologyAnalyzer(surface)
    profiles = analyzer.find_open_profiles(surface)
    print(f"Detected {len(profiles)} open profiles")
    for p in profiles:
        print(
            f"  id={p.id} area={p.area:.6f} perimeter={p.perimeter:.6f} "
            f"circularity={p.circularity:.3f} planarity={p.planarity:.3f} "
            f"centroid={p.centroid} normal={p.normal}"
        )
    classifier = OpenProfileClassifier()
    classified = classifier.classify(profiles)
    for p in classified:
        print(f"  -> role={p.role.value} confidence={p.confidence:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
