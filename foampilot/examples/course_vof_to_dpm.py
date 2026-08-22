"""Exercices pédagogiques pour le cours VOF-to-DPM de foampilot.

Usage:
    PYTHONPATH=src python examples/course_vof_to_dpm.py
    PYTHONPATH=src python examples/course_vof_to_dpm.py --case CASE --time 0.01
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np

try:
    from foampilot.utilities.vof_to_dpm import VofToDpmConverter
except ModuleNotFoundError:
    # Educational fallback: the converter itself only needs NumPy and stdlib.
    module_path = Path(__file__).parents[1] / "src" / "foampilot" / "utilities" / "vof_to_dpm.py"
    spec = importlib.util.spec_from_file_location("course_vof_to_dpm_module", module_path)
    if spec is None or spec.loader is None:
        raise
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    VofToDpmConverter = module.VofToDpmConverter


def synthetic_lesson() -> None:
    """Show volume and momentum weighting on two disconnected fragments."""
    alpha = np.array([1.0, 0.5, 1.0, 0.0])
    centres = np.array([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], dtype=float)
    volumes = np.array([2.0, 2.0, 1.0, 1.0])
    velocity = np.array([(1, 0, 0), (3, 0, 0), (5, 0, 0), (0, 0, 0)], dtype=float)
    neighbours = [[], [], [], []]

    converter = VofToDpmConverter(alpha_threshold=0.5)
    fragments = converter.extract(alpha, centres, volumes, neighbours, velocity)
    source_volume = float(np.sum(alpha * volumes))
    converted_volume = converter.total_volume(fragments)
    source_momentum = np.sum(alpha[:, None] * volumes[:, None] * velocity, axis=0)
    converted_momentum = sum(
        fragment.volume * np.asarray(fragment.velocity) for fragment in fragments
    )

    print("Synthetic VOF-to-DPM lesson")
    print(f"fragments              = {len(fragments)}")
    print(f"source liquid volume   = {source_volume:.6g}")
    print(f"converted volume       = {converted_volume:.6g}")
    print(f"volume residual        = {converted_volume - source_volume:.6g}")
    print(f"source weighted momentum = {source_momentum}")
    print(f"parcel momentum          = {converted_momentum}")
    for index, fragment in enumerate(fragments):
        print(
            f"fragment {index}: cells={fragment.cell_indices}, "
            f"V={fragment.volume:.6g}, d={fragment.equivalent_diameter:.6g}, "
            f"x={fragment.centroid}, U={fragment.velocity}"
        )


def case_lesson(case: str, time_directory: str) -> None:
    """Extract fragments from an ASCII OpenFOAM time directory."""
    converter = VofToDpmConverter(alpha_threshold=0.5, strict=True)
    fragments = converter.extract_case(case, time_directory=time_directory)
    outputs = converter.write_openfoam_outputs(
        fragments, f"{case}/constant", cloud_name="courseVofDpm"
    )
    print(f"case fragments = {len(fragments)}")
    print(f"liquid volume  = {converter.total_volume(fragments):.6e}")
    print(f"report         = {outputs['report']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case")
    parser.add_argument("--time", default="0")
    args = parser.parse_args()
    if args.case:
        case_lesson(args.case, args.time)
    else:
        synthetic_lesson()
