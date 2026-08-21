"""Reconstruct reference and locally deformed aorta STLs from analysis sections."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from section_stl_reconstruction import quality, reconstruct_branch, write_binary_stl
from validate_local_deformation_real import load_analysis

from foampilot.geometry.medical_build import (
    LocalDeformationSpec,
    apply_local_deformation,
    deformation_report,
)


def reconstruct_analysis(analysis, output: Path, points: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    branch_report = []
    combined_vertices = []
    combined_triangles = []
    vertex_offset = 0
    for branch in analysis.branches:
        vertices, triangles = reconstruct_branch(
            [section.as_dict() for section in branch.sections],
            n_points=points,
        )
        path = output / f"branch_{branch.branch_id:02d}.stl"
        write_binary_stl(vertices, triangles, path)
        branch_report.append({
            "branch_id": branch.branch_id,
            "path": str(path),
            **quality(vertices, triangles),
            "section_count": len(branch.sections),
        })
        combined_vertices.append(vertices)
        combined_triangles.append(triangles + vertex_offset)
        vertex_offset += len(vertices)
    vertices = np.vstack(combined_vertices)
    triangles = np.vstack(combined_triangles)
    combined_path = output / "aorta_sections_combined.stl"
    write_binary_stl(vertices, triangles, combined_path)
    return {
        "branches": branch_report,
        "combined": {"path": str(combined_path), **quality(vertices, triangles)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--branch-id", type=int, default=2)
    parser.add_argument("--amplitude", type=float, default=0.20)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--junction-protection", type=float, default=2.0)
    parser.add_argument("--points", type=int, default=32)
    args = parser.parse_args()

    source = load_analysis(args.contract)
    selected = next(branch for branch in source.branches if branch.branch_id == args.branch_id)
    center = 0.5 * (selected.sections[0].abscissa + selected.sections[-1].abscissa)
    spec = LocalDeformationSpec(
        branch_ids=(args.branch_id,),
        center_abscissa=center,
        sigma=args.sigma,
        amplitude=args.amplitude,
        junction_protection=args.junction_protection,
    )
    deformed = apply_local_deformation(source, spec)
    reference_report = reconstruct_analysis(source, args.output / "reference", args.points)
    deformed_report = reconstruct_analysis(deformed, args.output / "deformed", args.points)
    result = {
        "contract": str(args.contract),
        "reference_branch_count": len(source.branches),
        "deformation": deformation_report(deformed),
        "reference": reference_report,
        "deformed": deformed_report,
    }
    (args.output / "reconstruction_comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
