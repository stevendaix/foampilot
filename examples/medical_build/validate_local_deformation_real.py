"""Validate optional local deformation on a real medical_build analysis contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from foampilot.geometry.medical_build import (
    BranchRecord,
    GeometryAnalysisData,
    LocalDeformationSpec,
    SectionRecord,
    apply_local_deformation,
    deformation_report,
)


def load_analysis(path: Path) -> GeometryAnalysisData:
    data = json.loads(path.read_text())
    branches = []
    for raw_branch in data["branches"]:
        sections = [SectionRecord(**raw_section) for raw_section in raw_branch.get("sections", [])]
        branches.append(BranchRecord(
            branch_id=raw_branch["branch_id"],
            source_cap_id=raw_branch["source_cap_id"],
            target_cap_id=raw_branch["target_cap_id"],
            points=raw_branch["points"],
            abscissas=raw_branch["abscissas"],
            tangents=raw_branch["tangents"],
            length=raw_branch["length"],
            sections=sections,
            parent_branch_id=raw_branch.get("parent_branch_id"),
            children_branch_ids=raw_branch.get("children_branch_ids", []),
            diagnostics=raw_branch.get("diagnostics", {}),
        ))
    result = GeometryAnalysisData(
        coordinate_system=data.get("coordinate_system", "input"),
        source_cap_id=data.get("source_cap_id"),
        cap_records=data.get("cap_records", []),
        branches=branches,
        diagnostics=data.get("diagnostics", {}),
        quality_metrics=data.get("quality_metrics", {}),
        phase_timings=data.get("phase_timings", {}),
        warnings=data.get("warnings", []),
        metadata=data.get("metadata", {}),
    )
    result.validate()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--branch-id", type=int, default=2)
    parser.add_argument("--amplitude", type=float, default=0.20)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--center-abscissa", type=float, default=None)
    parser.add_argument("--junction-protection", type=float, default=2.0)
    args = parser.parse_args()

    source = load_analysis(args.contract)
    selected = next(branch for branch in source.branches if branch.branch_id == args.branch_id)
    center = args.center_abscissa
    if center is None:
        center = 0.5 * (selected.sections[0].abscissa + selected.sections[-1].abscissa)

    deformed = apply_local_deformation(source, LocalDeformationSpec(
        branch_ids=(args.branch_id,),
        center_abscissa=center,
        sigma=args.sigma,
        amplitude=args.amplitude,
        junction_protection=args.junction_protection,
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "source_contract": str(args.contract),
        "reference_branch_count": len(source.branches),
        "deformed_branch_count": len(deformed.branches),
        "report": deformation_report(deformed),
        "deformed_analysis": deformed.as_dict(),
    }, indent=2))
    print(json.dumps({
        "branches": len(source.branches),
        "deformed_branch": args.branch_id,
        "report": deformation_report(deformed),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
