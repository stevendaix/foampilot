import numpy as np

from foampilot.geometry.medical_build import (
    BranchRecord,
    GeometryAnalysisData,
    LocalDeformationSpec,
    SectionRecord,
    apply_local_deformation,
    deformation_report,
)


def make_analysis():
    sections = []
    for station, s in enumerate(np.linspace(0.0, 10.0, 5)):
        center = np.array([0.0, 0.0, s])
        theta = np.linspace(0.0, 2.0 * np.pi, 17)[:-1]
        points = np.column_stack((0.5 * np.cos(theta), 0.5 * np.sin(theta), np.full_like(theta, s)))
        sections.append(SectionRecord(
            branch_id=1,
            station_id=station,
            abscissa=float(s),
            center=center,
            tangent=np.array([0.0, 0.0, 1.0]),
            normal=np.array([1.0, 0.0, 0.0]),
            binormal=np.array([0.0, 1.0, 0.0]),
            points=points,
            phase_locked_points=points.copy(),
            area=np.pi * 0.25,
            perimeter=2.0 * np.pi * 0.5,
            equivalent_radius=0.5,
        ))
    branch = BranchRecord(
        branch_id=1,
        source_cap_id=0,
        target_cap_id=1,
        points=np.array([[0.0, 0.0, s] for s in np.linspace(0.0, 10.0, 5)]),
        abscissas=np.linspace(0.0, 10.0, 5),
        tangents=np.tile(np.array([0.0, 0.0, 1.0]), (5, 1)),
        length=10.0,
        sections=sections,
    )
    return GeometryAnalysisData(branches=[branch])


def test_none_is_exact_noop_and_does_not_mutate():
    source = make_analysis()
    before = source.as_dict()
    result = apply_local_deformation(source, None)
    assert result.as_dict() == before
    assert source.as_dict() == before


def test_gaussian_deformation_is_local_and_protected_at_junctions():
    source = make_analysis()
    spec = LocalDeformationSpec(
        branch_ids=(1,), center_abscissa=5.0, sigma=1.5,
        amplitude=0.25, junction_protection=2.0,
    )
    result = apply_local_deformation(source, spec)
    scales = [s.metadata.get("local_deformation_scale", 1.0) for s in result.branches[0].sections]
    assert scales[0] == 1.0
    assert scales[-1] == 1.0
    assert scales[2] > 1.0
    assert result.branches[0].sections[2].area > source.branches[0].sections[2].area
    assert source.branches[0].sections[2].area == np.pi * 0.25
    report = deformation_report(result)
    assert report["enabled"] is True
    assert report["max_scale"] > 1.0
