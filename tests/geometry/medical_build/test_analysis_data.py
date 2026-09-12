import numpy as np

from foampilot.geometry.medical_build.analysis_data import (
    BranchRecord,
    GeometryAnalysisData,
    SectionRecord,
)


def make_section(branch_id: int, station_id: int, abscissa: float) -> SectionRecord:
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    points = np.column_stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))
    return SectionRecord(
        branch_id=branch_id,
        station_id=station_id,
        abscissa=abscissa,
        center=np.array([0.0, 0.0, abscissa]),
        tangent=np.array([0.0, 0.0, 1.0]),
        normal=np.array([1.0, 0.0, 0.0]),
        binormal=np.array([0.0, 1.0, 0.0]),
        points=points,
        phase_locked_points=points,
        area=float(np.pi),
        perimeter=float(2.0 * np.pi),
        equivalent_radius=1.0,
    )


def test_geometry_analysis_data_roundtrip(tmp_path):
    sections = [make_section(0, 0, 0.0), make_section(0, 1, 1.0)]
    branch = BranchRecord(
        branch_id=0,
        source_cap_id=4,
        target_cap_id=0,
        points=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        abscissas=np.array([0.0, 1.0]),
        tangents=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        length=1.0,
        sections=sections,
    )
    data = GeometryAnalysisData(source_cap_id=4, branches=[branch])
    data.validate()
    destination = data.save_json(tmp_path / "analysis.json")
    assert destination.exists()
    assert len(data.as_dict()["branches"]) == 1
    assert len(data.as_dict()["branches"][0]["sections"]) == 2
