import numpy as np

from foampilot.geometry.medical_build import (
    BranchRecord,
    GeometryAnalysisData,
    SectionRecord,
    build_vascular_graph,
)


def section(branch_id, station, z):
    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    points = np.column_stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))
    return SectionRecord(
        branch_id=branch_id,
        station_id=station,
        abscissa=float(station),
        center=np.array([0.0, 0.0, z]),
        tangent=np.array([0.0, 0.0, 1.0]),
        normal=np.array([1.0, 0.0, 0.0]),
        binormal=np.array([0.0, 1.0, 0.0]),
        points=points,
        phase_locked_points=points,
        area=np.pi,
        perimeter=2 * np.pi,
        equivalent_radius=1.0,
    )


def branch(branch_id, source, target, start, end, parent=None):
    return BranchRecord(
        branch_id=branch_id,
        source_cap_id=source,
        target_cap_id=target,
        points=np.asarray([start, end], dtype=float),
        abscissas=np.array([0.0, 1.0]),
        tangents=np.asarray([[0, 0, 1], [0, 0, 1]], dtype=float),
        length=float(np.linalg.norm(np.asarray(end) - np.asarray(start))),
        sections=[section(branch_id, 0, 0), section(branch_id, 1, 1)],
        parent_branch_id=parent,
    )


def test_vascular_graph_detects_connected_bifurcation():
    data = GeometryAnalysisData(
        branches=[
            branch(0, 0, 1, [0, 0, 0], [0, 0, 1]),
            branch(1, 1, 2, [0, 0, 1], [1, 0, 2], parent=0),
            branch(2, 1, 3, [0, 0, 1], [-1, 0, 2], parent=0),
        ]
    )
    result = build_vascular_graph(data)
    validation = result.validate()
    assert validation.connected
    assert validation.acyclic
    assert validation.branch_count == 3
    assert len(validation.bifurcations) == 1
    assert len(validation.terminals) == 2


def test_vascular_graph_merges_spatially_close_caps():
    data = GeometryAnalysisData(
        branches=[
            branch(0, 0, 1, [0, 0, 0], [0, 0, 1]),
            branch(1, 2, 3, [0, 0, 1.01], [1, 0, 2]),
        ]
    )
    result = build_vascular_graph(data, endpoint_tolerance=0.02)
    assert result.validate().connected
    assert result.validate().component_count == 1
