import numpy as np

from foampilot.geometry.medical_build.section_filtering import (
    SectionFilterConfig,
    classify_station,
    contour_metrics,
    continuity_rejection,
)


def ring(radius=10.0, n=32, z=0.0):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.full(n, z)))
    return np.vstack((points, points[0]))


def test_regular_closed_contour_is_valid():
    current = contour_metrics(ring(10.0), [0, 0, 0], [0, 0, 1])
    previous = contour_metrics(ring(10.0, z=-1.0), [0, 0, -1], [0, 0, 1])
    next_ = contour_metrics(ring(10.0, z=1.0), [0, 0, 1], [0, 0, 1])
    selected, reason = classify_station([current], [0, 0, 0], [0, 0, 1], 10.0, previous, next_)
    assert selected is current
    assert reason == "VALID"
    assert current.status == "VALID"


def test_radius_spike_is_rejected():
    current = contour_metrics(ring(30.0), [0, 0, 0], [0, 0, 1])
    previous = contour_metrics(ring(10.0, z=-1.0), [0, 0, -1], [0, 0, 1])
    next_ = contour_metrics(ring(10.0, z=1.0), [0, 0, 1], [0, 0, 1])
    selected, reason = classify_station([current], [0, 0, 0], [0, 0, 1], 10.0, previous, next_)
    assert selected is None
    assert reason == "RADIUS_SPIKE"
    assert current.status == "REJECTED"


def test_open_contour_is_not_accepted():
    points = ring(10.0)[:-4]
    current = contour_metrics(points, [0, 0, 0], [0, 0, 1])
    rejected, reason = continuity_rejection(current, None, None, SectionFilterConfig())
    assert rejected is True
    assert reason == "OPEN_CONTOUR"
