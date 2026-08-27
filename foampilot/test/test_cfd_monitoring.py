"""Unit tests for engineering-oriented CFD monitors."""
from __future__ import annotations

import numpy as np

from foampilot.postprocess.monitoring import CFDMonitor


class FakeMesh:
    def __init__(self, point_data, cell_data=None):
        self.point_data = point_data
        self.cell_data = cell_data or {}

    def find_closest_point(self, point):
        return 0


class FakePostprocessor:
    def __init__(self):
        self.meshes = {
            0: FakeMesh({"p": np.array([100.0, 90.0]), "U": np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])}),
            1: FakeMesh({"p": np.array([95.0, 85.0]), "U": np.array([[6.0, 8.0, 0.0], [0.0, 0.0, 0.0]])}),
        }

    def get_all_time_steps(self):
        return [0, 1]

    def load_time_step(self, step):
        return {"cell": self.meshes[step], "boundaries": {}}


def test_statistics_include_engineering_percentiles_and_rms():
    stats = CFDMonitor.statistics(np.array([3.0, 4.0]))
    assert stats["mean"] == 3.5
    assert stats["p50"] == 3.5
    assert stats["rms"] == np.sqrt(12.5)


def test_track_region_supports_vector_magnitude():
    frame = CFDMonitor(FakePostprocessor()).track_region("U", magnitude=True)
    assert list(frame.index) == [0, 1]
    assert frame.loc[0, "max"] == 5.0
    assert frame.loc[1, "max"] == 10.0


def test_track_point_returns_values_over_time():
    frame = CFDMonitor(FakePostprocessor()).track_point((0.0, 0.0, 0.0), "p")
    assert list(frame["value"]) == [100.0, 95.0]
