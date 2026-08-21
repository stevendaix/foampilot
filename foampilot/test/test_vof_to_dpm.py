"""Tests for conservative VOF-to-DPM fragment extraction."""

import json
from pathlib import Path

import numpy as np
import pytest

from vof_to_dpm import VofToDpmConverter


def line_neighbours(n: int) -> list[list[int]]:
    neighbours = [[] for _ in range(n)]
    for index in range(n - 1):
        neighbours[index].append(index + 1)
        neighbours[index + 1].append(index)
    return neighbours


def test_two_fragments_conserve_volume_and_weighted_momentum(tmp_path: Path):
    converter = VofToDpmConverter(alpha_threshold=0.5)
    fragments = converter.extract(
        alpha=[1.0, 0.5, 1.0, 0.0],
        cell_centres=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
        cell_volumes=[2.0, 2.0, 1.0, 1.0],
        neighbours=[[], [], [], []],
        velocity=[(1, 0, 0), (3, 0, 0), (5, 0, 0), (0, 0, 0)],
    )

    assert len(fragments) == 3
    assert converter.total_volume(fragments) == pytest.approx(4.0)
    assert fragments[0].volume == pytest.approx(2.0)
    assert fragments[0].centroid == pytest.approx((0.0, 0.0, 0.0))
    assert fragments[0].velocity == pytest.approx((1.0, 0.0, 0.0))
    assert fragments[1].volume == pytest.approx(1.0)
    assert fragments[1].velocity == pytest.approx((3.0, 0.0, 0.0))
    assert fragments[2].volume == pytest.approx(1.0)

    outputs = converter.write_openfoam_outputs(fragments, tmp_path)
    assert set(outputs) == {"positions", "fragments", "report"}
    report = json.loads(outputs["report"].read_text())
    assert report["fragmentCount"] == 3
    assert report["liquidVolume"] == pytest.approx(4.0)
    assert outputs["positions"].read_text().count("(") >= 4


def test_connected_cells_form_one_volume_weighted_fragment():
    converter = VofToDpmConverter(alpha_threshold=0.25)
    fragments = converter.extract(
        alpha=[0.5, 1.0, 0.0],
        cell_centres=[(0, 0, 0), (2, 0, 0), (4, 0, 0)],
        cell_volumes=[2.0, 2.0, 2.0],
        neighbours=line_neighbours(3),
    )

    assert len(fragments) == 1
    assert fragments[0].cell_indices == (0, 1)
    assert fragments[0].volume == pytest.approx(3.0)
    assert fragments[0].centroid == pytest.approx((4.0 / 3.0, 0.0, 0.0))


def test_invalid_alpha_and_neighbour_indices_are_rejected():
    converter = VofToDpmConverter()
    with pytest.raises(ValueError, match="alpha values"):
        converter.extract(
            alpha=[1.1],
            cell_centres=[(0, 0, 0)],
            cell_volumes=[1],
            neighbours=[[]],
        )
    with pytest.raises(ValueError, match="invalid cell index"):
        converter.extract(
            alpha=[1.0],
            cell_centres=[(0, 0, 0)],
            cell_volumes=[1],
            neighbours=[[2]],
        )


def test_filters_are_explicit_and_do_not_change_retained_volume():
    converter = VofToDpmConverter(alpha_threshold=0.5, min_volume=2.0)
    fragments = converter.extract(
        alpha=[1.0, 1.0],
        cell_centres=[(0, 0, 0), (1, 0, 0)],
        cell_volumes=[1.0, 3.0],
        neighbours=[[], []],
    )
    assert len(fragments) == 1
    assert fragments[0].volume == pytest.approx(3.0)
