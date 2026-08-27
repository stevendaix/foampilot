from pathlib import Path
import importlib.util
import sys


ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "foampilot/src/foampilot/mesh/marine_overset.py"
spec = importlib.util.spec_from_file_location("marine_overset_under_test", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)

OversetZone = module.OversetZone
build_zone_id = module.build_zone_id
inverse_distance_interpolate = module.inverse_distance_interpolate
validate_zones = module.validate_zones
write_zone_id_field = module.write_zone_id_field
write_donor_stencils = module.write_donor_stencils
build_donor_stencil = module.build_donor_stencil
build_donor_stencils = module.build_donor_stencils
write_marine_overset_constraint = module.write_marine_overset_constraint
write_intermesh_stencils = module.write_intermesh_stencils


def test_zone_ids_are_consecutive_and_overlap_prefers_moving_zone():
    zones = [
        OversetZone("background", 0, (-2.0, -2.0), (2.0, 2.0)),
        OversetZone("hull", 1, (-0.5, -0.5), (0.5, 0.5)),
    ]
    points = [(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
    assert build_zone_id(points, zones) == [0, 1, 0]


def test_constant_scalar_is_reproduced_exactly():
    points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    values = [7.5] * len(points)
    assert inverse_distance_interpolate((0.4, 0.3), points, values) == 7.5


def test_vector_interpolation_has_finite_components():
    points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    values = [(x, 2.0 * y) for x, y in points]
    result = inverse_distance_interpolate((0.5, 0.5), points, values)
    assert all(abs(value) < 10.0 for value in result)
    assert abs(result[0] - 0.5) < 1e-12
    assert abs(result[1] - 1.0) < 1e-12


def test_zone_id_field_is_written_in_openfoam_shape(tmp_path):
    zones = [
        OversetZone("background", 0, (-2.0, -2.0), (2.0, 2.0)),
        OversetZone("hull", 1, (-0.5, -0.5), (0.5, 0.5)),
    ]
    output = write_zone_id_field(str(tmp_path), [(-1.0, 0.0), (0.0, 0.0)], zones)
    text = Path(output).read_text()
    assert "class volScalarField;" in text
    assert "internalField   nonuniform List<scalar>" in text
    assert "2\\n(\\n0\\n1\\n);" in text


def test_invalid_zone_ids_are_rejected():
    zones = [
        OversetZone("background", 0, (0.0, 0.0), (1.0, 1.0)),
        OversetZone("hull", 2, (0.2, 0.2), (0.8, 0.8)),
    ]
    try:
        validate_zones(zones)
    except ValueError as error:
        assert "consecutive" in str(error)
    else:
        raise AssertionError("non-consecutive zone ids must be rejected")


def test_donor_stencil_is_normalized_and_selects_nearest_cells():
    donors = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (5.0, 5.0)]
    stencil = build_donor_stencil((0.4, 0.3), donors, n_donors=4)
    assert len(stencil.donor_indices) == 4
    assert len(stencil.weights) == 4
    assert abs(sum(stencil.weights) - 1.0) < 1e-12
    assert 4 not in stencil.donor_indices

    exact = build_donor_stencil((1.0, 0.0), donors, n_donors=4)
    assert exact.donor_indices == (1,)
    assert exact.weights == (1.0,)


def test_donor_stencil_dictionary_is_written_as_a_readable_contract(tmp_path):
    output = write_donor_stencils(
        str(tmp_path),
        [(0.25, 0.25)],
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
        n_donors=4,
    )
    text = Path(output).read_text()
    assert "object marineOversetStencils;" in text
    assert "nAcceptors 1;" in text
    assert "donorIndices (0 1 2 3);" in text
    assert "weights (" in text
    assert "\n" in text
    assert "\\\\n" not in text


def test_multiple_acceptors_receive_independent_stencils():
    donors = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    stencils = build_donor_stencils([(0.2, 0.2), (0.8, 0.8)], donors, n_donors=3)
    assert len(stencils) == 2
    assert all(len(stencil.donor_indices) == 3 for stencil in stencils)
    assert all(abs(sum(stencil.weights) - 1.0) < 1e-12 for stencil in stencils)
    assert stencils[0] != stencils[1]


def test_donor_stencil_rejects_insufficient_overlap():
    try:
        build_donor_stencil((0.0, 0.0), [(1.0, 1.0)], n_donors=2, max_distance=2.0)
    except ValueError as error:
        assert "not enough donors" in str(error)
    else:
        raise AssertionError("insufficient overlap must be rejected")


def test_write_intermesh_stencils_contract(tmp_path):
    stencil = module.DonorStencil((4, 5), (0.25, 0.75))
    output = write_intermesh_stencils(
        str(tmp_path), (stencil,), (12,), donor_region="background", acceptor_region="hull"
    )
    text = Path(output).read_text(encoding="utf-8")
    assert "donorRegion background;" in text
    assert "acceptorRegion hull;" in text
    assert "index 12;" in text
    assert "donorIndices (4 5);" in text
    assert "weights (0.25 0.75);" in text


def test_write_marine_overset_constraint(tmp_path):
    output = write_marine_overset_constraint(
        str(tmp_path), fields=("U", "p_rgh"), library="libmarineOversetProbe.so"
    )
    text = Path(output).read_text(encoding="utf-8")
    assert "type marineOversetConstraint;" in text
    assert 'libs ("libmarineOversetProbe.so");' in text
    assert "fields (U p_rgh);" in text


def test_target_dimension_and_donor_count_are_validated():
    try:
        inverse_distance_interpolate((0.0, 0.0), [(0.0,)], [1.0])
    except ValueError as error:
        assert "dimension" in str(error)
    else:
        raise AssertionError("dimension mismatch must be rejected")

    try:
        inverse_distance_interpolate((0.0, 0.0), [(0.0, 0.0)], [1.0], n_donors=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("invalid donor count must be rejected")
