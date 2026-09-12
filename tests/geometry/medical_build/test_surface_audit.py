import pyvista as pv
from foampilot.geometry.medical_build.surface_audit import audit_surface


def test_closed_surface_has_valid_topology():
    result = audit_surface(pv.Sphere(theta_resolution=16, phi_resolution=12))
    assert result['closed']
    assert result['open_edges'] == 0
    assert result['non_manifold_edges'] == 0
    assert result['duplicate_triangles'] == 0
    assert result['volume'] > 0
    assert result['quality_ok']


def test_open_surface_is_rejected():
    result = audit_surface(pv.Sphere(theta_resolution=12, phi_resolution=8).extract_surface().clip(normal='x', invert=False))
    assert not result['closed'] or result['open_edges'] > 0
    assert not result['quality_ok']
