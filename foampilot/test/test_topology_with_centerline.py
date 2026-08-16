import pytest
import numpy as np
import trimesh
import pyvista as pv

from foampilot.geometry.topology import (
    BoundaryRole,
    OpenProfile,
    OpenProfileClassifier,
    TopologyCenterlineExtractor,
    TopologySectionExtractor,
    SurfaceTopologyAnalyzer,
)


def _make_cylinder_mesh(radius: float = 1.0, height: float = 5.0, n_points: int = 16):
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    bottom = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.zeros(n_points)))
    top = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.full(n_points, height)))
    verts = np.vstack([bottom, top])
    faces = []
    for i in range(n_points):
        a = i
        b = (i + 1) % n_points
        c = i + n_points
        d = (i + 1) % n_points + n_points
        faces.append([a, b, d])
        faces.append([a, d, c])
    bottom_cap = list(range(n_points))
    top_cap = list(range(n_points, 2 * n_points))
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


class TestTopologyCenterlineExtractor:
    def test_vmtk_axis_extraction(self):
        extractor = TopologyCenterlineExtractor()
        mesh = _make_cylinder_mesh()
        axis, origin = extractor.extract_axis(mesh)
        assert axis.shape == (3,)
        assert origin.shape == (3,)
        assert abs(np.linalg.norm(axis) - 1.0) < 1e-6

    def test_classify_profiles_two_inlets(self):
        extractor = TopologyCenterlineExtractor()
        mesh = _make_cylinder_mesh()
        profiles = [
            OpenProfile(id=0, centroid=np.array([0.0, 0.0, 0.0]), normal=np.array([0.0, 0.0, -1.0]), area=0.1, perimeter=1.0),
            OpenProfile(id=1, centroid=np.array([0.0, 0.0, 5.0]), normal=np.array([0.0, 0.0, 1.0]), area=0.2, perimeter=1.5),
        ]
        classified = extractor.classify_profiles(profiles, mesh)
        roles = [p.role for p in classified]
        assert BoundaryRole.INLET in roles
        assert all(p.role in (BoundaryRole.INLET, BoundaryRole.OUTLET, BoundaryRole.UNKNOWN) for p in classified)

    def test_classify_profiles_single(self):
        extractor = TopologyCenterlineExtractor()
        mesh = _make_cylinder_mesh()
        profiles = [
            OpenProfile(id=0, centroid=np.array([0.0, 0.0, 0.0]), normal=np.array([0.0, 0.0, -1.0]), area=0.1, perimeter=1.0),
        ]
        classified = extractor.classify_profiles(profiles, mesh)
        assert classified[0].role == BoundaryRole.INLET
        assert classified[0].confidence == 0.6


class TestTopologySectionExtractor:
    def test_extract_at_profiles(self):
        mesh = _make_cylinder_mesh(radius=1.0, height=5.0)
        profiles = [
            OpenProfile(
                id=0,
                centroid=np.array([0.0, 0.0, 1.0]),
                normal=np.array([0.0, 0.0, -1.0]),
                area=0.0,
                perimeter=0.0,
            ),
            OpenProfile(
                id=1,
                centroid=np.array([0.0, 0.0, 4.0]),
                normal=np.array([0.0, 0.0, 1.0]),
                area=0.0,
                perimeter=0.0,
            ),
        ]
        extractor = TopologySectionExtractor()
        updated = extractor.extract_at_profiles(mesh, profiles)
        assert updated[0].area > 0.0
        assert updated[1].area > 0.0
        assert updated[0].metadata.get("classification_method", "").endswith("_section")

    def test_extract_along_axis(self):
        mesh = _make_cylinder_mesh(radius=1.0, height=5.0)
        extractor = TopologySectionExtractor()
        axis = np.array([0.0, 0.0, 1.0])
        origin = np.array([0.0, 0.0, 0.0])
        sections = extractor.extract_along_axis(mesh, axis, origin, n_steps=5)
        assert len(sections) > 0


class TestCenterlineVsGeometricClassification:
    def test_consistency(self):
        mesh = _make_cylinder_mesh()
        pv_mesh = pv.wrap(mesh)
        analyzer = SurfaceTopologyAnalyzer(pv_mesh)
        surface = analyzer.extract_surface()
        profiles = analyzer.find_open_profiles(surface)
        assert len(profiles) >= 2

        geometric = OpenProfileClassifier().classify(list(profiles))
        centerline = TopologyCenterlineExtractor().classify_profiles(list(profiles), mesh)

        geom_roles = {p.id: p.role for p in geometric}
        cen_roles = {p.id: p.role for p in centerline}
        assert geom_roles[0] == cen_roles.get(0, geom_roles[0]) or geom_roles[1] == cen_roles.get(1, geom_roles[1])
