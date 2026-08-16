from pathlib import Path
from typing import Tuple, List, Optional
import gmsh

from foampilot.urban.geometry.cfd_geometry import CFDGeometry
from foampilot.urban.mesh.sizing import MeshConfig
from foampilot.urban.patches.patch_assigner import PatchAssigner
from foampilot.urban.validation.geometry_checks import GeometryValidator
from shapely.geometry import Point


class GmshQuarterBuilder:
    def __init__(self, case_path: Path, geometry: CFDGeometry):
        self.case_path = case_path
        self.geometry = geometry
        self._patch_assigner = PatchAssigner()
        self._built = False
        self._patches_assigned = False
        self._meshed = False

    def build(self) -> None:
        if self._built:
            return

        gmsh.initialize()
        gmsh.model.add("urban_cfd")
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Geometry.Tolerance", 1e-6)

        xmin, ymin, zmin, xmax, ymax, zmax = self.geometry.domain_box
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin

        fluid_tag = gmsh.model.occ.addBox(xmin, ymin, zmin, dx, dy, dz)

        building_tags = []
        for b in self.geometry.buildings:
            coords = list(b.footprint_local.exterior.coords)
            if len(coords) > 1 and coords[0] == coords[-1]:
                coords = coords[:-1]
            points = []
            for coord in coords:
                x = float(coord[0])
                y = float(coord[1])
                z = float(b.ground_z_local)
                points.append(gmsh.model.occ.addPoint(x, y, z))

            lines = []
            for i in range(len(points)):
                lines.append(gmsh.model.occ.addLine(points[i], points[(i + 1) % len(points)]))

            curve_loop = gmsh.model.occ.addCurveLoop(lines)
            surface = gmsh.model.occ.addPlaneSurface([curve_loop])

            height = b.roof_z_local - b.ground_z_local
            if height <= 0:
                continue

            roof_type = getattr(b, 'roof_type', 'flat')
            if roof_type == 'flat' or len(coords) != 4:
                result = gmsh.model.occ.extrude([(2, surface)], 0, 0, height)
                volume_tag = None
                for dim, tag in result:
                    if dim == 3:
                        volume_tag = tag
                        break
            else:
                volume_tag = self._create_lod2_building(coords, b.ground_z_local, height, roof_type)

            if volume_tag is not None:
                building_tags.append(volume_tag)

        gmsh.model.occ.synchronize()

        terrain_surface = None
        if self.geometry.terrain is not None and self.geometry.terrain.points:
            terrain_z = self.geometry.terrain.get_elevation(
                (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
            )
            terrain_points = [
                gmsh.model.occ.addPoint(xmin, ymin, terrain_z),
                gmsh.model.occ.addPoint(xmax, ymin, terrain_z),
                gmsh.model.occ.addPoint(xmax, ymax, terrain_z),
                gmsh.model.occ.addPoint(xmin, ymax, terrain_z),
            ]
            terrain_lines = [
                gmsh.model.occ.addLine(terrain_points[0], terrain_points[1]),
                gmsh.model.occ.addLine(terrain_points[1], terrain_points[2]),
                gmsh.model.occ.addLine(terrain_points[2], terrain_points[3]),
                gmsh.model.occ.addLine(terrain_points[3], terrain_points[0]),
            ]
            terrain_loop = gmsh.model.occ.addCurveLoop(terrain_lines)
            terrain_surface = gmsh.model.occ.addPlaneSurface([terrain_loop])
            gmsh.model.occ.synchronize()

        if building_tags:
            try:
                gmsh.model.occ.fragment(
                    [(3, fluid_tag)] + [(3, t) for t in building_tags],
                    [],
                )
                gmsh.model.occ.synchronize()
            except Exception as exc:
                print(f"WARNING: Gmsh fragment failed ({exc}), falling back to cut()")
                try:
                    current = [(3, fluid_tag)]
                    for t in building_tags:
                        current = gmsh.model.occ.cut(
                            current, [(3, t)], removeObject=True, removeTool=True
                        )
                    gmsh.model.occ.synchronize()
                except Exception as exc2:
                    print(f"WARNING: Gmsh cut also failed ({exc2}), keeping original geometry")

        if building_tags:
            all_volumes = gmsh.model.getEntities(dim=3)
            volumes_to_remove = []
            building_bboxes = [
                (b.footprint_local.bounds[0], b.footprint_local.bounds[1],
                 b.footprint_local.bounds[2], b.footprint_local.bounds[3],
                 b.ground_z_local, b.roof_z_local)
                for b in self.geometry.buildings
            ]
            for dim, tag in all_volumes:
                if tag == fluid_tag:
                    continue
                try:
                    com = gmsh.model.occ.getCenterOfMass(3, tag)
                except Exception:
                    continue
                if com is None:
                    continue
                cx, cy, cz = com
                
                # Fast bbox check first
                inside_any = False
                for xmin, ymin, xmax, ymax, zmin, zmax in building_bboxes:
                    if not (xmin <= cx <= xmax and ymin <= cy <= ymax and zmin <= cz <= zmax):
                        continue
                    inside_any = True
                    break
                
                if not inside_any:
                    continue
                
                # Precise check only for volumes inside building bboxes
                for b in self.geometry.buildings:
                    if b.ground_z_local - 1e-4 <= cz <= b.roof_z_local + 1e-4:
                        if Point(cx, cy).within(b.footprint_local):
                            volumes_to_remove.append(tag)
                            break

            if volumes_to_remove:
                gmsh.model.occ.remove([(3, t) for t in volumes_to_remove])
                gmsh.model.occ.synchronize()

        self._fluid_tag = fluid_tag
        self._building_tags = building_tags
        self._built = True

    def assign_patches(self) -> None:
        if not self._built or self._patches_assigned:
            return

        self._patch_assigner.assign(self)
        self._patches_assigned = True

    def _create_lod2_building(self, coords, ground_z, height, roof_type):
        p0 = gmsh.model.occ.addPoint(coords[0][0], coords[0][1], ground_z)
        p1 = gmsh.model.occ.addPoint(coords[1][0], coords[1][1], ground_z)
        p2 = gmsh.model.occ.addPoint(coords[2][0], coords[2][1], ground_z)
        p3 = gmsh.model.occ.addPoint(coords[3][0], coords[3][1], ground_z)

        l0 = gmsh.model.occ.addLine(p0, p1)
        l1 = gmsh.model.occ.addLine(p1, p2)
        l2 = gmsh.model.occ.addLine(p2, p3)
        l3 = gmsh.model.occ.addLine(p3, p0)

        loop = gmsh.model.occ.addCurveLoop([l0, l1, l2, l3])
        base = gmsh.model.occ.addPlaneSurface([loop])

        wall_height = height * 0.7
        roof_height = height * 0.3

        if roof_type == "gable":
            ridge_y = (coords[0][1] + coords[1][1]) / 2.0
            ridge_x = (coords[0][0] + coords[1][0]) / 2.0
            p_ridge0 = gmsh.model.occ.addPoint(coords[0][0], coords[0][1], ground_z + wall_height)
            p_ridge1 = gmsh.model.occ.addPoint(coords[1][0], coords[1][1], ground_z + wall_height)
            ridge_line = gmsh.model.occ.addLine(p_ridge0, p_ridge1)

            wall_surfaces = []
            for i in range(4):
                p_base = [p0, p1, p2, p3][i]
                p_next = [p1, p2, p3, p0][i]
                p_top = [p_ridge0, p_ridge1, p_ridge1, p_ridge0][i]
                tri_loop = gmsh.model.occ.addCurveLoop([
                    gmsh.model.occ.addLine(p_base, p_next),
                    gmsh.model.occ.addLine(p_next, p_top),
                    gmsh.model.occ.addLine(p_top, p_base),
                ])
                wall_surfaces.append(gmsh.model.occ.addPlaneSurface([tri_loop]))

            roof_surfaces = []
            for i in range(2):
                p_top_left = [p_ridge0, p_ridge1][i]
                p_top_right = [p_ridge1, p_ridge0][i]
                p_bottom_left = [p0, p1][i]
                p_bottom_right = [p3, p2][i]
                roof_loop = gmsh.model.occ.addCurveLoop([
                    gmsh.model.occ.addLine(p_top_left, p_top_right),
                    gmsh.model.occ.addLine(p_top_right, p_bottom_right),
                    gmsh.model.occ.addLine(p_bottom_right, p_bottom_left),
                    gmsh.model.occ.addLine(p_bottom_left, p_top_left),
                ])
                roof_surfaces.append(gmsh.model.occ.addPlaneSurface([roof_loop]))

            all_surfaces = [base] + wall_surfaces + roof_surfaces
            shell = gmsh.model.occ.addShell(all_surfaces)
            volume = gmsh.model.occ.addVolume([shell])
            return volume

        center_x = sum(c[0] for c in coords) / 4.0
        center_y = sum(c[1] for c in coords) / 4.0
        p_apex = gmsh.model.occ.addPoint(center_x, center_y, ground_z + height)

        if roof_type == "pyramid":
            wall_surfaces = []
            for i in range(4):
                p_bottom_left = [p0, p1, p2, p3][i]
                p_bottom_right = [p1, p2, p3, p0][i]
                tri_loop = gmsh.model.occ.addCurveLoop([
                    gmsh.model.occ.addLine(p_bottom_left, p_bottom_right),
                    gmsh.model.occ.addLine(p_bottom_right, p_apex),
                    gmsh.model.occ.addLine(p_apex, p_bottom_left),
                ])
                wall_surfaces.append(gmsh.model.occ.addPlaneSurface([tri_loop]))

            all_surfaces = [base] + wall_surfaces
            shell = gmsh.model.occ.addShell(all_surfaces)
            volume = gmsh.model.occ.addVolume([shell])
            return volume

        if roof_type == "hip":
            ridge_x = (coords[0][0] + coords[1][0]) / 2.0
            ridge_y = (coords[0][1] + coords[1][1]) / 2.0
            p_ridge0 = gmsh.model.occ.addPoint(ridge_x, ridge_y, ground_z + wall_height)

            wall_surfaces = []
            for i in range(4):
                p_bottom_left = [p0, p1, p2, p3][i]
                p_bottom_right = [p1, p2, p3, p0][i]
                p_top = [p_ridge0, p_ridge0, p_ridge0, p_ridge0][i]
                tri_loop = gmsh.model.occ.addCurveLoop([
                    gmsh.model.occ.addLine(p_bottom_left, p_bottom_right),
                    gmsh.model.occ.addLine(p_bottom_right, p_top),
                    gmsh.model.occ.addLine(p_top, p_bottom_left),
                ])
                wall_surfaces.append(gmsh.model.occ.addPlaneSurface([tri_loop]))

            all_surfaces = [base] + wall_surfaces
            shell = gmsh.model.occ.addShell(all_surfaces)
            volume = gmsh.model.occ.addVolume([shell])
            return volume

        return None

    def build_mesh(self, config: MeshConfig) -> None:
        if not self._built:
            raise RuntimeError("build() must be called before build_mesh()")

        lc_min = config.min_size.get_in("m")
        lc_max = config.max_size.get_in("m")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_max)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2)
        gmsh.option.setNumber("Mesh.Algorithm3D", config.algorithm_3d)
        gmsh.option.setNumber("Mesh.Algorithm", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

        if config.boundary_layers:
            bl = config.boundary_layers
            first_layer = bl.first_layer_height.get_in("m")
            growth = bl.growth_rate
            n_layers = bl.num_layers
            try:
                gmsh.option.setNumber("Mesh.BoundaryLayers", 1)
                gmsh.option.setNumber("Mesh.BoundaryLayers.FirstLayerHeight", first_layer)
                gmsh.option.setNumber("Mesh.BoundaryLayers.GrowthRate", growth)
                gmsh.option.setNumber("Mesh.BoundaryLayers.NbLayers", n_layers)
            except Exception:
                pass

        all_volumes = gmsh.model.getEntities(dim=3)
        building_faces = []
        ground_faces = []
        for dim, tag in all_volumes:
            if tag == self._fluid_tag:
                continue
            try:
                com = gmsh.model.occ.getCenterOfMass(3, tag)
            except Exception:
                continue
            if com is None:
                continue
            cx, cy, cz = com
            for b in self.geometry.buildings:
                if b.ground_z_local - 1e-4 <= cz <= b.roof_z_local + 1e-4:
                    if Point(cx, cy).within(b.footprint_local):
                        try:
                            boundary = gmsh.model.getBoundary([(3, tag)], combined=False, oriented=True)
                            building_faces.extend(boundary)
                        except Exception:
                            pass
                        break

        xmin, ymin, zmin, xmax, ymax, zmax = self.geometry.domain_box
        for dim, tag in gmsh.model.getEntities(dim=2):
            try:
                com = gmsh.model.occ.getCenterOfMass(2, tag)
            except Exception:
                continue
            if com is None:
                continue
            cx, cy, cz = com
            if abs(cz - zmin) < 1e-4:
                ground_faces.append((2, tag))

        if building_faces:
            try:
                gmsh.model.mesh.setSize(
                    gmsh.model.getBoundary(building_faces, recursive=True),
                    config.building_size.get_in("m"),
                )
            except Exception:
                pass

        if ground_faces:
            try:
                gmsh.model.mesh.setSize(
                    ground_faces,
                    config.ground_size.get_in("m"),
                )
            except Exception:
                pass

        background_faces = []
        building_and_ground = set(building_faces + ground_faces)
        for dim, tag in gmsh.model.getEntities(dim=2):
            if (dim, tag) not in building_and_ground:
                background_faces.append((dim, tag))

        if background_faces:
            try:
                gmsh.model.mesh.setSize(
                    background_faces,
                    config.global_size.get_in("m"),
                )
            except Exception:
                pass

        if config.wake_refinement is not None:
            wr = config.wake_refinement
            wake_lc = wr.target_size.get_in("m")
            wake_length = wr.length
            wake_width = wr.width
            wake_height = wr.height

            for b in self.geometry.buildings:
                bxmin, bymin, bxmax, bymax = b.footprint_local.bounds
                x_center = (bxmin + bxmax) / 2.0
                y_center = (bymin + bymax) / 2.0
                z_base = b.ground_z_local
                h = b.height

                wake_box = [
                    x_center,
                    bymax,
                    z_base,
                    x_center + wake_length * h,
                    bymax + wake_width * h,
                    z_base + wake_height * h,
                ]

                try:
                    box_tag = gmsh.model.occ.addBox(
                        wake_box[0], wake_box[1], wake_box[2],
                        wake_box[3] - wake_box[0],
                        wake_box[4] - wake_box[1],
                        wake_box[5] - wake_box[2],
                    )
                    gmsh.model.occ.synchronize()
                    gmsh.model.mesh.setSize(
                        gmsh.model.getBoundary([(3, box_tag)], recursive=True),
                        wake_lc,
                    )
                except Exception:
                    pass

        if config.refinement_regions:
            for region in config.refinement_regions:
                cx, cy, cz = region.center
                try:
                    if region.radius is not None:
                        r = region.radius.get_in("m")
                        ball = gmsh.model.occ.addSphere(cx, cy, cz, r)
                        gmsh.model.occ.synchronize()
                        gmsh.model.mesh.setSize(
                            gmsh.model.getBoundary([(3, ball)], recursive=True),
                            region.size.get_in("m"),
                        )
                    else:
                        sx = region.size.get_in("m")
                        sy = region.size.get_in("m")
                        sz = region.size.get_in("m")
                        box = gmsh.model.occ.addBox(cx - sx/2, cy - sy/2, cz - sz/2, sx, sy, sz)
                        gmsh.model.occ.synchronize()
                        gmsh.model.mesh.setSize(
                            gmsh.model.getBoundary([(3, box)], recursive=True),
                            region.size.get_in("m"),
                        )
                except Exception:
                    pass

        gmsh.model.mesh.generate(3)

        self._meshed = True

    def export_openfoam(self) -> Path:
        if not self._built:
            raise RuntimeError("build() must be called before export_openfoam()")

        from foampilot.mesh.direct_openfoam_exporter import DirectOpenFOAMExporter
        exporter = DirectOpenFOAMExporter(self.case_path)
        exporter.export_single_region()
        gmsh.finalize()
        return self.case_path / "constant" / "polyMesh"
