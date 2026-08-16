from pathlib import Path
from typing import Optional, List

from foampilot.urban.geometry.cfd_geometry import CFDGeometry
from foampilot.urban.mesh.sizing import MeshConfig


class SurfaceQuarterBuilder:
    """Build urban CFD geometry as STL surfaces for snappyHexMesh."""

    def __init__(self, case_path: Path, geometry: CFDGeometry):
        self.case_path = case_path
        self.geometry = geometry
        self._built = False
        self._stl_path = None

    def build(self) -> None:
        if self._built:
            return

        import gmsh

        gmsh.initialize()
        gmsh.model.add("urban_surface")
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Geometry.Tolerance", 1e-6)

        xmin, ymin, zmin, xmax, ymax, zmax = self.geometry.domain_box

        building_surfaces = []
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
            base_surface = gmsh.model.occ.addPlaneSurface([curve_loop])

            height = b.roof_z_local - b.ground_z_local
            if height <= 0:
                continue

            roof_type = getattr(b, 'roof_type', 'flat')
            if roof_type == 'flat' or len(coords) != 4:
                result = gmsh.model.occ.extrude([(2, base_surface)], 0, 0, height)
                for dim, tag in result:
                    if dim == 2:
                        building_surfaces.append(tag)
            else:
                walls, roofs = self._create_lod2_surfaces(coords, b.ground_z_local, height, roof_type)
                building_surfaces.extend(walls)
                building_surfaces.extend(roofs)

        gmsh.model.occ.synchronize()

        if building_surfaces:
            gmsh.model.addPhysicalGroup(2, building_surfaces, tag=1)
            gmsh.model.setPhysicalName(2, 1, "buildings")

        tri_surface_dir = self.case_path / "constant" / "triSurface"
        tri_surface_dir.mkdir(parents=True, exist_ok=True)
        self._stl_path = tri_surface_dir / "buildings.stl"

        gmsh.model.mesh.generate(2)
        gmsh.write(str(self._stl_path))

        gmsh.finalize()
        self._built = True

    def _create_lod2_surfaces(self, coords, ground_z, height, roof_type):
        walls = []
        roofs = []

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
        walls.append(base)

        wall_height = height * 0.7

        if roof_type == "gable":
            p_ridge0 = gmsh.model.occ.addPoint(coords[0][0], coords[0][1], ground_z + wall_height)
            p_ridge1 = gmsh.model.occ.addPoint(coords[1][0], coords[1][1], ground_z + wall_height)

            for i in range(4):
                p_base = [p0, p1, p2, p3][i]
                p_next = [p1, p2, p3, p0][i]
                p_top = [p_ridge0, p_ridge1, p_ridge1, p_ridge0][i]
                tri_loop = gmsh.model.occ.addCurveLoop([
                    gmsh.model.occ.addLine(p_base, p_next),
                    gmsh.model.occ.addLine(p_next, p_top),
                    gmsh.model.occ.addLine(p_top, p_base),
                ])
                walls.append(gmsh.model.occ.addPlaneSurface([tri_loop]))

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
                roofs.append(gmsh.model.occ.addPlaneSurface([roof_loop]))

        elif roof_type == "pyramid":
            center_x = sum(c[0] for c in coords) / 4.0
            center_y = sum(c[1] for c in coords) / 4.0
            p_apex = gmsh.model.occ.addPoint(center_x, center_y, ground_z + height)

            for i in range(4):
                p_bottom_left = [p0, p1, p2, p3][i]
                p_bottom_right = [p1, p2, p3, p0][i]
                tri_loop = gmsh.model.occ.addCurveLoop([
                    gmsh.model.occ.addLine(p_bottom_left, p_bottom_right),
                    gmsh.model.occ.addLine(p_bottom_right, p_apex),
                    gmsh.model.occ.addLine(p_apex, p_bottom_left),
                ])
                walls.append(gmsh.model.occ.addPlaneSurface([tri_loop]))

        elif roof_type == "hip":
            ridge_x = (coords[0][0] + coords[1][0]) / 2.0
            ridge_y = (coords[0][1] + coords[1][1]) / 2.0
            p_ridge0 = gmsh.model.occ.addPoint(ridge_x, ridge_y, ground_z + wall_height)

            for i in range(4):
                p_bottom_left = [p0, p1, p2, p3][i]
                p_bottom_right = [p1, p2, p3, p0][i]
                p_top = [p_ridge0, p_ridge0, p_ridge0, p_ridge0][i]
                tri_loop = gmsh.model.occ.addCurveLoop([
                    gmsh.model.occ.addLine(p_bottom_left, p_bottom_right),
                    gmsh.model.occ.addLine(p_bottom_right, p_top),
                    gmsh.model.occ.addLine(p_top, p_bottom_left),
                ])
                walls.append(gmsh.model.occ.addPlaneSurface([tri_loop]))

        return walls, roofs

    def assign_patches(self) -> None:
        pass

    def build_mesh(self, config: Optional[MeshConfig] = None) -> None:
        if not self._built:
            raise RuntimeError("build() must be called before build_mesh()")

        from foampilot.mesh.snappymesh import SnappyMesher

        if self._stl_path is None or not self._stl_path.exists():
            raise RuntimeError("STL surface not found. Call build() first.")

        mesher = SnappyMesher(
            parent=self,
            stl_file=self._stl_path,
            castellatedMesh=True,
            snap=True,
            addLayers=config.boundary_layers is not None if config else False,
        )

        if config and config.boundary_layers:
            bl = config.boundary_layers
            mesher.addLayersControls["expansionRatio"] = bl.growth_rate
            mesher.addLayersControls["finalLayerThickness"] = bl.first_layer_height.get_in("m")
            mesher.addLayersControls["nSurfaceLayers"] = bl.num_layers
            for patch_name in bl.patches:
                mesher.add_layer(patch_name, bl.num_layers)

        xmin, ymin, zmin, xmax, ymax, zmax = self.geometry.domain_box
        mesher.locationInMesh = (
            (xmin + xmax) / 2.0,
            (ymin + ymax) / 2.0,
            (zmin + zmax) / 2.0,
        )

        mesher.write_block_mesh_dict(padding=0.1)
        mesher.write_snappyHexMeshDict()

        if config:
            self._apply_sizing_to_snappy(mesher, config)

        mesher.write()

    def _apply_sizing_to_snappy(self, mesher, config: MeshConfig):
        stem = self._stl_path.stem if self._stl_path else "buildings"
        mesher.castellatedMeshControls["refinementSurfaces"][stem] = {
            "level": (config.building_size.get_in("m"), config.max_size.get_in("m"))
        }

        if config.wake_refinement is not None:
            wr = config.wake_refinement
            for i, b in enumerate(self.geometry.buildings):
                region_name = f"wake_{b.id}"
                mesher.castellatedMeshControls["refinementRegions"][region_name] = {
                    "mode": "inside",
                    "levels": (wr.target_size.get_in("m"), config.max_size.get_in("m")),
                }

        if config.refinement_regions:
            for i, region in enumerate(config.refinement_regions):
                region_name = f"refinement_{i}"
                mesher.castellatedMeshControls["refinementRegions"][region_name] = {
                    "mode": "inside",
                    "levels": (region.size.get_in("m"), config.max_size.get_in("m")),
                }

    def run(self) -> None:
        from foampilot.mesh.snappymesh import SnappyMesher

        if not self._built:
            raise RuntimeError("build() must be called before run()")

        mesher = SnappyMesher(
            parent=self,
            stl_file=self._stl_path,
            castellatedMesh=True,
            snap=True,
            addLayers=True,
        )
        mesher.run()

    def export_openfoam(self) -> Path:
        return self.case_path / "constant" / "polyMesh"
