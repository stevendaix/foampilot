#!/usr/bin/env python3
"""
Vector Gmsh builder: build OpenFOAM-ready single-region mesh from UrbanModel + CFDTerrain.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "foampilot" / "src"))

import gmsh
import shapely.geometry
import shapely.ops
from foampilot.urban.model.urban_model import Building, UrbanModel
from foampilot.urban.model.terrain import CFDTerrain
from foampilot.mesh.direct_openfoam_exporter import DirectOpenFOAMExporter
from shapely.geometry import Point


class VectorGmshBuilder:
    """Build single-region OpenFOAM mesh directly from vector data."""

    def __init__(self, urban: UrbanModel, terrain: CFDTerrain, mesh_size: float = 5.0, mesh_constraint: str = "none", fill_gaps: bool = False):
        self.urban = urban
        self.terrain = terrain
        self.mesh_size = mesh_size
        self.mesh_constraint = mesh_constraint
        self.fill_gaps = fill_gaps
        self.buildings = []
        self.building_tags = []
        self.fluid_tag = None
        self._built = False
        self._patches_assigned = False
        self._meshed = False
        self._building_surface_tags = []
        self._margin = None
        self._bottom_offset = 5.0
        self._init_gmsh()

    def _init_gmsh(self):
        gmsh.initialize()
        gmsh.model.add("voxcity_vector")
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Geometry.Tolerance", 1e-6)
        gmsh.option.setNumber("Mesh.Algorithm", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.05)

    def _extrude_polygon(self, footprint, base_z, height, eps_ground=1.0):
        try:
            base_polygon = shapely.geometry.Polygon([(x, y) for x, y in footprint.exterior.coords])
            if not base_polygon.is_valid:
                base_polygon = base_polygon.buffer(0)
            if base_polygon.is_empty or base_polygon.area < 1e-6:
                return None
            ll = base_polygon.bounds
            dx = ll[2] - ll[0]
            dy = ll[3] - ll[1]
            dz = max(1.0, height - base_z)
            cx = (ll[0] + ll[2]) / 2.0
            cy = (ll[1] + ll[3]) / 2.0
            cz = base_z + dz / 2.0
            base_tag = gmsh.model.occ.addBox(cx - dx / 2.0, cy - dy / 2.0, base_z, dx, dy, dz)
            gmsh.model.occ.synchronize()
            terrain_polygon = shapely.geometry.Polygon([
                (self.terrain.xmin, self.terrain.ymin),
                (self.terrain.xmax, self.terrain.ymin),
                (self.terrain.xmax, self.terrain.ymax),
                (self.terrain.xmin, self.terrain.ymax),
            ])
            try:
                base_polygon_proj = shapely.ops.transform(lambda x, y: (x, y), base_polygon)
            except Exception:
                base_polygon_proj = base_polygon
            if terrain_polygon.contains(base_polygon_proj) or terrain_polygon.intersects(base_polygon_proj):
                return base_tag
            gmsh.model.occ.remove([(3, base_tag)])
            gmsh.model.occ.synchronize()
            return None
        except Exception:
            return None

    def _create_building_volume(self, building):
        base_z = building.ground_z if building.ground_z is not None else 0.0
        roof_z = building.roof_z if building.roof_z is not None else (base_z + 10.0)
        height = roof_z - base_z
        if height <= 0:
            height = 10.0
        return self._extrude_polygon(building.footprint, base_z, height, eps_ground=1.0)

    def _identify_building_volumes(self, all_vols):
        building_footprints = []
        for b in self.urban.buildings():
            fp = b.footprint
            if fp is None or fp.is_empty:
                continue
            building_footprints.append((fp, b.ground_z, b.roof_z))
        building_volumes = []
        for dim, tag in all_vols:
            if tag == self.fluid_tag:
                continue
            try:
                bbox = gmsh.model.occ.getBoundingBox(3, tag)
                cx = (bbox[0] + bbox[3]) / 2.0
                cy = (bbox[1] + bbox[4]) / 2.0
                cz = (bbox[2] + bbox[5]) / 2.0
                for fp, gz, rz in building_footprints:
                    if gz is None:
                        gz = 0.0
                    if rz is None:
                        rz = gz + 10.0
                    if not (bbox[0] - 1e-4 <= cx <= bbox[3] + 1e-4 and bbox[1] - 1e-4 <= cy <= bbox[4] + 1e-4):
                        continue
                    if not (gz - 1e-4 <= cz <= rz + 1e-4):
                        continue
                    if Point(cx, cy).within(fp):
                        building_volumes.append(tag)
                        break
            except Exception:
                continue
        return building_volumes

    def build(self, margin: float = None, bottom_offset: float = 5.0):
        """Build fluid domain and building volumes.
        Domain margins follow building_aero rules when margin is None:
          - upstream  = 4 * Hmax
          - downstream = 7.5 * Hmax
          - lateral   = 2 * D (total building width)
          - top       = 1.25 * Hmax
        Otherwise, margin is used uniformly in X/Y/Z.
        """
        if self._built:
            return
        if not self.urban.buildings():
            raise RuntimeError("No buildings to build")
        self._margin = margin
        self._bottom_offset = bottom_offset
        self._preprocess_geometry()
        bbox = self.urban.bbox()
        xmin, ymin, zmin, xmax, ymax, zmax = bbox
        if margin is None:
            heights = [b.roof_z - b.ground_z for b in self.urban.buildings()]
            Hmax = max(heights) if heights else 10.0
            D = xmax - xmin
            upstream = 4.0 * Hmax
            downstream = 7.5 * Hmax
            lateral = 2.0 * max(D, 1.0)
            top = 1.25 * Hmax
            domain_xmin = xmin - upstream
            domain_ymin = ymin - lateral
            domain_zmin = zmin - bottom_offset
            domain_xmax = xmax + downstream
            domain_ymax = ymax + lateral
            domain_zmax = zmax + top
        else:
            domain_xmin = xmin - margin
            domain_ymin = ymin - margin
            domain_zmin = zmin - bottom_offset
            domain_xmax = xmax + margin
            domain_ymax = ymax + margin
            domain_zmax = zmax + margin
        dx = domain_xmax - domain_xmin
        dy = domain_ymax - domain_ymin
        dz = domain_zmax - domain_zmin
        self.fluid_tag = gmsh.model.occ.addBox(
            domain_xmin, domain_ymin, domain_zmin, dx, dy, dz
        )
        self._domain_bbox = (domain_xmin, domain_ymin, domain_zmin, domain_xmax, domain_ymax, domain_zmax)
        original_building_tags = []
        for building in self.urban.buildings():
            vol_tag = self._create_building_volume(building)
            if vol_tag is not None:
                self.building_tags.append(vol_tag)
                original_building_tags.append(vol_tag)
                self.buildings.append(building.id)
        gmsh.model.occ.synchronize()

        if self.building_tags:
            print(f"  Cutting {len(self.building_tags)} buildings from fluid domain...")
            fluid_volume = [(3, self.fluid_tag)]
            for btag in list(self.building_tags):
                try:
                    result, _ = gmsh.model.occ.cut(
                        fluid_volume, [(3, btag)], removeObject=True, removeTool=True
                    )
                    gmsh.model.occ.synchronize()
                    if result and len(result) > 0:
                        fluid_volume = list(result)
                    else:
                        print(f"  WARNING: cut of building {btag} returned empty result")
                        break
                except Exception as exc:
                    print(f"  WARNING: cut failed: {exc}")
                    print(f"    Skipping building {btag}")
                    continue
            print("  cut completed")
            
            all_vols = gmsh.model.getEntities(dim=3)
            if all_vols:
                masses = []
                for dim, tag in all_vols:
                    try:
                        mass = abs(gmsh.model.occ.getMass(3, tag))
                        masses.append((tag, mass))
                    except Exception:
                        continue
                if masses:
                    total_mass = sum(m for _, m in masses)
                    threshold = max(1e-6 * total_mass, 1e-3)
                    fluid_candidates = [tag for tag, mass in masses if mass >= threshold]
                    if fluid_candidates:
                        self.fluid_tag = fluid_candidates[0]
                        print(f"  Fluid tag after cuts: {self.fluid_tag}, mass={total_mass:.1f}")
                        
                        building_fragments = [tag for tag, _ in masses if tag != self.fluid_tag]
                        if building_fragments:
                            print(f"  Removing {len(building_fragments)} building fragments...")
                            try:
                                gmsh.model.occ.remove([(3, t) for t in building_fragments])
                                gmsh.model.occ.synchronize()
                            except Exception as exc:
                                print(f"  Warning: could not remove building fragments: {exc}")
            self.building_tags = []
        
        if self.fluid_tag in self.building_tags:
            self.fluid_tag = None
        self._built = True
        print(f"  Built {len(self.buildings)} buildings, fluid tag={self.fluid_tag}")

    def _setup_proximity_field(self):
        """Set up a proximity-based mesh size field using Gmsh Distance + Threshold."""
        if self.mesh_constraint != "proximity":
            return
        surface_tags = getattr(self, "_building_surface_tags", [])
        if not surface_tags:
            print("  No building surfaces available for proximity field")
            return
        print(f"  Setting up proximity mesh constraint on {len(surface_tags)} surfaces...")
        try:
            field_id = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(field_id, "FacesList", surface_tags)
            threshold_id = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold_id, "InField", field_id)
            gmsh.model.mesh.field.setNumber(threshold_id, "LcMin", self.mesh_size * 0.5)
            gmsh.model.mesh.field.setNumber(threshold_id, "LcMax", self.mesh_size * 3.0)
            gmsh.model.mesh.field.setNumber(threshold_id, "DistMin", self.mesh_size * 0.5)
            gmsh.model.mesh.field.setNumber(threshold_id, "DistMax", self.mesh_size * 6.0)
            gmsh.model.mesh.field.setNumber(threshold_id, "StopAtDistMax", 1)
            gmsh.model.mesh.field.setAsBackgroundMesh(threshold_id)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            print(f"  Proximity field set: LcMin={self.mesh_size * 0.5}, LcMax={self.mesh_size * 3.0}")
        except Exception as exc:
            print(f"  WARNING: Could not set proximity field ({exc})")

    def analyze_geometry(self):
        """Analyze geometry for potential meshing issues."""
        issues = []
        buildings = list(self.urban.buildings())
        if not buildings:
            return ["No buildings to analyze"]
        building_bboxes = []
        for b in buildings:
            building_bboxes.append((
                b.footprint.bounds[0], b.footprint.bounds[1],
                b.footprint.bounds[2], b.footprint.bounds[3],
                b.ground_z, b.roof_z
            ))
        for i, (xmin, ymin, xmax, ymax, zmin, zmax) in enumerate(building_bboxes):
            area = (xmax - xmin) * (ymax - ymin)
            if area < 1.0:
                issues.append(f"Building {i}: very small footprint area {area:.2f} m²")
            aspect_xy = max(xmax - xmin, ymax - ymin) / max(1e-6, min(xmax - xmin, ymax - ymin))
            if aspect_xy > 10:
                issues.append(f"Building {i}: high aspect ratio footprint {aspect_xy:.1f}")
            height = zmax - zmin
            if height < 1.0:
                issues.append(f"Building {i}: very low height {height:.2f} m")
            base_area = max(1e-6, (xmax - xmin) * (ymax - ymin))
            if height / base_area > 5:
                issues.append(f"Building {i}: tall/slender building")
        for i in range(len(building_bboxes)):
            for j in range(i + 1, len(building_bboxes)):
                xmin1, ymin1, xmax1, ymax1, zmin1, zmax1 = building_bboxes[i]
                xmin2, ymin2, xmax2, ymax2, zmin2, zmax2 = building_bboxes[j]
                dx = max(0, max(xmin1, xmin2) - min(xmax1, xmax2))
                dy = max(0, max(ymin1, ymin2) - min(ymax1, ymax2))
                dz = max(0, max(zmin1, zmin2) - min(zmax1, zmax2))
                dist = (dx**2 + dy**2 + dz**2) ** 0.5
                if dist < self.mesh_size * 0.5:
                    issues.append(f"Buildings {i}-{j}: very close {dist:.2f} m (mesh size={self.mesh_size})")
                if xmax1 > xmin2 and xmax2 > xmin1 and ymax1 > ymin2 and ymax2 > ymin1:
                    if zmax1 > zmin2 and zmax2 > zmin1:
                        issues.append(f"Buildings {i}-{j}: possible overlap in XY+Z")
        return issues if issues else ["No obvious geometry issues detected"]

    def _preprocess_geometry(self):
        """Clean individual building footprints without merging them.
        Each building is filtered and simplified independently:
        - Remove invalid/degenerate buildings (area < 1.0, height < 0.5)
        - Fix invalid geometries with buffer(0)
        - Simplify footprints to reduce vertex count
        """
        buildings = list(self.urban.buildings())
        if not buildings:
            return
        print(f"  Pre-processing {len(buildings)} buildings...")
        min_area = 1.0
        min_height = 0.5
        clean_buildings = []
        for b in buildings:
            area = b.footprint.area
            height = b.roof_z - b.ground_z
            if area < min_area or height < min_height:
                print(f"  Skipping building {b.id}: area={area:.1f}, height={height:.1f}")
                continue
            footprint = b.footprint
            if not footprint.is_valid:
                try:
                    footprint = footprint.buffer(0)
                except Exception:
                    print(f"  WARNING: Could not fix invalid footprint for {b.id}")
                    continue
            if footprint.is_empty:
                continue
            try:
                footprint = footprint.simplify(
                    tolerance=self.mesh_size * 0.5, preserve_topology=True
                )
            except Exception:
                pass
            if footprint.is_empty:
                continue
            cleaned = Building(
                id=b.id,
                footprint=footprint,
                ground_z=b.ground_z,
                roof_z=b.roof_z,
                source=b.source,
                confidence=b.confidence,
                attributes=getattr(b, "attributes", {}),
            )
            clean_buildings.append(cleaned)
        if len(clean_buildings) < len(buildings):
            print(f"  Filtered {len(buildings) - len(clean_buildings)} invalid/degenerate buildings")
        self.urban = UrbanModel()
        for b in clean_buildings:
            self.urban.add_building(b)
        print(f"  Kept {len(clean_buildings)} individual buildings")

    def _create_building_volume(self, building: Building):
        """Create a building volume from the real footprint polygon if possible."""
        footprint = building.footprint
        if footprint is None or footprint.is_empty:
            return None
        if not footprint.is_valid:
            try:
                footprint = footprint.buffer(0)
            except Exception:
                return None
        base_z = float(building.ground_z)
        roof_z = float(building.roof_z)
        height = roof_z - base_z
        if height <= 0:
            return None
        min_area = max(1.0, self.mesh_size * self.mesh_size)
        if footprint.area < min_area:
            return None
        try:
            return self._extrude_polygon(footprint, base_z, height, eps_ground=5.0)
        except Exception as exc:
            print(f"  WARNING: Polygon extrusion failed for {building.id}: {exc}")
            return None

    def _extrude_polygon(self, polygon, base_z: float, height: float, eps_ground: float = 0.0):
        """Extrude a 2D shapely polygon into a 3D Gmsh volume."""
        if polygon.geom_type == "Polygon":
            rings = [polygon.exterior] + list(polygon.interiors)
        elif polygon.geom_type == "MultiPolygon":
            return None
        else:
            return None
        actual_base = base_z - eps_ground - 0.1
        actual_height = height + eps_ground
        all_lines = []
        for ring in rings:
            coords = list(ring.coords)
            if len(coords) < 3:
                return None
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            pts = []
            for x, y in coords[:-1]:
                tag = gmsh.model.occ.addPoint(x, y, actual_base)
                pts.append(tag)
            lines = []
            for i in range(len(pts) - 1):
                tag = gmsh.model.occ.addLine(pts[i], pts[i + 1])
                lines.append(tag)
            if len(pts) > 1:
                tag = gmsh.model.occ.addLine(pts[-1], pts[0])
                lines.append(tag)
            all_lines.append(lines)
        loops = [gmsh.model.occ.addCurveLoop(lines) for lines in all_lines]
        surface = gmsh.model.occ.addPlaneSurface(loops)
        gmsh.model.occ.synchronize()
        result = gmsh.model.occ.extrude([(2, surface)], 0, 0, actual_height)
        volume_tag = next(
            (tag for dim, tag in result if dim == 3), None
        )
        return volume_tag

    def _identify_building_volumes(self, all_vols):
        """Identify ALL building volume fragments by checking COM against building footprints."""
        building_volumes = []
        buildings = list(self.urban.buildings())
        if not buildings:
            return building_volumes
        building_bboxes = []
        building_footprints = []
        for b in buildings:
            building_bboxes.append((
                b.footprint.bounds[0], b.footprint.bounds[1],
                b.footprint.bounds[2], b.footprint.bounds[3],
                b.ground_z, b.roof_z
            ))
            building_footprints.append(b.footprint)
        for dim, tag in all_vols:
            if dim != 3:
                continue
            if tag == self.fluid_tag:
                continue
            try:
                com = gmsh.model.occ.getCenterOfMass(3, tag)
            except Exception:
                continue
            if com is None:
                continue
            cx, cy, cz = com
            for i, (xmin, ymin, xmax, ymax, zmin, zmax) in enumerate(building_bboxes):
                if not (xmin - 1e-4 <= cx <= xmax + 1e-4 and ymin - 1e-4 <= cy <= ymax + 1e-4):
                    continue
                if not (zmin - 1e-4 <= cz <= zmax + 1e-4):
                    continue
                if Point(cx, cy).within(building_footprints[i]):
                    building_volumes.append(tag)
                    break
        return building_volumes

    def _remove_debris(self):
        """Remove tiny leftover volumes after Boolean ops to avoid Gmsh 3D errors."""
        all_volumes = gmsh.model.getEntities(dim=3)
        if not all_volumes:
            return
        bbox = self.urban.bbox()
        xmin, ymin, zmin, xmax, ymax, zmax = bbox
        domain_vol = max(1.0, (xmax - xmin)) * max(1.0, (ymax - ymin)) * max(1.0, (zmax - zmin))
        to_remove = []
        for dim, tag in all_volumes:
            if tag == self.fluid_tag:
                continue
            try:
                bbox_vol = gmsh.model.occ.getBoundingBox(3, tag)
            except Exception:
                continue
            if len(bbox_vol) != 6:
                continue
            vxmin, vymin, vzmin, vxmax, vymax, vzmax = bbox_vol
            vol = max(1e-12, (vxmax - vxmin)) * max(1e-12, (vymax - vymin)) * max(1e-12, (vzmax - vzmin))
            if vol < 1e-6 * domain_vol:
                to_remove.append(tag)
        if to_remove:
            gmsh.model.occ.remove([(3, t) for t in to_remove])
            gmsh.model.occ.synchronize()
        try:
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()
        except Exception:
            pass

    def _merge_building_volumes(self, volume_tags, merge_distance=None):
        """Merge nearby building volumes after extrusion to reduce boolean ops.
        Groups volumes whose centers of mass are within ``merge_distance``
        and fuses each group with ``gmsh.model.occ.fuse``. Returns the new
        list of volume tags.
        """
        if len(volume_tags) <= 1:
            return volume_tags
        if merge_distance is None:
            merge_distance = max(1.0, self.mesh_size * 1.5)
        com_map = {}
        valid_tags = []
        for tag in volume_tags:
            try:
                bbox = gmsh.model.occ.getBoundingBox(3, tag)
                if len(bbox) != 6:
                    continue
                vol = max(1e-12, bbox[3] - bbox[0]) * max(1e-12, bbox[4] - bbox[1]) * max(1e-12, bbox[5] - bbox[2])
                if vol < 1e-6:
                    continue
                com = gmsh.model.occ.getCenterOfMass(3, tag)
                if com is not None:
                    com_map[tag] = com
                    valid_tags.append(tag)
            except Exception:
                continue
        if len(valid_tags) <= 1:
            return valid_tags
        parent = list(range(len(valid_tags)))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
        for i in range(len(valid_tags)):
            for j in range(i + 1, len(valid_tags)):
                cx1, cy1, cz1 = com_map[valid_tags[i]]
                cx2, cy2, cz2 = com_map[valid_tags[j]]
                dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + (cz1 - cz2) ** 2) ** 0.5
                if dist < merge_distance:
                    union(i, j)
        groups = {}
        for i, tag in enumerate(valid_tags):
            root = find(i)
            groups.setdefault(root, []).append(tag)
        merged_tags = []
        for group in groups.values():
            if len(group) == 1:
                merged_tags.append(group[0])
            else:
                try:
                    result, _ = gmsh.model.occ.fuse(
                        [(3, t) for t in group], [], removeObject=True, removeTool=True
                    )
                    gmsh.model.occ.synchronize()
                    new_tags = [tag for dim, tag in result if dim == 3]
                    if new_tags:
                        merged_tags.extend(new_tags)
                    else:
                        merged_tags.extend(group)
                except Exception as exc:
                    print(f"  WARNING: fuse failed for volumes {group}: {exc}")
                    merged_tags.extend(group)
        return merged_tags

    def assign_patches(self):
        """Assign boundary patches by face classification."""
        if not self._built or self._patches_assigned:
            return
        if hasattr(self, '_domain_bbox'):
            xmin, ymin, zmin, xmax, ymax, zmax = self._domain_bbox
        else:
            margin = getattr(self, '_margin', 50.0)
            bottom_offset = getattr(self, '_bottom_offset', 5.0)
            xmin, ymin, zmin, xmax, ymax, zmax = self.urban.bbox()
            xmin -= margin
            ymin -= margin
            zmin -= bottom_offset
            xmax += margin
            ymax += margin
            zmax += margin
        print(f"  Patch classification bounds: x=[{xmin}, {xmax}], y=[{ymin}, {ymax}], z=[{zmin}, {zmax}]")
        all_faces = gmsh.model.getEntities(dim=2)
        patch_to_surfaces = {}
        for _, face in all_faces:
            try:
                com = gmsh.model.occ.getCenterOfMass(2, face)
            except Exception:
                continue
            if com is None:
                continue
            cx, cy, cz = com
            patch = self._classify_patch(cx, cy, cz, xmin, ymin, zmin, xmax, ymax, zmax)
            patch_to_surfaces.setdefault(patch, []).append(face)
        for patch_name, tags in patch_to_surfaces.items():
            if tags:
                gmsh.model.addPhysicalGroup(2, tags, name=patch_name)
        volumes = gmsh.model.getEntities(dim=3)
        if not volumes:
            raise RuntimeError("No 3D volume available for physical group 'fluid'.")
        masses = []
        for dim, tag in volumes:
            try:
                mass = abs(gmsh.model.occ.getMass(dim, tag))
            except Exception:
                mass = 0.0
            masses.append((tag, mass))
        total_mass = sum(m for _, m in masses)
        if total_mass <= 0.0:
            raise RuntimeError("All remaining 3D volumes have zero mass.")
        threshold = 1e-6 * total_mass
        fluid_tags = [tag for tag, mass in masses if mass >= threshold]
        if not fluid_tags:
            fluid_tags = [max(masses, key=lambda x: x[1])[0]]
        if fluid_tags:
            gmsh.model.addPhysicalGroup(3, fluid_tags, name="fluid")
        building_tags = [tag for tag, mass in masses if mass < threshold and mass > 0]
        if building_tags:
            gmsh.model.addPhysicalGroup(3, building_tags, name="buildings")
        self._patches_assigned = True
        print(f"  Patches assigned: {list(patch_to_surfaces.keys())}")
        print(f"  Fluid volumes: {len(fluid_tags)}, Building volumes: {len(building_tags)}")

    def _classify_patch(self, cx, cy, cz, xmin, ymin, zmin, xmax, ymax, zmax):
        """Classify a face center into a patch name with robust tolerances."""
        tol = 1e-4
        if abs(cx - xmin) <= tol:
            return "inlet"
        if abs(cx - xmax) <= tol:
            return "outlet"
        if abs(cy - ymin) <= tol:
            return "side_left"
        if abs(cy - ymax) <= tol:
            return "side_right"
        if abs(cz - zmax) <= tol:
            return "top"
        if abs(cz - zmin) <= tol:
            return "ground"
        return "buildings"

    def build_mesh(self, mesh_size: float = 5.0):
        """Generate the 3D mesh."""
        if not self._built:
            raise RuntimeError("build() must be called before build_mesh()")

        if self.building_tags:
            try:
                gmsh.model.occ.remove([(3, t) for t in self.building_tags])
                gmsh.model.occ.synchronize()
                print(f"  Removed {len(self.building_tags)} building volumes for meshing")
            except Exception as exc:
                print(f"  Warning: could not remove building volumes: {exc}")

        if self.fluid_tag is not None:
            try:
                fluid_surfs = gmsh.model.getBoundary([(3, self.fluid_tag)], oriented=False)
                surf_map = {}
                for dim, tag in fluid_surfs:
                    surf_map[tag] = surf_map.get(tag, 0) + 1
                duplicate_surfs = [tag for tag, count in surf_map.items() if count > 1]
                if duplicate_surfs:
                    print(f"  Removing {len(duplicate_surfs)} duplicate surfaces...")
                    gmsh.model.occ.remove([(2, t) for t in duplicate_surfs], recursive=False)
                    gmsh.model.occ.synchronize()
            except Exception as exc:
                print(f"  Warning: duplicate surface removal failed: {exc}")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 2.0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.1)
        gmsh.option.setNumber("Mesh.Smoothing", 10)

        gmsh.model.removePhysicalGroups()
        self._patches_assigned = False
        self.assign_patches()

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 2.0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.1)
        gmsh.option.setNumber("Mesh.Smoothing", 10)

        all_surfaces = gmsh.model.getEntities(dim=2)
        if all_surfaces:
            gmsh.model.mesh.setSize(all_surfaces, mesh_size)

        try:
            gmsh.model.mesh.clear()
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.clear()
            gmsh.model.mesh.generate(3)
        except Exception as exc:
            print(f"WARNING: Gmsh 3D meshing failed ({exc})")
            print("  Trying with Mesh.Algorithm3D=3...")
            try:
                gmsh.option.setNumber("Mesh.Algorithm3D", 3)
                gmsh.model.mesh.clear()
                gmsh.model.mesh.generate(3)
                print("  3D meshing succeeded with Algorithm3D=3")
            except Exception as exc2:
                print(f"  Retry also failed ({exc2})")
                raise

        self._meshed = True

    def export_openfoam(self, case_path: Path):
        """Export directly to OpenFOAM polyMesh using DirectOpenFOAMExporter."""
        if not self._meshed:
            raise RuntimeError("build_mesh() must be called before export_openfoam()")

        if not gmsh.model.getPhysicalGroups(dim=3) and self.fluid_tag is not None:
            try:
                gmsh.model.addPhysicalGroup(3, [self.fluid_tag], name="fluid")
            except Exception:
                pass

        exporter = DirectOpenFOAMExporter(case_path)
        exporter.export_single_region()
        print(f"OpenFOAM mesh exported to {case_path / 'constant' / 'polyMesh'}")

        try:
            msh_path = case_path / "mesh.msh"
            gmsh.write(str(msh_path))
            print(f"Gmsh mesh saved to {msh_path}")
        except Exception as exc:
            print(f"WARNING: could not save Gmsh mesh: {exc}")

    def finalize(self):
        gmsh.finalize()
