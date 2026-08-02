import gmsh
from pathlib import Path
import subprocess
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union

from foampilot.mesh.direct_openfoam_exporter import DirectOpenFOAMExporter

logger = logging.getLogger(__name__)

class GmshMesher:
    def __init__(self, parent, model_name: str = "cfd_model", verbose: bool = True):
        """Initialize the CFD geometry handler.
        
        Args:
            model_name: Name for the Gmsh model
            verbose: Whether to print progress messages
        """
        gmsh.initialize()
        gmsh.model.add(model_name)
        self.parent = parent                       
        self.case_path = parent.case_path 
        self.model_name = model_name
        self.domain_box = None
        self.boundary_conditions: Dict[str, List[int]] = {}
        self.materials: Dict[str, List[int]] = {}
        self.verbose = verbose
        self.unassigned_tag = "UNASSIGNED"
        self._log(f"Initialized GeometryCFD model \'{model_name}\'")

    def _log(self, message: str):
        """Internal logging method."""
        logger.info(f"[GeometryCFD] {message}")

    def load_geometry(self, filepath: Union[Path, str]) -> List[Tuple[int, int]]:
        """Load a STEP or STL geometry file.
        
        Args:
            filepath: Path to the geometry file
            
        Returns:
            List of (dimension, tag) pairs for the loaded entities
            
        Raises:
            ValueError: For unsupported file formats
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Geometry file not found: {filepath}")

        self._log(f"Loading geometry from {filepath}")

        if filepath.suffix.lower() == ".step":
            gmsh.merge(str(filepath))
        gmsh.model.occ.synchronize()

    # ================================================================
    #  PRIMITIVES GÉOMÉTRIQUES (API Python Gmsh)
    # ================================================================

    def add_point(self, x: float, y: float, z: float, lc: float = 1.0) -> int:
        """Add a point entity.

        Args:
            x, y, z: Coordinates.
            lc: Local mesh size at this point.

        Returns:
            The Gmsh tag of the created point.
        """
        tag = gmsh.model.occ.addPoint(x, y, z)
        if lc > 0:
            gmsh.model.mesh.setSize([(0, tag)], lc)
        self._log(f"Point {tag} added at ({x}, {y}, {z}) with lc={lc}")
        return tag

    def add_line(
        self, x1: float, y1: float, z1: float, x2: float, y2: float, z2: float, lc: float = 1.0
    ) -> int:
        """Add a line between two points.

        Args:
            x1, y1, z1: Start point coordinates.
            x2, y2, z2: End point coordinates.
            lc: Local mesh size.

        Returns:
            The Gmsh tag of the created line.
        """
        p1 = self.add_point(x1, y1, z1, lc)
        p2 = self.add_point(x2, y2, z2, lc)
        tag = gmsh.model.occ.addLine(p1, p2)
        self._log(f"Line {tag} created from point {p1} to {p2}")
        return tag

    def add_circle(
        self, cx: float, cy: float, z: float, radius: float, num_points: int = 64, lc: float = 1.0
    ) -> int:
        """Add a circle (closed curve) in the XY plane at height z.

        Args:
            cx, cy: Center of the circle.
            z: Z-coordinate (height).
            radius: Radius of the circle.
            num_points: Number of points on the circle.
            lc: Local mesh size.

        Returns:
            The Gmsh tag of the created circle wire.
        """
        angles = [2.0 * np.pi * i / num_points for i in range(num_points)]
        pts = [
            self.add_point(cx + radius * np.cos(a), cy + radius * np.sin(a), z, lc)
            for a in angles
        ]
        tags = []
        for i in range(len(pts)):
            tag = gmsh.model.occ.addLine(pts[i], pts[(i + 1) % len(pts)])
            tags.append(tag)
        wire = gmsh.model.occ.addCurveLoop(tags)
        self._log(f"Circle wire {wire} created at ({cx}, {cy}, {z}) r={radius}")
        return wire

    def add_rectangle(
        self, xmin: float, ymin: float, z: float, xmax: float, ymax: float, lc: float = 1.0
    ) -> int:
        """Add a rectangular surface in the XY plane at height z.

        Args:
            xmin, ymin: Bottom-left corner.
            z: Height.
            xmax, ymax: Top-right corner.
            lc: Local mesh size.

        Returns:
            Gmsh tag of the created surface.
        """
        p1 = self.add_point(xmin, ymin, z, lc)
        p2 = self.add_point(xmax, ymin, z, lc)
        p3 = self.add_point(xmax, ymax, z, lc)
        p4 = self.add_point(xmin, ymax, z, lc)
        l1 = gmsh.model.occ.addLine(p1, p2)
        l2 = gmsh.model.occ.addLine(p2, p3)
        l3 = gmsh.model.occ.addLine(p3, p4)
        l4 = gmsh.model.occ.addLine(p4, p1)
        wire = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
        surface = gmsh.model.occ.addPlaneSurface([wire])
        self._log(f"Rectangle surface {surface} at z={z}: ({xmin},{ymin})→({xmax},{ymax})")
        return surface

    def extrude_surface(
        self,
        surface_tag: int,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 1.0,
        num_layers: int = 1,
        lc: float = 1.0,
    ) -> Tuple[int, List[int], List[int]]:
        """Extrude a 2D surface into a 3D volume (prismatic layers).

        Args:
            surface_tag: Tag of the surface to extrude.
            dx, dy, dz: Extrusion vector.
            num_layers: Number of prism layers.
            lc: Local mesh size for the extruded elements.

        Returns:
            Tuple of (volume_tag, surface_tags, volume_tags).
        """
        surfaces = [(2, surface_tag)]
        result = gmsh.model.occ.extrude(
            surfaces,
            dx, dy, dz,
            numElements=[num_layers] if num_layers > 1 else None,
            heights=[lc] if num_layers > 1 else None,
        )
        self._log(f"Surface {surface_tag} extruded by ({dx}, {dy}, {dz})")
        return result

    def extrude_profile(
        self,
        curve_loop_tags: List[int],
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 1.0,
        num_layers: int = 1,
        lc: float = 1.0,
    ) -> Tuple[int, List[int]]:
        """Extrude a 2D profile (curve loop) into a 3D volume.

        Args:
            curve_loop_tags: Tags of the curve loops defining the profile.
            dx, dy, dz: Extrusion vector.
            num_layers: Number of prism layers along the extrusion.
            lc: Local mesh size.

        Returns:
            Tuple of (volume_tag, surface_tags).
        """
        curves = [(1, tag) for tag in curve_loop_tags]
        result = gmsh.model.occ.extrude(
            curves,
            dx, dy, dz,
            numElements=[num_layers] if num_layers > 1 else None,
            heights=[lc] if num_layers > 1 else None,
        )
        self._log(f"Profile extruded by ({dx}, {dy}, {dz})")
        return result

    def boolean_union(self, dim: int, tags: List[int]) -> List[Tuple[int, int]]:
        """Boolean union of entities using fuse.

        Args:
            dim: Dimension of the entities (2=surface, 3=volume).
            tags: List of entity tags.

        Returns:
            List of (dimension, tag) pairs of resulting entities.
        """
        objects = [(dim, t) for t in tags]
        result, _ = gmsh.model.occ.fuse(objects, [], removeObject=True, removeTool=True)
        self._log(f"Boolean union of {len(tags)} {dim}D entities → {len(result)} result(s)")
        return result

    def boolean_difference(
        self, dim: int, object_tags: List[int], tool_tag: int
    ) -> List[Tuple[int, int]]:
        """Boolean subtraction: objects - tool.

        Args:
            dim: Dimension of the entities.
            object_tags: Tags of entities to subtract from.
            tool_tag: Tag of the entity to subtract.

        Returns:
            List of (dimension, tag) pairs of resulting entities.
        """
        objects = [(dim, t) for t in object_tags]
        tool = [(dim, tool_tag)]
        result = gmsh.model.occ.cut(objects, tool, removeObject=True, removeTool=True)
        self._log(f"Boolean difference: {len(objects)} objects − tool → {len(result)} result(s)")
        return result

    def boolean_intersection(self, dim: int, tags: List[int]) -> List[Tuple[int, int]]:
        """Boolean intersection of entities.

        Args:
            dim: Dimension of the entities.
            tags: List of entity tags to intersect.

        Returns:
            List of (dimension, tag) pairs of resulting entities.
        """
        objects = [(dim, t) for t in tags]
        result = gmsh.model.occ.intersect(objects, [], removeObject=True, removeTool=True)
        self._log(f"Boolean intersection of {len(tags)} {dim}D entities → {len(result)} result(s)")
        return result

    # ================================================================
    #  NOMMAGE AUTOMATIQUE DES PATCHS PAR DIRECTION / NORMALE
    # ================================================================

    def assign_patches_by_normal(
        self,
        angle_tol: float = 15.0,
        custom_mapping: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ) -> Dict[str, List[int]]:
        """Assign patch names to faces based on their normal direction.

        Args:
            angle_tol: Tolerance in degrees for direction matching.
            custom_mapping: Optional dict {patch_name: (nx, ny, nz)} for
                user-defined direction→patch mapping.

        Defaults (axis-aligned):
            +X → INLET, -X → OUTLET
            +Y → SIDE_NORTH, -Y → SIDE_SOUTH
            +Z → TOP, -Z → GROUND

        Returns:
            Dictionary mapping patch names to face tags.
        """
        faces = gmsh.model.getEntities(dim=2)
        patch_map: Dict[str, List[int]] = {}

        for _, face in faces:
            edges = gmsh.model.getBoundary([face])
            if len(edges) >= 3:
                nodes = gmsh.model.mesh.getNodes(2, face[1])
                node_tags = nodes[1][:3]
                coord = gmsh.model.mesh.getCoordinates()
                pts = np.array(
                    [
                        [coord[3 * i], coord[3 * i + 1], coord[3 * i + 2]]
                        for i in node_tags
                    ]
                )
                v1 = pts[1] - pts[0]
                v2 = pts[2] - pts[0]
                normal = np.cross(v1, v2)
                mag = np.linalg.norm(normal)
                if mag > 0:
                    normal = normal / mag
                else:
                    continue
            else:
                continue

            patch_name = self._match_normal_to_patch(normal, angle_tol, custom_mapping)
            if patch_name not in patch_map:
                patch_map[patch_name] = []
            patch_map[patch_name].append(face)

        for patch_name, face_tags in patch_map.items():
            if face_tags:
                gid = gmsh.model.addPhysicalGroup(2, face_tags)
                gmsh.model.setPhysicalName(2, gid, patch_name)
                self.boundary_conditions[patch_name] = face_tags
                self._log(f"Assign patch '{patch_name}' to {len(face_tags)} face(s)")

        volumes = [v[1] for v in gmsh.model.getEntities(dim=3)]
        if volumes:
            gid = gmsh.model.addPhysicalGroup(3, volumes)
            gmsh.model.setPhysicalName(3, gid, "FLUID")

        gmsh.model.occ.synchronize()
        return patch_map

    def _match_normal_to_patch(
        self,
        normal: np.ndarray,
        angle_tol: float,
        custom_mapping: Optional[Dict[str, Tuple[float, float, float]]],
    ) -> str:
        """Match a face normal vector to a patch name.

        Args:
            normal: Normalized normal vector (nx, ny, nz).
            angle_tol: Tolerance angle in degrees.
            custom_mapping: Optional user-defined mapping.

        Returns:
            Patch name string.
        """
        default_mapping = {
            "TOP": ([0, 0, 1], angle_tol),
            "GROUND": ([0, 0, -1], angle_tol),
            "SIDE_NORTH": ([0, 1, 0], angle_tol),
            "SIDE_SOUTH": ([0, -1, 0], angle_tol),
            "INLET": ([1, 0, 0], angle_tol),
            "OUTLET": ([-1, 0, 0], angle_tol),
        }

        mapping = custom_mapping or default_mapping

        for name, (direction, tol) in mapping.items():
            d = np.array(direction) / (np.linalg.norm(direction) + 1e-12)
            dot = np.dot(normal, d)
            if dot > np.cos(np.radians(tol)):
                return name
            if dot < -np.cos(np.radians(tol)):
                return name

        return self.unassigned_tag

    # ================================================================
    #  DÉTECTION DE PATCH
    # ================================================================

    # ----------------------------------------------------
    # 3) DETECT PATCH BASED ON CENTER OF MASS OR NORMAL
    # ----------------------------------------------------
    def detect_patch(self, com, bbox, tol=15.0, normal=None):
        x, y, z = com
        
        # First try center of mass (works for non-rotated geometry)
        # Use tolerance proportional to domain size for rotated geometries
        if abs(x - bbox["xmin"]) < tol: return "INLET"
        if abs(x - bbox["xmax"]) < tol: return "OUTLET"
        if abs(z - bbox["zmin"]) < tol: return "GROUND"
        if abs(z - bbox["zmax"]) < tol: return "TOP"
        if abs(y - bbox["ymax"]) < tol: return "SIDE_NORTH"
        if abs(y - bbox["ymin"]) < tol: return "SIDE_SOUTH"
        
        # If normal is provided, try based on normal direction (works for rotated geometry)
        if normal is not None:
            nx, ny, nz = normal
            # Normalize
            mag = (nx**2 + ny**2 + nz**2)**0.5
            if mag > 0:
                nx, ny, nz = nx/mag, ny/mag, nz/mag
                
            # Check if normal points in -X direction (inlet)
            if nx < -0.9: return "INLET"
            # Check if normal points in +X direction (outlet)
            if nx > 0.9: return "OUTLET"
            # Check if normal points in -Y direction (SIDE_SOUTH)
            if ny < -0.9: return "SIDE_SOUTH"
            # Check if normal points in +Y direction (SIDE_NORTH)
            if ny > 0.9: return "SIDE_NORTH"
            # Check if normal points in -Z direction (ground)
            if nz < -0.9: return "GROUND"
            # Check if normal points in +Z direction (top)
            if nz > 0.9: return "TOP"
        
        return None

    # ----------------------------------------------------
    # 4) MAIN PATCH ASSIGNMENT
    # ----------------------------------------------------
    def assign_boundary_patches(self, **bbox_args):

        bbox = self.compute_bbox(**bbox_args)
        self.fragment_volumes()

        faces = gmsh.model.getEntities(dim=2)

        patch_map = {
            "INLET": [],
            "OUTLET": [],
            "GROUND": [],
            "TOP": [],
            "SIDE_NORTH": [],
            "SIDE_SOUTH": [],
            self.unassigned_tag: []
        }

        for _, face in faces:
            try:
                com = gmsh.model.occ.getCenterOfMass(2, face)
            except:
                patch_map[self.unassigned_tag].append(face)
                continue
            
            # Try to get the normal vector of the face
            normal = None
            try:
                # Get the normal by getting boundary edges and computing cross product
                edges = gmsh.model.getBoundary([face])
                if len(edges) >= 3:
                    # Get coordinates of first 3 vertices to compute normal
                    nodes = gmsh.model.mesh.getNodes(2, face[1])
                    if nodes[0] >= 3:
                        # Get node coordinates
                        node_tags = nodes[1][:3]
                        coord = gmsh.model.mesh.getCoordinates()
                        # Extract coordinates of the 3 nodes
                        pts = []
                        for nt in node_tags:
                            idx = list(nodes[1]).index(nt)
                            pts.append([nodes[2][3*idx], nodes[2][3*idx+1], nodes[2][3*idx+2]])
                        # Compute two edge vectors
                        v1 = [pts[1][0]-pts[0][0], pts[1][1]-pts[0][1], pts[1][2]-pts[0][2]]
                        v2 = [pts[2][0]-pts[0][0], pts[2][1]-pts[0][1], pts[2][2]-pts[0][2]]
                        # Cross product
                        normal = [
                            v1[1]*v2[2] - v1[2]*v2[1],
                            v1[2]*v2[0] - v1[0]*v2[2],
                            v1[0]*v2[1] - v1[1]*v2[0]
                        ]
            except:
                pass  # Use center of mass only

            patch = self.detect_patch(com, bbox, normal=normal)
            if patch:
                patch_map[patch].append(face)
            else:
                patch_map[self.unassigned_tag].append(face)

        # Create groups
        for patch, tags in patch_map.items():
            if tags:
                gid = gmsh.model.addPhysicalGroup(2, tags)
                gmsh.model.setPhysicalName(2, gid, patch)

        # Tag FLUID volume
        volumes = [v[1] for v in gmsh.model.getEntities(3)]
        if volumes:
            gid = gmsh.model.addPhysicalGroup(3, volumes)
            gmsh.model.setPhysicalName(3, gid, "FLUID")

        gmsh.model.occ.synchronize()




    def set_material(self, name: str, volume_tags: List[int]):
        """Assign a material name to volume(s).
        
        Args:
            name: Material name
            volume_tags: List of volume tags to assign to this material
        """
        self.materials[name] = volume_tags
        self._log(f"Assigned material \'{name}\' to {len(volume_tags)} volumes")

    def mesh_volume(self, lc_min: float = 1, lc_max: float = 5,
                    refine_regions: Optional[Dict[Tuple[float, float, float], Tuple[float, float]]] = None):
        """Generate a 3D mesh using TetGen and verify tetrahedra exist for OpenFOAM.

        Args:
            lc_min: Minimum characteristic length.
            lc_max: Maximum characteristic length.
            refine_regions: Optional dict {center: (radius, refined_lc)} for local refinement.
        """
        self._log(f"Generating 3D mesh (TetGen) with lc_min={lc_min}, lc_max={lc_max}")

        # Set global mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_max)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2)
        gmsh.option.setNumber("Mesh.Algorithm3D", 4)  # TetGen

        # Remove duplicates
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

        # Apply local refinements
        if refine_regions:
            for center, (radius, refined_lc) in refine_regions.items():
                entities = gmsh.model.getEntitiesInBoundingBox(
                    center[0]-radius, center[1]-radius, center[2]-radius,
                    center[0]+radius, center[1]+radius, center[2]+radius
                )
                if entities:
                    gmsh.model.mesh.setSize(entities, refined_lc)

        # Retrieve volumes
        volumes = [v[1] for v in gmsh.model.getEntities(dim=3)]
        if not volumes:
            self._log("No 3D volumes found. Cannot generate 3D mesh.")
            return

        # Vérifier s'il existe déjà un Physical Group pour les volumes
        existing_groups = gmsh.model.getPhysicalGroups(dim=3)
        fluid_group_exists = any(name == "FLUID" for (dim, tag) in existing_groups
                                for name in [gmsh.model.getPhysicalName(dim, tag)])

        if fluid_group_exists:
            self._log("Physical Group 'FLUID' already exists, skipping creation.")
        else:
            gmsh.model.addPhysicalGroup(3, volumes, name="FLUID")
            self._log(f"Physical Group 'FLUID' created for volumes: {volumes}")

        # Generate 3D mesh
        gmsh.model.mesh.generate(3)

    def get_unassigned_faces(self) -> list[int]:
        """Return the tags of faces in the 'UNASSIGNED' physical group (2D)."""
        phys_groups = gmsh.model.getPhysicalGroups(dim=2)

        # Chercher l'ID du groupe "UNASSIGNED"
        phys_id = next(
            (pid for dim, pid in phys_groups if gmsh.model.getPhysicalName(dim, pid) == self.unassigned_tag),
            None
        )

        if phys_id is not None:
            return gmsh.model.getEntitiesForPhysicalGroup(2, phys_id)
        return []


    def get_volume_tags(self) -> List[int]:
        """
        Retourne la liste des tags des volumes 3D dans le modèle.
        """
        volumes = gmsh.model.getEntities(dim=3)
        return [v[1] for v in volumes] if volumes else []

    def get_face_tags(self) -> List[int]:
        """
        Retourne la liste des tags des faces 2D dans le modèle.
        """
        faces = gmsh.model.getEntities(dim=2)
        return [f[1] for f in faces] if faces else []

    def get_basic_mesh_stats(self) -> Dict[str, int]:
        """Get basic mesh statistics (nodes, elements, surface elements).
        
        Returns:
            Dictionary containing:
            - num_nodes: Total node count
            - num_elements: Total element count
            - num_surface_elements: Surface element count
        """
        node_ids, _, _ = gmsh.model.mesh.getNodes()
        _, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
        _, surf_elem_tags, _ = gmsh.model.mesh.getElements(dim=2)

        return {
            "num_nodes": len(node_ids),
            "num_elements": len(elem_tags[0]) if elem_tags else 0,
            "num_surface_elements": sum(len(tags) for tags in surf_elem_tags) if surf_elem_tags else 0
        }

    def analyze_mesh_quality(self) -> Dict[str, float]:
        """Analyze mesh quality and return relevant metrics.
        
        Returns:
            Dictionary containing mesh quality metrics (e.g., min/max element size, aspect ratio).
        """
        self._log("Analyzing mesh quality...")

        # Initialize quality metrics
        metrics = {
            "min_element_size": float("inf"),
            "max_element_size": 0.0,
            "average_element_size": 0.0,
            "min_aspect_ratio": float("inf"),
            "max_aspect_ratio": 0.0,
            "average_aspect_ratio": 0.0,
            "num_bad_elements": 0
        }

        # Get all elements (2D and 3D)
        all_elements = gmsh.model.mesh.getElements()

        total_elements = 0
        total_size = 0.0
        total_aspect_ratio = 0.0

        for i, element_type in enumerate(all_elements[0]):
            element_tags = all_elements[1][i]
            node_tags_for_type = all_elements[2][i]

            for j, tag in enumerate(element_tags):
                # Get nodes for the current element
                # The node tags are already available in node_tags_for_type
                # We need to slice node_tags_for_type to get the nodes for the current element
                # The number of nodes per element type is fixed:
                # Triangle (2): 3 nodes
                # Tetrahedron (4): 4 nodes
                num_nodes_per_element = 0
                if element_type == 2: # Triangle
                    num_nodes_per_element = 3
                elif element_type == 4: # Tetrahedron
                    num_nodes_per_element = 4
                else:
                    continue # Skip unsupported element types

                start_index = j * num_nodes_per_element
                end_index = start_index + num_nodes_per_element
                element_node_ids = node_tags_for_type[start_index:end_index]

                coords = []
                for node_id in element_node_ids:
                    coord = gmsh.model.mesh.getNode(node_id)[0]
                    coords.append(coord)
                coords = np.array(coords)
                # Calculate element size (simple approximation for now)
                if element_type == 2: # Triangle
                    a = np.linalg.norm(coords[1] - coords[0])
                    b = np.linalg.norm(coords[2] - coords[1])
                    c = np.linalg.norm(coords[0] - coords[2])
                    s = (a + b + c) / 2.0
                    area = np.sqrt(s * (s - a) * (s - b) * (s - c)) if s * (s - a) * (s - b) * (s - c) > 0 else 0
                    element_size = np.sqrt(area) # Approximation

                    # Aspect ratio for triangle (simple approximation: longest edge / shortest edge)
                    edges = [a, b, c]
                    aspect_ratio = max(edges) / min(edges) if min(edges) > 0 else float("inf")

                elif element_type == 4: # Tetrahedron
                    # Volume of tetrahedron
                    v = abs(np.dot(coords[0] - coords[3], np.cross(coords[1] - coords[3], coords[2] - coords[3]))) / 6.0
                    element_size = v**(1/3) # Approximation

                    # Aspect ratio for tetrahedron (more complex, using radius ratio)
                    # For simplicity, we\'ll use a placeholder or a more robust library if available
                    aspect_ratio = 1.0 # Placeholder

                else:
                    element_size = 0.0
                    aspect_ratio = 1.0

                if element_size > 0:
                    metrics["min_element_size"] = min(metrics["min_element_size"], element_size)
                    metrics["max_element_size"] = max(metrics["max_element_size"], element_size)
                    total_size += element_size
                    total_elements += 1

                if aspect_ratio < float("inf"):
                    metrics["min_aspect_ratio"] = min(metrics["min_aspect_ratio"], aspect_ratio)
                    metrics["max_aspect_ratio"] = max(metrics["max_aspect_ratio"], aspect_ratio)
                    total_aspect_ratio += aspect_ratio

                    # Example: count elements with bad aspect ratio
                    if aspect_ratio > 10: # Threshold for bad quality
                        metrics["num_bad_elements"] += 1

        if total_elements > 0:
            metrics["average_element_size"] = total_size / total_elements
            metrics["average_aspect_ratio"] = total_aspect_ratio / total_elements

        self._log("Mesh quality analysis complete.")
        return metrics

    def export_to_openfoam(self, run_gmshtofoam: bool = True):
        """Export mesh to OpenFOAM format.
        
        Args:
            folder: Destination folder for OpenFOAM case
            run_gmshtofoam: Whether to run gmshToFoam conversion
        """
        folder = self.parent.case_path

        msh_path = folder / "mesh.msh"
        self._log(f"Exporting mesh to {msh_path}")
        gmsh.option.setNumber("Mesh.MshFileVersion", 2)

        gmsh.write(str(msh_path))

        if run_gmshtofoam:
            self._log("Running gmshToFoam conversion")
            try:
                subprocess.run(["gmshToFoam", str(msh_path.name)], cwd=str(folder), check=True)
                self._log("OpenFOAM conversion successful")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"gmshToFoam failed: {e}")
            except FileNotFoundError:
                raise RuntimeError("gmshToFoam not found - is OpenFOAM properly sourced?")

    def export_to_openfoam_direct(self, region_map: Optional[Dict[str, str]] = None) -> Union[Path, List[Path]]:
        """Export mesh to OpenFOAM native polyMesh format directly.

        Writes ``constant/polyMesh`` (or ``constant/<region>/polyMesh`` for
        CHT multi-region meshes) files directly from the Gmsh API, without
        invoking ``gmshToFoam``.

        Requires that ``gmsh.model.mesh.generate(3)`` has already been called.

        Args:
            region_map: Optional mapping ``{volume_phys_name: region_dir}``
                for multi-region CHT.  If *None*, a single-region mesh is
                written to ``constant/polyMesh``.

        Returns:
            Path of the written polyMesh directory (single-region) or a
            list of paths for multi-region.
        """
        folder = self.parent.case_path
        exporter = DirectOpenFOAMExporter(folder)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2)

        if region_map is not None:
            self._log("Direct-exporting multi-region CHT mesh")
            return exporter.export_multi_region(region_map)
        else:
            self._log("Direct-exporting single-region mesh")
            return exporter.export_single_region()

    def visualize(self):
        """Launch Gmsh GUI to visualize the geometry and mesh."""
        self._log("Launching Gmsh GUI")
        gmsh.fltk.run()

    def add_rectangle(self, x0: float, y0: float, z0: float, dx: float, dy: float,
                      dz: float, name: str, layer_thickness: Optional[float] = None) -> int:
        """Create a 3D rectangular box by extruding a surface with optional boundary layers.

        Args:
            x0, y0, z0: Origin corner coordinates.
            dx, dy, dz: Extents along x, y, z axes.
            name: Physical group name for the volume.
            layer_thickness: If provided, create boundary layers of this
                thickness during extrusion.

        Returns:
            The Gmsh tag of the created volume.
        """
        p1 = gmsh.model.occ.addPoint(x0, y0, z0)
        p2 = gmsh.model.occ.addPoint(x0 + dx, y0, z0)
        p3 = gmsh.model.occ.addPoint(x0 + dx, y0 + dy, z0)
        p4 = gmsh.model.occ.addPoint(x0, y0 + dy, z0)
        l1 = gmsh.model.occ.addLine(p1, p2)
        l2 = gmsh.model.occ.addLine(p2, p3)
        l3 = gmsh.model.occ.addLine(p3, p4)
        l4 = gmsh.model.occ.addLine(p4, p1)
        wire = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
        surface = gmsh.model.occ.addPlaneSurface([wire])
        gmsh.model.occ.synchronize()
        if layer_thickness and layer_thickness > 0:
            num_layers = max(1, int(round(dz / layer_thickness)))
            heights = [(i + 1) * layer_thickness / dz for i in range(num_layers)]
            result = gmsh.model.occ.extrude(
                [(2, surface)],
                0, 0, dz,
                numElements=[1] * num_layers,
                heights=heights,
            )
        else:
            result = gmsh.model.occ.extrude(
                [(2, surface)],
                0, 0, dz,
            )
        gmsh.model.occ.synchronize()
        volumes = [v[1] for v in gmsh.model.getEntities(dim=3)]
        if volumes:
            gid = gmsh.model.addPhysicalGroup(3, volumes, name=name)
            gmsh.model.setPhysicalName(3, gid, name)
        self._log(f"Box '{name}' created at ({x0},{y0},{z0}) dims=({dx},{dy},{dz})")
        return surface

    def add_circle(self, cx: float, cy: float, z0: float, radius: float,
                    name: str, layer_thickness: Optional[float] = None) -> int:
        """Create a circular cylinder by extruding a circle with optional boundary layers.

        Args:
            cx, cy: Center of the circle in the XY plane.
            z0: Z-coordinate of the circle plane.
            radius: Radius of the circle.
            name: Physical group name for the volume.
            layer_thickness: If provided, create boundary layers of this
                thickness during extrusion. The extrusion height equals the radius.

        Returns:
            The Gmsh tag of the created volume.
        """
        wire = gmsh.model.occ.addCircle(cx, cy, z0, radius)
        surface = gmsh.model.occ.addPlaneSurface([wire])
        gmsh.model.occ.synchronize()
        dz = radius
        if layer_thickness and layer_thickness > 0:
            num_layers = max(1, int(round(dz / layer_thickness)))
            heights = [(i + 1) * layer_thickness / dz for i in range(num_layers)]
            result = gmsh.model.occ.extrude(
                [(2, surface)],
                0, 0, dz,
                numElements=[1] * num_layers,
                heights=heights,
            )
        else:
            result = gmsh.model.occ.extrude(
                [(2, surface)],
                0, 0, dz,
            )
        gmsh.model.occ.synchronize()
        volumes = [v[1] for v in gmsh.model.getEntities(dim=3)]
        if volumes:
            gid = gmsh.model.addPhysicalGroup(3, volumes, name=name)
            gmsh.model.setPhysicalName(3, gid, name)
        self._log(f"Cylinder '{name}' created at ({cx},{cy},{z0}) r={radius}")
        return surface

    def add_cylinder_points(self, points_list: List[Tuple[float, float, float]],
                            name: str,
                            layer_thickness: Optional[float] = None) -> int:
        """Create a cylinder from a list of axis points and assign a physical group.

        Args:
            points_list: List of (x, y, z) tuples defining the cylinder axis.
                The first two points define the axis direction and length.
                The radius is set to 1.0 by default.
            name: Physical group name for the volume.
            layer_thickness: If provided, extrude the end faces with boundary
                layers of this thickness.

        Returns:
            The Gmsh tag of the created volume.
        """
        if len(points_list) < 2:
            raise ValueError("points_list must contain at least two points")
        p1 = points_list[0]
        p2 = points_list[1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        axis_length = np.sqrt(dx * dx + dy * dy + dz * dz)
        radius = max(axis_length * 0.1, 1.0) if axis_length > 0 else 1.0
        cyl_tag = gmsh.model.occ.addCylinder(p1[0], p1[1], p1[2], dx, dy, dz, radius)
        gmsh.model.occ.synchronize()
        if layer_thickness and layer_thickness > 0 and axis_length > 0:
            num_layers = max(1, int(round(axis_length / layer_thickness)))
            heights = [(i + 1) * layer_thickness / axis_length for i in range(num_layers)]
            faces = gmsh.model.getEntities(dim=2)
            for dim, tag in faces:
                com = gmsh.model.occ.getCenterOfMass(2, (dim, tag))
                if com is None:
                    continue
                dist_start = np.sqrt(
                    (com[0] - p1[0]) ** 2 + (com[1] - p1[1]) ** 2 + (com[2] - p1[2]) ** 2
                )
                dist_end = np.sqrt(
                    (com[0] - p2[0]) ** 2 + (com[1] - p2[1]) ** 2 + (com[2] - p2[2]) ** 2
                )
                if min(dist_start, dist_end) < axis_length * 0.01:
                    gmsh.model.occ.extrude(
                        [(dim, tag)],
                        0, 0, 0,
                        numElements=[1] * num_layers,
                        heights=heights,
                    )
            gmsh.model.occ.synchronize()
        volumes = [v[1] for v in gmsh.model.getEntities(dim=3)]
        if volumes:
            gid = gmsh.model.addPhysicalGroup(3, volumes, name=name)
            gmsh.model.setPhysicalName(3, gid, name)
        self._log(f"Cylinder '{name}' created from {p1} to {p2} r={radius:.4f}")
        return cyl_tag

    def extrude_2d_surface(self, surface_tag: int, dx: float = 0.0, dy: float = 0.0,
                           dz: float = 1.0, layer_thickness: Optional[float] = None,
                           name: Optional[str] = None) -> Tuple[int, List[int], List[int]]:
        """Extrude a 2D surface into a 3D volume with optional boundary layers.

        Args:
            surface_tag: Tag of the surface to extrude.
            dx, dy, dz: Extrusion vector.
            layer_thickness: If provided, create boundary layers of this thickness
                along the extrusion direction.
            name: Optional physical group name for the resulting volume.

        Returns:
            Tuple of (volume_tag, surface_tags, volume_tags).
        """
        total_height = np.sqrt(dx * dx + dy * dy + dz * dz)
        surfaces = [(2, surface_tag)]
        if layer_thickness and layer_thickness > 0 and total_height > 0:
            num_layers = max(1, int(round(total_height / layer_thickness)))
            heights = [(i + 1) * layer_thickness / total_height for i in range(num_layers)]
            result = gmsh.model.occ.extrude(
                surfaces,
                dx, dy, dz,
                numElements=[1] * num_layers,
                heights=heights,
            )
        else:
            result = gmsh.model.occ.extrude(surfaces, dx, dy, dz)
        gmsh.model.occ.synchronize()
        if name:
            volumes = [v[1] for v in gmsh.model.getEntities(dim=3)]
            if volumes:
                gid = gmsh.model.addPhysicalGroup(3, volumes, name=name)
                gmsh.model.setPhysicalName(3, gid, name)
        self._log(f"Surface {surface_tag} extruded by ({dx}, {dy}, {dz}) with name='{name}'")
        return result

    def assign_physical_groups(self, patch_map: Dict[str, List[Tuple[int, int]]]) -> None:
        """Assign physical groups to entities by patch name.

        Args:
            patch_map: Dictionary mapping patch names to lists of
                (dimension, tag) tuples.
        """
        for pname, entities in patch_map.items():
            if not entities:
                continue
            dim = entities[0][0]
            tags = [e[1] for e in entities]
            gid = gmsh.model.addPhysicalGroup(dim, tags)
            gmsh.model.setPhysicalName(dim, gid, pname)
            self._log(f"Physical group '{pname}' assigned to {len(tags)} entity(ies)")
        gmsh.model.occ.synchronize()

    def write_geo(self, filepath: Union[Path, str]) -> None:
        """Write the current geometry to a .geo file.

        Args:
            filepath: Path to the output .geo file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append("// Gmsh geometry file generated by foampilot")
        lines.append("")
        phys_groups = gmsh.model.getPhysicalGroups()
        for dim, ptag in phys_groups:
            pname = gmsh.model.getPhysicalName(dim, ptag)
            entity_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, ptag)
            if entity_tags:
                tags_str = ", ".join(str(t) for t in entity_tags)
                dim_name = ["Point", "Curve", "Surface", "Volume"][dim]
                lines.append(f'Physical {dim_name}("{pname}") = {{{tags_str}}};')
        filepath.write_text("\n".join(lines) + "\n")
        self._log(f"Geometry written to {filepath}")

    def finalize(self):
        """Finalize the Gmsh API session."""
        self._log("Finalizing Gmsh session")
        gmsh.finalize()

