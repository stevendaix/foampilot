import gmsh
from pathlib import Path
import subprocess
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class GeometryCFD:
    def __init__(self, model_name: str = "cfd_model", verbose: bool = True):
        """Initialize the CFD geometry handler.
        
        Args:
            model_name: Name for the Gmsh model
            verbose: Whether to print progress messages
        """
        gmsh.initialize()
        gmsh.model.add(model_name)
        self.model_name = model_name
        self.domain_box = None
        self.boundary_conditions: Dict[str, List[int]] = {}
        self.materials: Dict[str, List[int]] = {}
        self.verbose = verbose
        self._log(f"Initialized GeometryCFD model \'{model_name}\'")

    def _log(self, message: str):
        """Internal logging method."""
        if self.verbose:
            print(f"[GeometryCFD] {message}")

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
            entities = gmsh.model.occ.importShapes(str(filepath))
            gmsh.model.occ.synchronize()
        elif filepath.suffix.lower() == ".stl":
            gmsh.merge(str(filepath))
            # Pour les STL, nous devons créer une topologie à partir des surfaces discrètes
            gmsh.model.mesh.createTopology()
            gmsh.model.occ.synchronize()
            entities = gmsh.model.getEntities()
        else:
            raise ValueError(f"Unsupported format: {filepath.suffix}")

        return entities

    def merge_geometry(self, filepath: Union[Path, str]) -> List[Tuple[int, int]]:
        """Merge another geometry into the current model.
        
        Args:
            filepath: Path to the geometry file to merge
            
        Returns:
            List of (dimension, tag) pairs for the merged entities
        """
        entities = self.load_geometry(filepath)
        self._log("Removing duplicate entities")
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
        return entities

    def wrap_surfaces(self, angle: float = 40.0):
        """Clean and wrap surfaces (useful for STL files).
        
        Args:
            angle: Feature angle in degrees for surface classification
        """
        self._log(f"Wrapping surfaces with angle threshold {angle}°")
        angle_rad = angle * (np.pi / 180)
        gmsh.model.mesh.classifySurfaces(angle_rad)
        gmsh.model.mesh.createGeometry()
        gmsh.model.occ.synchronize()

        # Optionnel: Remesher les surfaces pour améliorer la qualité après la classification
        # gmsh.option.setNumber("Mesh.Algorithm", 6) # Delaunay
        # gmsh.option.setNumber("Mesh.MeshSizeFactor", 0.1) # Taille de maille pour le remeshing
        # gmsh.model.mesh.generate(2) # Générer un maillage 2D sur les surfaces
        # gmsh.model.occ.synchronize() # Synchroniser pour mettre à jour la géométrie OCC

    def define_physical_group(self, dim: int, tags: List[int], name: str) -> int:
        """Define a named physical group.
        
        Args:
            dim: Dimension of entities (0=point, 1=line, 2=surface, 3=volume)
            tags: List of entity tags
            name: Name for the physical group
            
        Returns:
            Physical group ID
        """
        if not tags:
            raise ValueError("Empty tag list provided")

        phys_id = gmsh.model.addPhysicalGroup(dim, tags)
        gmsh.model.setPhysicalName(dim, phys_id, name)
        self.boundary_conditions[name] = tags
        self._log(f"Created physical group \'{name}\' (dim={dim}) with {len(tags)} entities")
        return phys_id

    def define_surface_group_by_tag(self, tag: int, name: str) -> int:
        """Define a physical group from a single surface tag."""
        return self.define_physical_group(2, [tag], name)

    def define_all_surfaces_group(self, name: str) -> int:
        """Define a physical group containing all surfaces."""
        surfaces = gmsh.model.getEntities(dim=2)
        surface_tags = [s[1] for s in surfaces]
        return self.define_physical_group(2, surface_tags, name)

    def detect_inlets_outlets(self, 
                           surface_tags_to_process: Optional[List[int]] = None,
                           normal_tolerance: float = 0.2,
                           inlet_name: str = "inlet",
                           outlet_name: str = "outlet",
                           wall_name: str = "walls") -> Dict[str, List[int]]:
        """Automatically detect inlets and outlets based on surface normals.
        
        Args:
            surface_tags_to_process: Optional list of surface tags to process. If None, processes all surfaces.
            normal_tolerance: Tolerance for considering a surface normal aligned with axis
            inlet_name: Name for inlet surfaces
            outlet_name: Name for outlet surfaces
            wall_name: Name for remaining surfaces
            
        Returns:
            Dictionary of detected boundary groups
        """
        self._log("Detecting inlets and outlets based on surface normals")

        if surface_tags_to_process is None:
            surfaces = gmsh.model.getEntities(dim=2)
            surface_tags_to_process = [s[1] for s in surfaces]

        inlets = []
        outlets = []
        walls = []

        for tag in surface_tags_to_process:
            # Get surface normal (approximate using first triangle)
            nodes, _, _ = gmsh.model.mesh.getNodes(dim=2, tag=tag)
            if len(nodes) < 3:
                continue

            # Get coordinates of first three nodes
            coords = gmsh.model.mesh.getNode(nodes[0])[0]
            p1 = np.array(coords)
            coords = gmsh.model.mesh.getNode(nodes[1])[0]
            p2 = np.array(coords)
            coords = gmsh.model.mesh.getNode(nodes[2])[0]
            p3 = np.array(coords)

            # Compute normal vector
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)

            # Classify based on normal direction
            if normal[0] > (1 - normal_tolerance):
                outlets.append(tag)
            elif normal[0] < -(1 - normal_tolerance):
                inlets.append(tag)
            elif abs(normal[1]) > (1 - normal_tolerance) or abs(normal[2]) > (1 - normal_tolerance):
                walls.append(tag)
            else:
                walls.append(tag)

        # Create physical groups
        results = {}
        if inlets:
            self.define_physical_group(2, inlets, inlet_name)
            results[inlet_name] = inlets
        if outlets:
            self.define_physical_group(2, outlets, outlet_name)
            results[outlet_name] = outlets
        if walls:
            self.define_physical_group(2, walls, wall_name)
            results[wall_name] = walls

        return results

    def create_external_domain(self, padding: float = 1.0) -> int:
        """Create an external air domain around existing geometry.
        
        Args:
            padding: Padding distance around the geometry
            
        Returns:
            Tag of the created box volume
        """
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.getEntities(dim=3)

        # Déterminer la boîte englobante de la géométrie existante
        self._log("Calculating bounding box from existing entities.")

        # Get all entities in the model
        all_entities = gmsh.model.getEntities()

        # Initialize min/max coordinates
        xmin, ymin, zmin = float("inf"), float("inf"), float("inf")
        xmax, ymax, zmax = float("-inf"), float("-inf"), float("-inf")

        if not all_entities:
            raise RuntimeError("No entities found in the model to calculate bounding box.")

        for dim, tag in all_entities:
            bbox = gmsh.model.getBoundingBox(dim, tag)
            xmin = min(xmin, bbox[0])
            ymin = min(ymin, bbox[1])
            zmin = min(zmin, bbox[2])
            xmax = max(xmax, bbox[3])
            ymax = max(ymax, bbox[4])
            zmax = max(zmax, bbox[5])

        self._log(f"Bounding box: ({xmin:.2f}, {ymin:.2f}, {zmin:.2f}) to ({xmax:.2f}, {ymax:.2f}, {zmax:.2f})")

        # Créer une boîte englobante plus grande comme domaine externe
        external_box_tag = gmsh.model.occ.addBox(
            xmin - padding, ymin - padding, zmin - padding,
            (xmax - xmin) + 2 * padding,
            (ymax - ymin) + 2 * padding,
            (zmax - zmin) + 2 * padding
        )
        gmsh.model.occ.synchronize()
        self._log(f"External domain (bounding box) created with tag: {external_box_tag}")

        # Pour les STL, nous ne créons pas de volume solide à partir des surfaces directement.
        # Nous allons simplement définir le domaine externe comme le volume fluide.
        # Les surfaces du STL serviront de frontières internes pour le maillage.
        self.domain_box = external_box_tag
        fluid_volume_tag = self.domain_box
        self.define_physical_group(3, [fluid_volume_tag], "fluid_domain")

        # Créer un volume physique pour les surfaces du STL
        stl_surfaces = gmsh.model.getEntities(dim=2)
        stl_surface_tags = [s[1] for s in stl_surfaces]
        if stl_surface_tags:
            self.define_physical_group(2, stl_surface_tags, "stl_internal_walls")

        return fluid_volume_tag

    def set_material(self, name: str, volume_tags: List[int]):
        """Assign a material name to volume(s).
        
        Args:
            name: Material name
            volume_tags: List of volume tags to assign to this material
        """
        self.materials[name] = volume_tags
        self._log(f"Assigned material \'{name}\' to {len(volume_tags)} volumes")

    def mesh_volume(self, lc: float = 0.01, refine_regions: Optional[Dict[Tuple[float, float, float], Tuple[float, float]]] = None):
        """Generate volume mesh with optional local refinement.
        
        Args:
            lc: Global characteristic length
            refine_regions: Dictionary of {center: (radius, refined_lc)} for local refinement
        """
        self._log(f"Generating mesh with characteristic length {lc}")

        # Set global mesh size
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

        # Add local refinement if specified
        if refine_regions:
            for center, (radius, refined_lc) in refine_regions.items():
                gmsh.model.mesh.setSize(gmsh.model.getEntitiesInBoundingBox(
                    center[0]-radius, center[1]-radius, center[2]-radius,
                    center[0]+radius, center[1]+radius, center[2]+radius
                ), refined_lc)

        # Nettoyer les surfaces avant de mailler
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

        # Vérifier si des volumes existent et si le domaine fluide est défini
        volumes = gmsh.model.getEntities(dim=3)
        if self.domain_box and (3, self.domain_box) in volumes:
            self._log(f"Fluid volume (tag {self.domain_box}) detected. Generating 3D mesh.")
            # Set the background mesh field to define mesh size based on distance to the STL surfaces
            # This ensures a finer mesh near the STL object.
            gmsh.model.mesh.field.add("Distance", 1)
            gmsh.model.mesh.field.setNumbers(1, "FacesList", [s[1] for s in gmsh.model.getEntities(dim=2)])

            gmsh.model.mesh.field.add("Threshold", 2)
            gmsh.model.mesh.field.setNumber(2, "InField", 1)
            gmsh.model.mesh.field.setNumber(2, "SizeMin", lc / 5) # Finer mesh near STL
            gmsh.model.mesh.field.setNumber(2, "SizeMax", lc) # Coarser mesh away from S        # Vérifier si des volumes existent et si le domaine fluide est défini
        volumes = gmsh.model.getEntities(dim=3)
        if self.domain_box and (3, self.domain_box) in volumes:
            self._log(f"Fluid volume (tag {self.domain_box}) detected. Generating 3D mesh.")
            # Set the background mesh field to define mesh size based on distance to the STL surfaces
            # This ensures a finer mesh near the STL obj            # Clear existing mesh fields before adding new ones
            existing_field_ids = gmsh.model.mesh.field.getNumbers(-1, -1) # Use -1, -1 to get all field numbers
            for field_id in existing_field_ids:
                gmsh.model.mesh.field.remove(field_id)

            gmsh.model.mesh.field.add("Distance", 1)
            gmsh.model.mesh.field.setNumbers(1, "FacesList", [s[1] for s in gmsh.model.getEntities(dim=2)])

            gmsh.model.mesh.field.add("Threshold", 2)
            gmsh.model.mesh.field.setNumber(2, "InField", 1)
            gmsh.model.mesh.field.setNumber(2, "SizeMin", lc / 5) # Finer mesh near STL
            gmsh.model.mesh.field.setNumber(2, "SizeMax", lc) # Coarser mesh away from STL
            gmsh.model.mesh.field.setNumber(2, "DistMin", 0.1) # Distance from STL where mesh is finest
            gmsh.model.mesh.field.setNumber(2, "DistMax", 1.0) # Distance from STL where mesh is coarsest

            gmsh.model.mesh.field.setAsBackgroundMesh(2)         # Essayer un algorithme de maillage 3D plus robuste
            gmsh.option.setNumber("Mesh.Algorithm3D", 4) # 4 pour TetGen
            gmsh.model.mesh.generate(3)
        else:
            self._log("No fluid volume detected or fluid volume not found. Generating 2D mesh on surfaces only.")
            # Pour les STL sans volume défini, on maille les surfaces
            gmsh.model.mesh.generate(2)
        self._log("Mesh generation complete")

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

    def export_to_openfoam(self, folder: Union[Path, str], run_gmshtofoam: bool = True):
        """Export mesh to OpenFOAM format.
        
        Args:
            folder: Destination folder for OpenFOAM case
            run_gmshtofoam: Whether to run gmshToFoam conversion
        """
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)

        msh_path = folder / "mesh.msh"
        self._log(f"Exporting mesh to {msh_path}")
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

    def visualize(self):
        """Launch Gmsh GUI to visualize the geometry and mesh."""
        self._log("Launching Gmsh GUI")
        gmsh.fltk.run()

    def finalize(self):
        """Finalize the Gmsh API session."""
        self._log("Finalizing Gmsh session")
        gmsh.finalize():
        """Clean up and close Gmsh."""
        self._log("Finalizing Gmsh session")
        gmsh.finalize()


# Example usage
if __name__ == "__main__":
    # Create a new CFD case
    case = GeometryCFD("tutorial_case")

    try:
        # Load main geometry
        case.load_geometry("geometry.step")

        # Merge additional components
        case.merge_geometry("inlet_pipe.step")

        # Wrap surfaces if needed (for STL)
        # case.wrap_surfaces()

        # Automatic boundary detection
        case.detect_inlets_outlets()

        # Alternatively, manual boundary definition
        # case.define_all_surfaces_group("walls")
        # case.define_surface_group_by_tag(10, "inlet")

        # Create external domain
        case.create_external_domain(padding=2.0)

        # Assign materials
        case.set_material("fluid", [1])
        case.set_material("solid", [2])

        # Generate mesh with local refinement near origin
        case.mesh_volume(lc=0.1, refine_regions={
            (0, 0, 0): (1.0, 0.02)  # Refine to 0.02 within 1m of origin
        })

        # Print mesh statistics
        print("Mesh statistics:", case.get_mesh_stats())

        # Export to OpenFOAM
        case.export_to_openfoam("openfoam_case")

        # Visualize in Gmsh GUI
        case.visualize()

    finally:
        case.finalize()