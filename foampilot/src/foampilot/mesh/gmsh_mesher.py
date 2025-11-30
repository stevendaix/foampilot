import gmsh
from pathlib import Path
import subprocess
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

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
            gmsh.merge(str(filepath))
            gmsh.model.occ.synchronize()
        elif filepath.suffix.lower() == ".stl":
            raise ValueError(f"Run with snappyHexMesh for STL files, not direct import.")
        else:
            raise ValueError(f"Unsupported format: {filepath.suffix}")


    def merge_geometry(self, filepath: Union[Path, str]) -> List[Tuple[int, int]]:
        """Merge another geometry into the current model.
        
        Args:
            filepath: Path to the geometry file to merge
            
        Returns:
            List of (dimension, tag) pairs for the merged entities
        """
        self.load_geometry(filepath)
        self._log("Removing duplicate entities")
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

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

    # ----------------------------------------------------
    # 1) BOUNDING BOX
    # ----------------------------------------------------
    def compute_bbox(self, 
                     xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None):

        if xmin is not None:
            return {
                "xmin": xmin, "xmax": xmax,
                "ymin": ymin, "ymax": ymax,
                "zmin": zmin, "zmax": zmax,
            }

        else :
            raise ValueError("You must provide all bounding box parameters: xmin, xmax, ymin, ymax, zmin, zmax")



    # ----------------------------------------------------
    # 2) FRAGMENTATION
    # ----------------------------------------------------
    def fragment_volumes(self):
        volumes = gmsh.model.getEntities(dim=3)
        if not volumes:
            return

        try:
            gmsh.model.occ.fragment(volumes, [])
        except:
            for v in volumes:
                try: gmsh.model.occ.fragment([v], [])
                except: pass

        gmsh.model.occ.synchronize()

    # ----------------------------------------------------
    # 3) DETECT PATCH BASED ON CENTER OF MASS
    # ----------------------------------------------------
    def detect_patch(self, com, bbox, tol=1e-3):
        x, y, z = com

        if abs(x - bbox["xmin"]) < tol: return "INLET"
        if abs(x - bbox["xmax"]) < tol: return "OUTLET"
        if abs(z - bbox["zmin"]) < tol: return "GROUND"
        if abs(z - bbox["zmax"]) < tol: return "TOP"
        if abs(y - bbox["ymax"]) < tol: return "SIDE_NORTH"
        if abs(y - bbox["ymin"]) < tol: return "SIDE_SOUTH"

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

            patch = self.detect_patch(com, bbox)
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

    def visualize(self):
        """Launch Gmsh GUI to visualize the geometry and mesh."""
        self._log("Launching Gmsh GUI")
        gmsh.fltk.run()

    def finalize(self):
        """Finalize the Gmsh API session."""
        self._log("Finalizing Gmsh session")
        gmsh.finalize()

