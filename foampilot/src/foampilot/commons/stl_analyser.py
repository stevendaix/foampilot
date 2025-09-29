import pyvista as pv
from pathlib import Path
import numpy as np

class STLAnalyzer:
    """
    A class for analyzing and processing STL (Stereolithography) files.

    This class provides methods to load STL files, extract geometric properties,
    perform spatial queries, and calculate mesh-related parameters.

    Attributes:
        filename (Path): Path object pointing to the STL file
        mesh (pv.PolyData): PyVista mesh object containing the loaded STL geometry
        reader (pyvista.reader): Reader object for the STL file
    """

    def __init__(self, filename: Path):
        """
        Initialize the STL analyzer with a given STL file path.

        Args:
            filename (Path): Path object pointing to the STL file to analyze
        """
        if not filename.exists():
            raise FileNotFoundError(f"The file {filename} does not exist.")
        self.filename = filename
        self.mesh = None
        self.reader = pv.get_reader(str(self.filename))

    def load(self) -> pv.PolyData:
        """
        Load and return the STL file as a PyVista mesh.

        Returns:
            pv.PolyData: The loaded mesh object

        Note:
            The loaded mesh is also stored in the instance's mesh attribute.
        """
        try:
            self.mesh = self.reader.read()
            return self.mesh
        except Exception as e:
            raise IOError(f"Error loading STL file: {e}")

    def get_file_type(self) -> str:
        """
        Determine if the STL file is ASCII or Binary.

        Returns:
            str: 'ASCII' or 'Binary'
        """
        with open(self.filename, 'rb') as f:
            header = f.read(80).decode('utf-8', errors='ignore')
            if header.startswith('solid'):
                return 'ASCII'
            else:
                return 'Binary'

    def get_info(self) -> dict:
        """
        Get comprehensive geometric information about the loaded mesh.

        Returns:
            dict: Dictionary containing mesh properties including:
                - 'File Type': 'ASCII' or 'Binary'
                - 'Number of Points': Total vertices in the mesh
                - 'Number of Cells': Total cells/faces in the mesh
                - 'Number of Triangles': Total triangles in the mesh
                - 'Dimensions': Bounding box dimensions (xmin, xmax, ymin, ymax, zmin, zmax)
                - 'Surface Area': Total surface area of the mesh
                - 'Volume': Enclosed volume of the mesh (if watertight)
                - 'Center of Mass': (x, y, z) coordinates of the center of mass
                - 'Is Watertight': True if the mesh is watertight, False otherwise
                - 'Is Manifold': True if the mesh is manifold, False otherwise
                - 'Principal Axes': Principal axes of inertia
                - 'Moments of Inertia': Moments of inertia around the principal axes

        Raises:
            ValueError: If the mesh hasn't been loaded yet
        """
        if self.mesh is None:
            raise ValueError("The mesh has not been loaded. Please call 'load()' first.")

        info = {
            'File Type': self.get_file_type(),
            'Number of Points': self.mesh.n_points,
            'Number of Cells': self.mesh.n_cells,
            'Number of Triangles': self.mesh.n_faces,
            'Dimensions': self.mesh.bounds,
            'Surface Area': self.mesh.area,
            'Volume': self.mesh.volume,
            'Center of Mass': self.mesh.center_of_mass().tolist(),
            'Is Watertight': self.mesh.is_manifold and self.mesh.is_closed,
            'Is Manifold': self.mesh.is_manifold,
        }

        # Calculate principal axes and moments of inertia
        try:
            # PyVista does not have a direct `compute_inertia` method on PolyData for the full tensor.
            # However, we can compute the principal axes using `pyvista.principal_axes` on the mesh points.
            # For moments of inertia, it's more complex and usually involves integrating over the volume
            # or using a library specifically designed for rigid body dynamics.
            # For a simple approximation, we can use the bounding box or mass properties if available.
            # Given the context, we'll stick to what PyVista directly offers or can be easily derived.

            # Principal axes can be derived from the points directly
            principal_axes = pv.principal_axes(self.mesh.points)
            info["Principal Axes"] = principal_axes.tolist()

            # For moments of inertia, PyVista's `volume` and `center_of_mass` are available.
            # A full inertia tensor calculation for an arbitrary mesh is non-trivial.
            # We'll indicate that a direct calculation is not readily available via PyVista's core methods.
            info["Moments of Inertia"] = "Requires advanced geometric computation or specialized library (e.g., trimesh)"

        except Exception as e:
            info["Principal Axes"] = 'Could not calculate: ' + str(e)
            info["Moments of Inertia"] = 'Could not calculate: ' + str(e)
        return info

    def get_facets_data(self) -> dict:
        """
        Get the normal and vertices for each facet (triangle) in the mesh.

        Returns:
            dict: Dictionary containing:
                - 'normals': A numpy array of shape (n_triangles, 3) with the normal vector for each triangle.
                - 'vertices': A numpy array of shape (n_triangles, 3, 3) with the coordinates of the three vertices for each triangle.

        Raises:
            ValueError: If the mesh hasn't been loaded yet
        """
        if self.mesh is None:
            raise ValueError("The mesh has not been loaded. Please call 'load()' first.")

        # PyVista stores faces as [n_points_in_face, p1_idx, p2_idx, p3_idx, ...]
        # For triangles, it's [3, p1_idx, p2_idx, p3_idx]
        faces = self.mesh.faces.reshape(-1, 4)
        # Extract vertex indices for each triangle
        triangle_indices = faces[:, 1:]
        # Get the actual vertex coordinates
        vertices = self.mesh.points[triangle_indices]

        # Calculate normals (PyVista's mesh.face_normals is more robust)
        normals = self.mesh.face_normals

        return {
            'normals': normals,
            'vertices': vertices
        }

    def is_point_inside(self, point: tuple) -> bool:
        """
        Check if a 3D point lies inside the closed STL mesh.

        Args:
            point (tuple): (x, y, z) coordinates of the point to test

        Returns:
            bool: True if the point is inside the mesh, False otherwise

        Raises:
            ValueError: If the mesh hasn't been loaded yet
        """
        if self.mesh is None:
            raise ValueError("The mesh has not been loaded. Please call 'load()' first.")

        point_array = np.array(point).reshape(-1, 3)
        points = pv.PolyData(point_array)
        selected = self.mesh.select_enclosed_points(points, check_surface=False)
        return bool(selected['SelectedPoints'].max())

    def get_center_of_mass(self) -> tuple:
        """
        Calculate the center of mass of the STL mesh.

        Returns:
            tuple: (x, y, z) coordinates of the center of mass

        Raises:
            ValueError: If the mesh hasn't been loaded yet
        """
        if self.mesh is None:
            raise ValueError("The mesh has not been loaded. Please call 'load()' first.")
        center_of_mass = self.mesh.center_of_mass()
        return tuple(center_of_mass)

    def get_curvature(self) -> dict:
        """
        Calculate mean and Gaussian curvature values for the mesh.

        Returns:
            dict: Dictionary containing:
                - 'Mean_Curvature': Array of mean curvature values at each point
                - 'Gaussian_Curvature': Array of Gaussian curvature values at each point

        Raises:
            ValueError: If the mesh hasn't been loaded yet
        """
        if self.mesh is None:
            raise ValueError("The mesh has not been loaded. Please call 'load()' first.")

        mean_curvature = self.mesh.curvature(curv_type='mean')
        gaussian_curvature = self.mesh.curvature(curv_type='gaussian')
        return {
            'Mean_Curvature': mean_curvature,
            'Gaussian_Curvature': gaussian_curvature
        }

    def get_smallest_curvature(self) -> float:
        """
        Find the minimum mean curvature value in the mesh.

        Returns:
            float: The smallest mean curvature value

        Raises:
            ValueError: If the mesh hasn't been loaded yet
        """
        curvature_values = self.get_curvature()['Mean_Curvature']
        smallest_curvature = curvature_values.min()
        return float(smallest_curvature)

    # The calc_mesh_settings method and its dependencies (stlAnalysis.calc_domain_size, etc.)
    # are outside the scope of general STL analysis and seem to be specific to CFD simulations.
    # They also rely on an undefined 'stlAnalysis' module. I will remove them for now.
    # If these functionalities are still required, the 'stlAnalysis' module needs to be provided or implemented.


if __name__ == "__main__":
    # Create a dummy STL file for testing purposes
    from pyvista import examples
    dummy_stl_file = Path.cwd() / "dummy_mesh.stl"
    mesh_to_save = examples.download_bunny()
    mesh_to_save.save(dummy_stl_file)

    analyzer = STLAnalyzer(dummy_stl_file)

    # Load the mesh
    mesh = analyzer.load()

    # Get comprehensive mesh information
    info = analyzer.get_info()
    print("\n--- Comprehensive Mesh Information ---")
    for key, value in info.items():
        print(f"{key}: {value}")

    # Get facets data (normals and vertices)
    facets_data = analyzer.get_facets_data()
    print("\n--- Facets Data (first 5 triangles) ---")
    print(f"Normals shape: {facets_data['normals'].shape}")
    print(f"Vertices shape: {facets_data['vertices'].shape}")
    print("First 5 Normals:\n", facets_data['normals'][:5])
    print("First 5 Vertices:\n", facets_data['vertices'][:5])

    # Check if a point is inside the mesh
    # For the bunny, a point near its center should be inside
    test_point_inside = mesh.center
    inside = analyzer.is_point_inside(test_point_inside)
    print(f"\nIs point {test_point_inside} inside the mesh: {inside}")

    test_point_outside = (mesh.bounds[1] + 1, mesh.bounds[3] + 1, mesh.bounds[5] + 1) # A point outside the bounding box
    outside = analyzer.is_point_inside(test_point_outside)
    print(f"Is point {test_point_outside} inside the mesh: {outside}")

    # Calculate the center of mass
    center_of_mass = analyzer.get_center_of_mass()
    print(f"\nThe center of mass is: {center_of_mass}")

    # Calculate the curvature
    curvature = analyzer.get_curvature()
    print(f"\nMean Curvature (first 5 values): {curvature['Mean_Curvature'][:5]}")
    print(f"Gaussian Curvature (first 5 values): {curvature['Gaussian_Curvature'][:5]}")

    # Calculate the minimum curvature
    smallest_curvature = analyzer.get_smallest_curvature()
    print(f"The minimum mean curvature is: {smallest_curvature}")

    # Clean up the dummy file
    dummy_stl_file.unlink()