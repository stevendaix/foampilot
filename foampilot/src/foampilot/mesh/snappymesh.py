import pyvista as pv
from pathlib import Path
import numpy as np
import subprocess


class SnappyMesher:
    """
    Class for configuring and generating snappyHexMeshDict based on STL geometry.

    Attributes:
    - base_path (Path): Path to the OpenFOAM case directory.
    - stl_file (Path): Path to the STL file.
    - snappy_hex_mesh_dict_path (Path): Path to the snappyHexMeshDict file.
    """

    def __init__(self, parent, stl_file, castellatedMesh=True, snap=True, addLayers=False):
        """
        Initialize snappyHexMesh main options.

        Args:
            base_path (str): Path to the OpenFOAM directory containing the case.
            stl_file (str): Path to the STL geometry file.
            castellatedMesh (bool): Enable initial castellated mesh structure.
            snap (bool): Enable mesh projection onto the STL surface.
            addLayers (bool): Enable the addition of boundary layers.
        """
        
        self.parent = parent                       
        self.case_path = parent.case_path 
        self.snappy_hex_mesh_dict_path = self.base_path / "system" / "snappyHexMeshDict"
        self.stl_file = Path(stl_file)
        
        self.locationInMesh = (0.1, 0.1, 0.1)

        self.castellatedMesh = castellatedMesh
        self.snap = snap
        self.addLayers = addLayers
        
        self.geometry = {
            self.stl_file.stem: {
                "type": "triSurfaceMesh",
                "name": self.stl_file.stem,
                "regions": {}
            }
        }
        
        self.castellatedMeshControls = {
            "maxLocalCells": 100000,
            "maxGlobalCells": 2000000,
            "minRefinementCells": 10,
            "nCellsBetweenLevels": 3,
            "locationInMesh": self.locationInMesh,
            "refinementSurfaces": {
                self.stl_file.stem: {"level": (2, 3)}
            },
            "features": [],
            "refinementRegions": {}
        }

        self.snapControls = {
            "nSmoothPatch": 3,
            "tolerance": 2.0,
            "nSolveIter": 30,
            "nRelaxIter": 5,
            "nFeatureSnapIter": 10,
            "implicitFeatureSnap": False,
            "explicitFeatureSnap": True,
            "multiRegionFeatureSnap": False
        }

        self.addLayersControls = {
            "relativeSizes": True,
            "expansionRatio": 1.2,
            "finalLayerThickness": 0.5,
            "minThickness": 0.1,
            "layers": {
                self.stl_file.stem: {
                    "nSurfaceLayers": 3
                }
            }
        }

        self.meshQualityControls = {
            "maxNonOrtho": 75,
            "maxBoundarySkewness": 20,
            "maxInternalSkewness": 4,
            "maxConcave": 80,
            "minVol": 1.0e-13,
            "minTetQuality": 1e-15,
            "minArea": -1,
            "minTwist": 0.02,
            "minDeterminant": 0.001,
            "minFaceWeight": 0.05,
            "minVolRatio": 0.01,
            "minTriangleTwist": -1,
            "minFlatness": 0.5,
            "nSmoothScale": 4,
            "errorReduction": 0.75
        }

        self.debugFlags = []
        self.writeFlags = []

    def add_feature(self, feature_file, level):
        """
        Adds a feature edge file (extracted with surfaceFeatureExtract) to refine geometry edges.
        
        Args:
            feature_file (str): Path to the .eMesh file.
            level (int): Refinement level for edge features.
        """
        self.castellatedMeshControls["features"].append({
            "file": feature_file,
            "level": level
        })

    def add_refinement_region(self, name, mode, levels):
        """
        Adds a specific refinement region.

        Args:
            name (str): Name of the region in the geometry.
            mode (str): Refinement mode (e.g., 'inside', 'outside').
            levels (tuple): Refinement levels for the region (e.g., ((1, 2))).
        """
        self.castellatedMeshControls["refinementRegions"][name] = {
            "mode": mode,
            "levels": levels
        }

    def add_layer(self, surface, n_surface_layers):
        """
        Sets the number of mesh layers around a specific surface.

        Args:
            surface (str): Name of the surface.
            n_surface_layers (int): Number of surface layers.
        """
        self.addLayersControls["layers"][surface] = {"nSurfaceLayers": n_surface_layers}

    def write_snappyHexMeshDict(self):
        """
        Generate the snappyHexMeshDict file with the defined options for snappyHexMesh.
        """
        # Write configuration data to snappyHexMeshDict in OpenFOAM format here.
        pass


# Integrated Example
if __name__ == "__main__":
    # Path to the STL file
    stl_file = Path.cwd() / "Chess_Pawn.stl"
    
    # Initialize STL Analyzer
    analyzer = STLAnalyzer(stl_file)
    mesh = analyzer.load()

    # Compute dimensions and center for SnappyHexMesh configuration
    max_dim = analyzer.get_max_dim()
    center_of_mass = analyzer.get_center_of_mass()

    # Initialize SnappyHexMesh with base parameters
    base_path = Path.cwd() / "openfoam_case"
    snappy_hex_mesh = SnappyHexMesh(
        base_path=base_path,
        stl_file=stl_file,
        castellatedMesh=True,
        snap=True,
        addLayers=True
    )
    
    # Auto-configure SnappyHexMesh options
    snappy_hex_mesh.locationInMesh = center_of_mass
    snappy_hex_mesh.add_refinement_region("main_region", mode="inside", levels=((1, 2)))
    snappy_hex_mesh.add_layer(surface=stl_file.stem, n_surface_layers=3)

    # Automatically configure the domain size
    domain_size = analyzer.calc_domain_size(sizeFactor=max_dim * 0.5)
    print(f"Domain size: X={domain_size[0]}, Y={domain_size[1]}, Z={domain_size[2]}")

    # Write snappyHexMeshDict
    snappy_hex_mesh.write_snappyHexMeshDict()
    print("snappyHexMeshDict has been generated successfully.")


