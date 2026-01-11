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
    # ----------------------
    # SurfaceFeaturesDict
    # ----------------------
    def write_surface_features_dict(self, stl_list=None, included_angle=30):
        """
        Write system/surfaceFeaturesDict based on STL files
        """
        if stl_list is None:
            stl_list = [f"{self.stl_file.stem}.stl"]

        system_path = self.case_path / "system"
        system_path.mkdir(exist_ok=True, parents=True)
        dict_file = system_path / "surfaceFeaturesDict"

        lines = [
            "FoamFile",
            "{",
            "    version     2.0;",
            "    format      ascii;",
            "    class       dictionary;",
            "    object      surfaceFeaturesDict;",
            "}",
            "",
            "surfaces",
            "("
        ]
        for stl in stl_list:
            lines.append(f'    "{stl}"')
        lines.append(");")
        lines.append(f"\nincludedAngle   {included_angle};\n")

        dict_file.write_text("\n".join(lines))
        print(f"surfaceFeaturesDict written to {dict_file}")

    # ----------------------
    # Utilities
    # ----------------------
    def run_surface_feature_extract(self):
        """
        Runs surfaceFeatureExtract utility for the case.
        """
        cmd = ["surfaceFeatureExtract", "-case", str(self.case_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Error running surfaceFeatureExtract:")
            print(result.stderr)
        else:
            print("surfaceFeatureExtract finished successfully.")
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

    def write(self):
        """Writes all meshing-related configuration files to the disk.

        This method triggers the `write` methods of the primary mesher 
        (e.g., generating `blockMeshDict`) and then iterates through any 
        `additional_files` to write them into the `system/` directory of the case.

        Note:
            Ensure that `self.case_path` is writable before calling this method.
        """
        # Write primary mesher files (blockMesh, snappy, gmsh)
        # Note: In the original snippet, self.blockmesh and self.snappy 
        # were called directly. Assuming these are handled by self.mesher:
        if hasattr(self.mesher, 'write'):
            self.mesher.write()
        
        # Write extra files to the system directory
        system_path = self.case_path / "system"
        system_path.mkdir(exist_ok=True)
        
        for fname, fcontent in self.additional_files.items():
            fcontent.write(system_path / fname)

    def run(self):
        """
        Runs snappyHexMesh in the given OpenFOAM case.
        """
        case_path = Path(case_path)

        base_path = self.parent.case_path
        log_file = base_path / "log.blockMesh"

        if not base_path.exists():
            raise FileNotFoundError(f"The case path '{base_path}' does not exist.")

        if not base_path.is_dir():
            raise NotADirectoryError(f"The case path '{base_path}' is not a directory.")

        cmd = ["snappyHexMesh", "-overwrite", "-case", str(case_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Error running snappyHexMesh:")
            print(result.stderr)
        else:
            print("snappyHexMesh finished successfully.")
            print(result.stdout)