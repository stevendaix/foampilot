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
        self.snappy_hex_mesh_dict_path = self.case_path / "system" / "snappyHexMeshDict"
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
            "layers": {}
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
        dict_path = self.case_path / "system" / "snappyHexMeshDict"
        self.case_path.joinpath("system").mkdir(parents=True, exist_ok=True)

        lines = [
            "FoamFile",
            "{",
            "    version     2.0;",
            "    format      ascii;",
            "    class       dictionary;",
            "    location    \"system\";",
            "    object      snappyHexMeshDict;",
            "}",
            "",
            f"castellatedMesh {str(self.castellatedMesh).lower()};",
            f"snap {str(self.snap).lower()};",
            f"addLayers {str(self.addLayers).lower()};",
            "",
            "geometry",
            "{"
        ]

        # Geometry
        for name, geo in self.geometry.items():
            stl_file = geo["file"] if "file" in geo else f"{name}.stl"
            lines += [
                f'    "{stl_file}"',
                "    {",
                f'        type {geo.get("type", "triSurfaceMesh")};',
                f'        file "{stl_file}";',
                f'        name {geo.get("name", name)};',
                "    }"
            ]
        lines.append("};\n")

        # CastellatedMeshControls
        cm = self.castellatedMeshControls
        lines.append("castellatedMeshControls")
        lines.append("{")
        for key in ["maxLocalCells","maxGlobalCells","minRefinementCells","maxLoadUnbalance","nCellsBetweenLevels"]:
            if key in cm:
                lines.append(f"    {key} {cm[key]};")
        lines.append(f"    locationInMesh ({' '.join(map(str, self.locationInMesh))});")
        # Features
        lines.append("    features")
        lines.append("    (")
        for f in cm.get("features", []):
            lines.append("        {")
            lines.append(f'            file "{f["file"]}";')
            lines.append(f'            level {f["level"]};')
            lines.append("        }")
        lines.append("    );")
        # refinementSurfaces
        lines.append("    refinementSurfaces")
        lines.append("    {")
        for surf, val in cm.get("refinementSurfaces", {}).items():
            lines.append(f"        {surf} {{ level ({val['level'][0]} {val['level'][1]}); }}")
        lines.append("    }")
        # refinementRegions
        lines.append("    refinementRegions")
        lines.append("    {")
        for reg, val in cm.get("refinementRegions", {}).items():
            lines.append(f"        {reg} {{ mode {val['mode']}; levels ({val['levels'][0]} {val['levels'][1]}); }}")
        lines.append("    }")
        # Extra parameters
        lines.append("    allowFreeStandingZoneFaces true;")
        lines.append("    resolveFeatureAngle 30;")
        lines.append("};\n")

        # SnapControls
        lines.append("snapControls")
        lines.append("{")
        for k,v in self.snapControls.items():
            lines.append(f"    {k} {str(v).lower() if isinstance(v,bool) else v};")
        lines.append("};\n")

        # AddLayersControls
        lines.append("addLayersControls")
        lines.append("{")
        for k,v in self.addLayersControls.items():
            if k != "layers":
                lines.append(f"    {k} {v};")
        lines.append("    layers")
        lines.append("    {")
        for name, layer in self.addLayersControls.get("layers", {}).items():
            lines.append(f'        "{name}"')
            lines.append("        {")
            lines.append(f'            nSurfaceLayers {layer["nSurfaceLayers"]};')
            lines.append("        }")
        lines.append("    }")
        lines.append("};\n")

        # MeshQualityControls
        lines.append("meshQualityControls")
        lines.append("{")
        for k,v in self.meshQualityControls.items():
            lines.append(f"    {k} {v};")
        lines.append("};\n")

        # debug / writeFlags / mergeTolerance
        lines.append("debug 0;\n")
        lines.append("writeFlags (scalarLevels layerSets layerFields);")
        lines.append("mergeTolerance 1e-06;")

        dict_path.write_text("\n".join(lines))
        print(f"snappyHexMeshDict written to {dict_path}")


    def write(self):
        """Write all SnappyHexMesh-related files."""
        # Write snappyHexMeshDict
        self.write_snappyHexMeshDict()

        # Write surfaceFeaturesDict
        self.write_surface_features_dict()

    def run(self):
        """
        Runs snappyHexMesh in the given OpenFOAM case.
        """
        base_path = self.case_path  # <- utilise directement l'attribut de la classe
        log_file = base_path / "log.meshing"

        if not base_path.exists():
            raise FileNotFoundError(f"The case path '{base_path}' does not exist.")

        if not base_path.is_dir():
            raise NotADirectoryError(f"The case path '{base_path}' is not a directory.")

        cmd = ["snappyHexMesh", "-overwrite", "-case", str(base_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Error running snappyHexMesh:")
            print(result.stderr)
        else:
            print("snappyHexMesh finished successfully.")
            print(result.stdout)
