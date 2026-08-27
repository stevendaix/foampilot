import pyvista as pv
from pathlib import Path
import numpy as np
import subprocess
import shutil


class SnappyMesher:
    """
    Class for configuring and generating snappyHexMeshDict based on STL geometry.

    Attributes:
    - base_path (Path): Path to the OpenFOAM case directory.
    - stl_file (Path): Path to the STL file.
    - snappy_hex_mesh_dict_path (Path): Path to the snappyHexMeshDict file.
    """

    def __init__(self, parent=None, stl_file=None, case_path=None, castellatedMesh=True, snap=True, addLayers=False):
        if parent is not None:
            self.parent = parent
            self.case_path = parent.case_path
        elif case_path is not None:
            self.parent = None
            self.case_path = Path(case_path)
        else:
            raise ValueError("Either parent with case_path or direct case_path must be provided")

        self.snappy_hex_mesh_dict_path = self.case_path / "system" / "snappyHexMeshDict"
        self.stl_file = Path(stl_file) if stl_file else None
        
        self.locationInMesh = (0.1, 0.1, 0.1)

        self.castellatedMesh = castellatedMesh
        self.snap = snap
        self.addLayers = addLayers
        
        self.geometry = {}
        if self.stl_file:
            self.add_geometry(self.stl_file.stem, self.stl_file)
        
        default_refinement = {}
        if self.stl_file:
            default_refinement[self.stl_file.stem] = {"level": (2, 3)}
        
        self.castellatedMeshControls = {
            "maxLocalCells": 100000,
            "maxGlobalCells": 2000000,
            "minRefinementCells": 10,
            "nCellsBetweenLevels": 3,
            "locationInMesh": self.locationInMesh,
            "refinementSurfaces": default_refinement,
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
            # --- REQUIRED (OF13) ---
            "relativeSizes": True,
            "nGrow": 0,

            # --- Thickness model ---
            "expansionRatio": 1.2,
            "finalLayerThickness": 0.3,
            "minThickness": 0.1,

            # --- Feature handling ---
            "featureAngle": 60,
            "slipFeatureAngle": 30,

            # --- Smoothing / relaxation ---
            "nRelaxIter": 5,
            "nSmoothNormals": 1,
            "nSmoothSurfaceNormals": 1,
            "nSmoothThickness": 10,

            # --- Quality / safety limits ---
            "maxFaceThicknessRatio": 0.5,
            "maxThicknessToMedialRatio": 0.3,
            "minMedianAxisAngle": 90,

            # --- Extrusion control ---
            "nBufferCellsNoExtrude": 0,
            "nLayerIter": 50,

            # --- Per-surface layers ---
            "layers": {}
        }

        self.meshQualityControls = {
            "maxNonOrtho": 70,
            "maxBoundarySkewness": 20,
            "maxInternalSkewness": 4,
            "maxConcave": 80,
            "minVol": -1e30,
            "minTetQuality": 1e-30,
            "minArea": -1,
            "minTwist": 0.05,
            "minDeterminant": 0.001,
            "minFaceWeight": 0.05,
            "minVolRatio": 0.01,
            "minTriangleTwist": -1,
            "minFlatness": 0.5,
            "nSmoothScale": 4,
            "errorReduction": 0.75,
        }

        self.debugFlags = []
        self.writeFlags = []
        self.insidePoint = (-0.7, 0.0, 0.0)
        self.resolveFeatureAngle = 45

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

    def import_reference_surface(self, source_surface, target_name=None):
        """Import a reference OBJ/STL surface through the Foampilot API."""
        source = Path(source_surface)
        if not source.exists():
            raise FileNotFoundError(f"Reference surface not found: {source}")
        target_name = target_name or source.name
        target = self.case_path / "constant" / "triSurface" / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".gz":
            import gzip
            with gzip.open(source, "rb") as source_stream, open(target, "wb") as target_stream:
                shutil.copyfileobj(source_stream, target_stream)
        else:
            shutil.copy2(source, target)
        self.stl_file = target
        self.geometry = {}
        self.add_geometry(target.stem, target)
        self.castellatedMeshControls["refinementSurfaces"] = {
            target.stem: {"level": (2, 3)}
        }
        return target

    def add_geometry(self, name, stl_path, geo_type="triSurfaceMesh"):
        """Add an STL/OBJ geometry to the snappyHexMesh configuration."""
        stl_file = Path(stl_path)
        self.geometry[name] = {
            "type": geo_type,
            "name": name,
            "file": stl_file.name,
            "regions": {}
        }
    # ----------------------
    # SurfaceFeaturesDict
    # ----------------------
    def write_surface_features_dict(self, stl_list_for_emesh: list[str] = None, included_angle: float = 30) -> Path:
        """
        Write system/surfaceFeaturesDict for snappyHexMesh based on a list of STL files.
        """

        if not stl_list_for_emesh:
            raise ValueError("stl_list_for_emesh must not be empty")

        system_path = self.case_path / "system"
        system_path.mkdir(parents=True, exist_ok=True)
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

        # Ici on ajoute tous les STL correctement, un par ligne
        for stl in stl_list_for_emesh:
            lines.append(f'    "{stl}"')  # <- pas de ; ici, OpenFOAM n'en veut pas

        lines.append(");")  # Fermeture de la liste
        lines.append(f"\nincludedAngle   {included_angle};\n")  # Angle inclus

        # Écriture finale
        dict_file.write_text("\n".join(lines))
        print(f"surfaceFeaturesDict written to {dict_file} with {len(stl_list_for_emesh)} STL files.")
        return dict_file



    # ----------------------
    # Utilities
    # ----------------------
    def run_surface_features(self):
        """Run the OpenFOAM surface-feature utility for this case.

        OpenFOAM 13 provides ``surfaceFeatures`` and deprecates
        ``surfaceFeatureExtract``.  The method selects the modern executable
        when available, preserves compatibility with older installations, and
        verifies that every requested feature file was actually created.
        """
        system_path = self.case_path / "system"
        system_path.mkdir(parents=True, exist_ok=True)
        dict_file = system_path / "surfaceFeaturesDict"

        if not dict_file.exists():
            stl_names = [geo.get("file", geo["name"]) for geo in self.geometry.values()]
            lines = [
                "FoamFile", "{", "    version     2.0;", "    format      ascii;",
                "    class       dictionary;", "    location    \"system\";",
                "    object      surfaceFeaturesDict;", "}", "", "surfaces", "(",
            ]
            lines.extend(f'    "{name}"' for name in stl_names)
            lines.extend([");", "", "includedAngle 60;", ""])
            dict_file.write_text("\n".join(lines), encoding="utf-8")

        executable = shutil.which("surfaceFeatures") or shutil.which("surfaceFeatureExtract")
        if executable is None:
            raise RuntimeError("Neither surfaceFeatures nor surfaceFeatureExtract is available")
        result = subprocess.run([executable, "-case", str(self.case_path)], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"{Path(executable).name} failed: {result.stderr}")

        missing = []
        for feature in self.castellatedMeshControls.get("features", []):
            feature_path = self.case_path / "constant" / "triSurface" / feature["file"]
            if not feature_path.exists():
                missing.append(str(feature_path))
        if missing:
            raise RuntimeError("Surface feature files were not created: " + ", ".join(missing))
        print(f"{Path(executable).name} finished successfully.")

    def run_surface_feature_extract(self):
        """Backward-compatible alias for :meth:`run_surface_features`."""
        self.run_surface_features()

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

    def write_block_mesh_dict(
        self,
        padding: float = 0.2,
        base_cell_size: float = None
    ):
        """
        Generate a blockMeshDict that encloses the STL geometry.

        Args:
            padding (float): Relative padding added around the STL bounding box.
                            0.2 means +20% in each direction.
            base_cell_size (float): Target cell size. If None, estimated automatically.
        """
        tri_surface_dir = self.case_path / "constant" / "triSurface"
        bounds = None
        for name, geo in self.geometry.items():
            stl_path = tri_surface_dir / geo["file"]
            if not stl_path.exists():
                continue
            mesh = pv.read(str(stl_path))
            if bounds is None:
                bounds = mesh.bounds
            else:
                bounds = (
                    min(bounds[0], mesh.bounds[0]),
                    min(bounds[1], mesh.bounds[1]),
                    min(bounds[2], mesh.bounds[2]),
                    max(bounds[3], mesh.bounds[3]),
                    max(bounds[4], mesh.bounds[4]),
                    max(bounds[5], mesh.bounds[5]),
                )

        if bounds is None:
            raise FileNotFoundError(f"No STL files found in {tri_surface_dir}")

        xmin, xmax, ymin, ymax, zmin, zmax = bounds

        # Expand bounding box
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin

        xmin -= padding * dx
        xmax += padding * dx
        ymin -= padding * dy
        ymax += padding * dy
        zmin -= padding * dz
        zmax += padding * dz

        # Automatic cell size estimation
        if base_cell_size is None:
            base_cell_size = min(dx, dy, dz) / 20.0

        nx = max(1, int((xmax - xmin) / base_cell_size))
        ny = max(1, int((ymax - ymin) / base_cell_size))
        nz = max(1, int((zmax - zmin) / base_cell_size))

        system_path = self.case_path / "system"
        system_path.mkdir(parents=True, exist_ok=True)
        dict_path = system_path / "blockMeshDict"

        lines = [
            "FoamFile",
            "{",
            "    version     2.0;",
            "    format      ascii;",
            "    class       dictionary;",
            "    location    \"system\";",
            "    object      blockMeshDict;",
            "}",
            "",
            "convertToMeters 1;",
            "",
            "vertices",
            "(",
            f"    ({xmin} {ymin} {zmin})",
            f"    ({xmax} {ymin} {zmin})",
            f"    ({xmax} {ymax} {zmin})",
            f"    ({xmin} {ymax} {zmin})",
            f"    ({xmin} {ymin} {zmax})",
            f"    ({xmax} {ymin} {zmax})",
            f"    ({xmax} {ymax} {zmax})",
            f"    ({xmin} {ymax} {zmax})",
            ");",
            "",
            "blocks",
            "(",
            f"    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)",
            ");",
            "",
            "edges ();",
            "",
            "boundary",
            "(",
            "    domain",
            "    {",
            "        type patch;",
            "        faces",
            "        (",
            "            (0 1 2 3)",
            "            (4 5 6 7)",
            "            (0 1 5 4)",
            "            (1 2 6 5)",
            "            (2 3 7 6)",
            "            (3 0 4 7)",
            "        );",
            "    }",
            ");",
            "",
            "mergePatchPairs ();"
        ]

        dict_path.write_text("\n".join(lines))
        print(f"blockMeshDict written to {dict_path}")

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
        lines.append(f"    resolveFeatureAngle {self.resolveFeatureAngle};")
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

        for k, v in self.addLayersControls.items():
            if k != "layers":
                if isinstance(v, bool):
                    v_str = str(v).lower()
                else:
                    v_str = v
                lines.append(f"    {k} {v_str};")

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

        dict_path.write_text("\n".join(lines))
        print(f"snappyHexMeshDict written to {dict_path}")

        # Write separate meshQualityDict for cases that include it explicitly
        self.write_mesh_quality_dict()

    def write_mesh_quality_dict(self):
        """Write system/meshQualityDict using the current meshQualityControls."""
        system_path = self.case_path / "system"
        system_path.mkdir(parents=True, exist_ok=True)
        dict_path = system_path / "meshQualityDict"
        lines = [
            "/*--------------------------------*- C++ -*----------------------------------*\\",
            "  =========                 |",
            "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox",
            "   \\\\    /   O peration     | Website:  https://openfoam.org",
            "    \\\\  /    A nd           | Version:  13",
            "     \\\\/     M anipulation  |",
            "\\*---------------------------------------------------------------------------*/",
            "FoamFile",
            "{",
            "    format      ascii;",
            "    class       dictionary;",
            '    location    "system";',
            "    object      meshQualityDict;",
            "}",
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //",
            "",
            "//- Maximum non-orthogonality allowed. Set to 180 to disable.",
            f"maxNonOrtho {self.meshQualityControls.get('maxNonOrtho', 70)};",
            "",
            "//- Max skewness allowed. Set to <0 to disable.",
            f"maxBoundarySkewness {self.meshQualityControls.get('maxBoundarySkewness', 20)};",
            f"maxInternalSkewness {self.meshQualityControls.get('maxInternalSkewness', 4)};",
            "",
            "//- Max concaveness allowed. Is angle (in degrees) below which concavity",
            "//  is allowed. 0 is straight face, <0 would be convex face.",
            "//  Set to 180 to disable.",
            f"maxConcave {self.meshQualityControls.get('maxConcave', 80)};",
            "",
            "//- Minimum cell pyramid volume relative to min bounding box length^3",
            "//  Set to a fraction of the smallest cell volume expected.",
            "//  Set to very negative number (e.g. -1e30) to disable.",
            f"minVol {self.meshQualityControls.get('minVol', 1e-13)};",
            "",
            "//- Minimum quality of the tet formed by the face-centre",
            "//  and variable base point minimum decomposition triangles and",
            "//  the cell centre.  Set to very negative number (e.g. -1e30) to",
            "//  disable.",
            "//     <0 = inside out tet,",
            "//      0 = flat tet",
            "//      1 = regular tet",
            f"minTetQuality {self.meshQualityControls.get('minTetQuality', 1e-15)};",
            "",
            "//- Minimum face twist. Set to <-1 to disable. dot product of face normal",
            "//  and face centre triangles normal",
            f"minTwist {self.meshQualityControls.get('minTwist', 0.02)};",
            "",
            "//- Minimum normalised cell determinant",
            "//  1 = hex, <= 0 = folded or flattened illegal cell",
            f"minDeterminant {self.meshQualityControls.get('minDeterminant', 0.001)};",
            "",
            "//- minFaceWeight (0 -> 0.5)",
            f"minFaceWeight {self.meshQualityControls.get('minFaceWeight', 0.05)};",
            "",
            "//- minVolRatio (0 -> 1)",
            f"minVolRatio {self.meshQualityControls.get('minVolRatio', 0.01)};",
            "",
            "// Advanced",
            "",
            "//- Number of error distribution iterations",
            f"nSmoothScale {self.meshQualityControls.get('nSmoothScale', 4)};",
            "//- Amount to scale back displacement at error points",
            f"errorReduction {self.meshQualityControls.get('errorReduction', 0.75)};",
            "",
            "// Optional : some meshing phases allow usage of relaxed rules.",
            "// See e.g. addLayersControls::nRelaxedIter.",
            "relaxed",
            "{",
            "    //- Maximum non-orthogonality allowed. Set to 180 to disable.",
            f"    maxNonOrtho {self.meshQualityControls.get('maxNonOrtho', 75)};",
            "}",
            "",
            "// ************************************************************************* //",
        ]
        dict_path.write_text("\n".join(lines))
        print(f"meshQualityDict written to {dict_path}")


    def write(self):
        """Write all SnappyHexMesh-related files."""
        # Write snappyHexMeshDict
        self.write_snappyHexMeshDict()

    def run_block_mesh(self):
        """
        Runs blockMesh for the case.
        """
        cmd = ["blockMesh", "-case", str(self.case_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("Error running blockMesh:")
            print(result.stderr)
            raise RuntimeError("blockMesh failed")
        else:
            print("blockMesh finished successfully.")

    def run(self):
        """
        Full meshing pipeline:
        1. blockMesh
        2. surfaceFeatureExtract
        3. snappyHexMesh
        """

        if not self.case_path.exists():
            raise FileNotFoundError(f"Case path '{self.case_path}' does not exist.")

        # 1. blockMesh
        
        self.run_block_mesh()

        # 2. surfaceFeatureExtract
        self.run_surface_feature_extract()

        # 3. snappyHexMesh
        cmd = ["snappyHexMesh", "-overwrite", "-case", str(self.case_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("Error running snappyHexMesh:")
            print(result.stderr)
            raise RuntimeError("snappyHexMesh failed")
        else:
            print("snappyHexMesh finished successfully.")
