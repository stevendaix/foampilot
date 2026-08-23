import os
from pathlib import Path
import json
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent
CASE = ROOT / "openfoam_cube_case"
TRI = CASE / "constant" / "triSurface"
for directory in (CASE / "0", CASE / "constant", CASE / "system", CASE / "comms", TRI):
    directory.mkdir(parents=True, exist_ok=True)

source_npz = Path(os.getenv("MAKEHUMAN_BASE_NPZ", "/usr/share/makehuman-community/data/3dobjs/base.npz"))
data = np.load(source_npz, allow_pickle=True)
vertices = 0.1 * np.asarray(data["coord"], dtype=float)
all_raw_faces = np.asarray(data["fvert"], dtype=np.int64)
all_groups = np.asarray(data["group"], dtype=np.int64)
body_mask = all_groups == 0
raw_faces = all_raw_faces[body_mask]
triangles = []
for raw_face in raw_faces:
    face = raw_face[:4]
    face = face[face >= 0]
    if len(face) == 3:
        candidates = [face]
    elif len(face) == 4:
        candidates = [face[[0, 1, 2]], face[[0, 2, 3]]]
    else:
        candidates = []
    for tri in candidates:
        if len(set(map(int, tri))) == 3 and np.all(tri < len(vertices)):
            triangles.append(tri)
faces = np.asarray(triangles, dtype=np.int64)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
mesh.merge_vertices(digits_vertex=8)
if hasattr(mesh, "nondegenerate_faces"):
    mesh.update_faces(mesh.nondegenerate_faces())
mesh.remove_unreferenced_vertices()
trimesh.repair.fix_winding(mesh)
mesh.export(TRI / "human.stl")
quality = {
    "source": str(source_npz),
    "raw_faces": int(len(raw_faces)),
    "raw_faces_all_groups": int(len(all_raw_faces)),
    "selected_group_id": 0,
    "exported_triangles": int(len(mesh.faces)),
    "vertices": int(len(mesh.vertices)),
    "area_m2": float(mesh.area),
    "volume_m3": float(mesh.volume),
    "watertight": bool(mesh.is_watertight),
    "winding_consistent": bool(mesh.is_winding_consistent),
    "components": int(len(mesh.split(only_watertight=False))),
}
(TRI / "human_stl_quality.json").write_text(json.dumps(quality, indent=2) + "\\n", encoding="utf-8")
if not mesh.is_watertight:
    print("WARNING: human.stl is not watertight; snappyHexMesh volume subtraction is not validated", flush=True)

# The MakeHuman vertical axis is y. The cube leaves approximately 0.25 m on
# the lateral/depth directions and 0.3 m above/below the body.
(CASE / "system" / "blockMeshDict").write_text(r'''FoamFile
{
    format ascii;
    class dictionary;
    object blockMeshDict;
}
convertToMeters 1;
vertices
(
    (-0.75 -1.10 -0.40) (0.75 -1.10 -0.40) (0.75 1.10 -0.40) (-0.75 1.10 -0.40)
    (-0.75 -1.10 0.60)  (0.75 -1.10 0.60)  (0.75 1.10 0.60)  (-0.75 1.10 0.60)
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (30 44 20) simpleGrading (1 1 1)
);
edges ();
boundary
(
    inlet   { type patch; faces ((0 4 7 3)); }
    outlet  { type patch; faces ((1 2 6 5)); }
    floor   { type wall; faces ((0 1 5 4)); }
    ceiling { type wall; faces ((3 7 6 2)); }
    sideA   { type wall; faces ((0 3 2 1)); }
    sideB   { type wall; faces ((4 5 6 7)); }
);
mergePatchPairs ();
''')

(CASE / "system" / "snappyHexMeshDict").write_text(r'''FoamFile
{
    format ascii;
    class dictionary;
    object snappyHexMeshDict;
}
castellatedMesh true;
snap true;
addLayers false;
geometry
{
    human.stl
    {
        type triSurfaceMesh;
        file "human.stl";
        name human;
    }
}
castellatedMeshControls
{
    maxLocalCells 200000;
    maxGlobalCells 400000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 2;
    features ();
    refinementSurfaces
    {
        human { level (2 2); patchInfo { type wall; } }
    }
    resolveFeatureAngle 30;
    refinementRegions { }
    locationInMesh (0.0 -1.0 0.0);
    allowFreeStandingZoneFaces true;
}
snapControls
{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
}
addLayersControls
{
    relativeSizes true;
    layers { }
    expansionRatio 1.0;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}
meshQualityControls
{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-9;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.02;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}
writeFlags (scalarLevels layerSets layerFields);
mergeTolerance 1e-6;
''')

(CASE / "system" / "controlDict").write_text(r'''FoamFile
{
    format ascii;
    class dictionary;
    object controlDict;
}
solver          fluid;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         0.2;
deltaT          0.05;
writeControl    timeStep;
writeInterval   20;
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
runTimeModifiable true;
''')

(CASE / "README.md").write_text("""# Cas OpenFOAM 13 de thermorégulation autour d’un humain MakeHuman

Ce cas utilise le maillage de base MakeHuman comme géométrie humaine, le redimensionne par `0.1` afin d’obtenir une taille d’environ `1.7 m`, puis crée un domaine d’air cubique autour du corps. Cette correction d’échelle est indispensable : le maillage brut MakeHuman mesure environ `17` unités sur son axe vertical et produirait une aire corporelle irréaliste si cette conversion n’était pas appliquée.

Le domaine est `x ∈ [-0.75, 0.75] m`, `y ∈ [-1.10, 1.10] m`, `z ∈ [-0.40, 0.60] m`. Le patch humain s’appelle `human`. Le maillage est destiné à une validation de chaîne géométrique et de transfert thermique ; il ne prétend pas encore reproduire une soufflerie ou une chambre climatique expérimentale.

## Génération et maillage

```bash
cd examples/thermoregulation/makehuman
python3 create_openfoam_cube_case.py
cd openfoam_cube_case
source /opt/openfoam13/etc/bashrc
./Allrun
```

`Allrun` exécute `blockMesh`, `snappyHexMesh`, `createExternalCoupledPatchGeometry`, le mapping des faces et `checkMesh`. Le fichier `zone_mapping_openfoam.csv` est construit à partir des centres de faces du patch OpenFOAM réel ; il ne réutilise pas directement les identifiants des triangles STL. La configuration CFD humaine est laminaire et utilise `fixedFluxPressure` sur les frontières de pression.

## Validation de référence

La validation du solveur est séparée de la géométrie humaine. Le script `examples/thermoregulation/validation/run_openfoam13_references.py` exécute `buoyantCavity` et `multiRegion/CHT/coolingSphere`; le rapport est écrit dans `examples/thermoregulation/validation/results/openfoam13_reference_report.md`.

Le cas humain réutilise ensuite la même logique de vérification : `checkMesh`, contrôle de l’aire du patch, mapping par face, contrôle du nombre de faces et échange `data.out/data.in` par `externalCoupledTemperature`. Le STL doit contenir uniquement le groupe MakeHuman `body` (groupe 0), et non les groupes `joint-*` ou `helper-*`. Pour la convection naturelle, la formulation Boussinesq de `constant/physicalProperties` est recommandée ; elle a été testée avec succès sur la configuration ouverte body-only jusqu’à `0.2 s` avec le couplage JOS-3.

## Limite physique actuelle

Pour une validation expérimentale du corps humain, il faudra ajouter des conditions d’entrée documentées, une vitesse ou une convection naturelle définie, des propriétés de l’air dépendantes de la température et un modèle de rayonnement. Les 17 zones JOS-3 ne sont pas déduites automatiquement par OpenFOAM : elles sont attribuées face par face par `map_openfoam_human_faces.py`.

[1]: https://doc.cfd.direct/openfoam/user-guide-v13/case-management "OpenFOAM User Guide v13"
""")

print(f"created {TRI / 'human.stl'}: {len(mesh.vertices)} vertices, {len(mesh.faces)} triangles")
print("bounds:", mesh.bounds.tolist())
print("surface_area_m2:", float(mesh.area))
