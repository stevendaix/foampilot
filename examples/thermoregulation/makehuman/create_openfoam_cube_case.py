from pathlib import Path
import shutil
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent
CASE = ROOT / 'openfoam_cube_case'
TRI = CASE / 'constant' / 'triSurface'
TRI.mkdir(parents=True, exist_ok=True)
for d in ['0', 'constant', 'system', 'comms']:
    (CASE / d).mkdir(parents=True, exist_ok=True)

# MakeHuman base mesh supplied by the Ubuntu package. The socket exporter can
# replace this STL with makehuman_body_only.stl without changing the CFD case.
source_npz = Path('/usr/share/makehuman-community/data/3dobjs/base.npz')
data = np.load(source_npz, allow_pickle=True)
vertices = data['coord'].astype(float)
quads = data['fvert'].astype(np.int64)
faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=0)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
mesh.remove_unreferenced_vertices()
mesh.export(TRI / 'human.stl')

# Cube is chosen from the measured MakeHuman bounds: x ±4.97, y [-8.45, 8.50],
# z [-1.10, 3.26]. The 1.0 m clearance is intentionally explicit.
(CASE / 'system' / 'blockMeshDict').write_text(r'''FoamFile
{
    format ascii;
    class dictionary;
    object blockMeshDict;
}
convertToMeters 1;
vertices
(
    (-6 -10 -3) (6 -10 -3) (6 10 -3) (-6 10 -3)
    (-6 -10 6)  (6 -10 6)  (6 10 6)  (-6 10 6)
);
blocks ((hex (0 1 2 3 4 5 6 7) (24 40 18) simpleGrading (1 1 1)));
edges ();
boundary
(
    inlet  { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((1 2 6 5)); }
    floor  { type wall; faces ((0 1 5 4)); }
    ceiling{ type wall; faces ((3 7 6 2)); }
    sideA  { type wall; faces ((0 3 2 1)); }
    sideB  { type wall; faces ((4 5 6 7)); }
);
mergePatchPairs ();
''')
(CASE / 'system' / 'snappyHexMeshDict').write_text(r'''FoamFile
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
        name human;
    }
}
castellatedMeshControls
{
    maxLocalCells 100000;
    maxGlobalCells 200000;
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
    locationInMesh (0 0 0);
    allowFreeStandingZoneFaces true;
}
snapControls
{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
}
addLayersControls { relativeSizes true; layers {}; expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls { maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-13; minTetQuality 1e-9; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }
writeFlags (scalarLevels layerSets layerFields);
mergeTolerance 1e-6;
''')
(CASE / 'system' / 'controlDict').write_text(r'''FoamFile
{
    format ascii;
    class dictionary;
    object controlDict;
}
application     foamRun;
solver          incompressibleFluid;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         10;
deltaT          1;
writeControl    timeStep;
writeInterval   5;
purgeWrite      0;
functions
{
    humanCoupling
    {
        type externalCoupled;
        libs (fieldFunctionObjects);
        commsDir "${FOAM_CASE}/comms";
        regions
        {
            region0
            {
                human
                {
                    writeFields (T);
                    readFields (T);
                }
            }
        }
        initByExternal true;
        waitInterval 1;
        timeOut 300;
        calcFrequency 1;
    }
}
''')
# Temperature field for externalCoupledTemperature must be installed after snappyHexMesh.
(CASE / 'README.md').write_text('''# Cas OpenFOAM cube autour d’un humain MakeHuman\n\nCe cas crée un domaine cubique autour du maillage humain MakeHuman. Le maillage de démonstration est converti depuis `base.npz` installé par MakeHuman Community ; il peut être remplacé par `output/makehuman_body_only.stl` exporté par le socket MakeHuman.\n\nLe cube est `[-6,6] x [-10,10] x [-3,6] m` et l’humain mesuré est approximativement `[-4.97,4.97] x [-8.45,8.50] x [-1.10,3.26] m` dans le repère MakeHuman. La surface humaine devient un patch OpenFOAM nommé `human`.\n\nCommandes :\n\n```bash\npython3 ../create_openfoam_cube_case.py\nsource /opt/openfoam13/etc/bashrc\ncd openfoam_cube_case\nblockMesh\nsurfaceFeatureExtract\nsnappyHexMesh -overwrite\ncheckMesh\n```\n\nAprès `snappyHexMesh`, installer le champ `T` avec la condition `externalCoupledTemperature` et lancer le pilote FoamPilot. Le mapping des faces OpenFOAM vers les 17 zones JOS-3 doit être calculé après génération du maillage à partir des centres de faces du patch `human`; le mapping STL triangle n’est pas encore l’ordre des faces OpenFOAM.\n''')
print(f'created {TRI / "human.stl"}: {len(mesh.vertices)} vertices, {len(mesh.faces)} triangles')
''
