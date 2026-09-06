from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parent
TRI = ROOT / 'constant' / 'triSurface'
TRI.mkdir(parents=True, exist_ok=True)
parser = ArgumentParser(description='Create the medical OpenFOAM snappy case from a patch-labelled VTP surface.')
parser.add_argument('--input', type=Path, required=True, help='patch-labelled VTP input surface')
args = parser.parse_args()
SRC = args.input.expanduser().resolve()
if not SRC.is_file():
    parser.error(f'input surface not found: {SRC}')
OUT = TRI / 'aorta_multiregion.stl'
NAMES = {0:'outlet_0',1:'outlet_1',2:'outlet_2',3:'outlet_3',4:'inlet',5:'outlet_5',6:'outlet_6',7:'outlet_7',8:'outlet_8',9:'wall'}
m = pv.read(SRC).triangulate()
f = m.faces.reshape(-1, 4)[:, 1:]
ids = np.asarray(m.cell_data['PatchId'])
keys = [tuple(sorted(map(int, t))) for t in f]
counts = {k: keys.count(k) for k in set(keys)}
def normal(t):
    n = np.cross(t[1]-t[0], t[2]-t[0]); q = np.linalg.norm(n)
    return n/q if q else np.zeros(3)
with OUT.open('w') as h:
    for pid in sorted(set(ids.tolist())):
        name = NAMES.get(int(pid), f'patch{pid}')
        h.write(f'solid {name}\n')
        for tri in f[ids == pid]:
            key = tuple(sorted(map(int, tri)))
            if counts.get(key, 1) > 1:
                continue
            t = np.asarray(m.points[tri]); n = normal(t)
            h.write(f' facet normal {n[0]:.9g} {n[1]:.9g} {n[2]:.9g}\n  outer loop\n')
            for p in t: h.write(f'   vertex {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n')
            h.write('  endloop\n endfacet\n')
        h.write(f'endsolid {name}\n')

(ROOT/'system').mkdir(exist_ok=True)
(ROOT/'system'/'blockMeshDict').write_text('''FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }\nconvertToMeters 1;\nvertices ( (-145 -35 -10) (-25 -35 -10) (-25 285 -10) (-145 285 -10) (-145 -35 60) (-25 -35 60) (-25 285 60) (-145 285 60) );\nblocks ( hex (0 1 2 3 4 5 6 7) (24 64 14) simpleGrading (1 1 1) );\nedges ();\nboundary ( outer { type patch; faces ( (0 1 5 4) (1 2 6 5) (2 3 7 6) (3 0 4 7) (4 5 6 7) (3 2 1 0) ); } );\nmergePatchPairs ();\n''')
(ROOT/'system'/'controlDict').write_text('''FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\napplication snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 6; runTimeModifiable true;\n''')
(ROOT/'system'/'snappyHexMeshDict').write_text('''FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }\ncastellatedMesh true; snap true; addLayers false;\ngeometry { aorta_surface { type triSurfaceMesh; file "aorta_multiregion.stl"; } }\ncastellatedMeshControls { maxLocalCells 100000; maxGlobalCells 500000; minRefinementCells 10; maxLoadUnbalance 0.10; nCellsBetweenLevels 2; features (); refinementSurfaces { aorta_surface { level (2 3); regions { inlet { level (3 3); patchInfo { type patch; } } outlet_0 { level (3 3); patchInfo { type patch; } } outlet_1 { level (3 3); patchInfo { type patch; } } outlet_2 { level (3 3); patchInfo { type patch; } } outlet_3 { level (3 3); patchInfo { type patch; } } outlet_5 { level (3 3); patchInfo { type patch; } } outlet_6 { level (3 3); patchInfo { type patch; } } outlet_7 { level (3 3); patchInfo { type patch; } } outlet_8 { level (3 3); patchInfo { type patch; } } wall { level (2 2); patchInfo { type wall; } } } } } resolveFeatureAngle 30; refinementRegions {}; locationInMesh (-100 100 25); allowFreeStandingZoneFaces true; }\nsnapControls { nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5; nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }\naddLayersControls { relativeSizes true; layers {}; }\nmeshQualityControls { maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-13; minTetQuality 1e-9; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; errorReduction 0.75; nSmoothScale 4; }\ndebug 0; mergeTolerance 1e-6;\n''')
print(OUT)
