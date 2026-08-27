#!/usr/bin/env python3
"""Build the reproducible Turning35 Foundation 13 FoamPilot case.

The geometry is intentionally kept in the case directory.  The generator only
writes solver dictionaries and runners; it never copies OpenFOAM.com files or
requires maneuveringLib at runtime.
"""
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "FoamPilotCases" / "Turning35Foundation13"

HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website: https://openfoam.org
    \\\\  /    A nd           | Version: 13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
"""

CONTROL = HEADER + r'''FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}
application     marineFoam;
solver          incompressibleVoF;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         0.20;
deltaT          0.001;
writeControl    adjustableRunTime;
writeInterval   0.01;
purgeWrite      0;
writeFormat     ascii;
writePrecision  10;
writeCompression off;
timeFormat      general;
timePrecision   8;
runTimeModifiable yes;
adjustTimeStep  yes;
maxCo           0.5;
maxAlphaCo      0.25;
maxDeltaT       0.01;
'''

MARINE_PROPERTIES = '''FoamFile
{
    format ascii;
    class dictionary;
    object marineProperties;
}
mode turning35;
solver incompressibleVoF;
meshBackend snappyHexMesh;
fluid waterAir;
referenceSpeed 1.3381;
turningAngle 35;
turningStartTime 0.05;
'''

PHASE_PROPERTIES = HEADER + '''FoamFile
{
    format      ascii;
    class       dictionary;
    location    "constant";
    object      phaseProperties;
}
phases          (water air);
sigma           0.072;
'''

DYNAMIC_MESH = HEADER + '''FoamFile
{
    format ascii;
    class dictionary;
    object dynamicMeshDict;
}
mover
{
    type motionSolver;
    libs ("librigidBodyMeshMotion.so");
    motionSolver rigidBodyMotion;
    report on;
    solver { type Newmark; }
    accelerationRelaxation 0.4;
    bodies
    {
        hull
        {
            type rigidBody;
            parent root;
            centreOfMass (0 0 0);
            mass 412.73;
            inertia (40 0 0 921 0 921);
            transform (1 0 0 0 1 0 0 0 1) (2.929541 0 0.2);
            joint
            {
                type composite;
                joints
                (
                    { type Px; }
                    { type Py; }
                    { type Pz; }
                    { type Rx; }
                    { type Ry; }
                    { type Rz; }
                );
            }
            patches (hull rudder);
            innerDistance 0.3;
            outerDistance 1.0;
        }
    }
    restraints
    {
        translationDamper
        {
            type linearDamper;
            body hull;
            coeff 8596;
        }
        rotationDamper
        {
            type sphericalAngularDamper;
            body hull;
            coeff 11586;
        }
    }
}
'''

FVMODELS = HEADER + '''FoamFile
{
    format ascii;
    class dictionary;
    location "constant";
    object fvModels;
}
disk1
{
    type actuationDisk;
    cellZone rotor;
    diskArea 0.0068;
    diskDir (-1 0 0);
    Cp 0.18;
    Ct 0.32;
    upstreamPoint (2.65275 0 -0.078651);
    downstreamPoint (2.67172 0 -0.078651);
}
'''

SETFIELDS = HEADER + '''FoamFile
{
    format ascii;
    class dictionary;
    location "system";
    object setFieldsDict;
}
defaultFieldValues
(
    volScalarFieldValue alpha.water 0
);
regions
(
    boxToCell
    {
        box (-1 -0.5 -0.35) (3.5 0.5 0.0);
        fieldValues (volScalarFieldValue alpha.water 1);
    }
);
'''

TOPOSET = HEADER + '''FoamFile
{
    format ascii;
    class dictionary;
    location "system";
    object topoSetDict;
}
actions
(
    {
        name rotor;
        type cellZoneSet;
        action new;
        source boxToCell;
        box (2.55 -0.12 -0.12) (2.78 0.12 0.12);
    }
);
'''

FORCES = HEADER + '''FoamFile
{
    format ascii;
    class dictionary;
    location "system";
    object functions;
}
rigidBodyForces
{
    type rigidBodyForces;
    libs ("librigidBodyForces.so");
    body hull;
    patches (hull rudder);
    log on;
    writeControl timeStep;
    writeInterval 1;
}
forces
{
    type forces;
    libs ("libforces.so");
    patches (hull rudder);
    rho rho.water;
    CofR (0 0 0);
    writeControl timeStep;
    writeInterval 1;
}
'''

MESH = HEADER + '''FoamFile
{
    format ascii;
    class dictionary;
    location "system";
    object snappyHexMeshDict;
}
castellatedMesh true;
snap true;
addLayers false;
geometry
{
    hull.stl { type triSurfaceMesh; name hull; }
    rudder.stl { type triSurfaceMesh; name rudder; }
    BoxRefinement { type searchableBox; min (-0.85 -0.45 -0.3); max (3.25 0.45 0.3); }
}
castellatedMeshControls
{
    maxLocalCells 2000000;
    maxGlobalCells 4000000;
    minRefinementCells 10;
    nCellsBetweenLevels 3;
    features ();
    refinementSurfaces
    {
        hull { level (1 2); patchInfo { type wall; } }
        rudder { level (1 2); patchInfo { type wall; } }
    }
    refinementRegions
    {
        BoxRefinement { mode inside; levels ((1 1)); }
    }
    locationInMesh (0 0 0.0215);
    allowFreeStandingZoneFaces true;
}
snapControls
{
    nSmoothPatch 3;
    tolerance 2;
    nSolveIter 30;
    nRelaxIter 5;
}
meshQualityControls
{
    #include "meshQualityDict"
}
mergeTolerance 1e-8;
'''

BLOCK = HEADER + '''FoamFile
{
    format ascii;
    class dictionary;
    object blockMeshDict;
}
convertToMeters 1;
vertices
(
    (-1 -0.5 -0.35) (3.5 -0.5 -0.35) (3.5 0.5 -0.35) (-1 0.5 -0.35)
    (-1 -0.5 0.35) (3.5 -0.5 0.35) (3.5 0.5 0.35) (-1 0.5 0.35)
);
blocks ((hex (0 1 2 3 4 5 6 7) (90 20 14) simpleGrading (1 1 1)));
edges ();
boundary
(
    inlet { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((1 2 6 5)); }
    side { type symmetryPlane; faces ((0 1 5 4) (3 7 6 2)); }
    bottom { type wall; faces ((0 3 2 1)); }
    atmosphere { type patch; faces ((4 5 6 7)); }
);
mergePatchPairs ();
'''

MESH_QUALITY = '''FoamFile
{
    format ascii;
    class dictionary;
    object meshQualityDict;
}
maxNonOrtho 70;
maxBoundarySkewness 20;
maxInternalSkewness 4;
maxConcave 80;
minVol 1e-13;
minTetQuality 1e-15;
minArea 1e-13;
minTwist 0.05;
minDeterminant 1e-6;
minFaceWeight 0.02;
minVolRatio 0.01;
minTriangleTwist -1;
'''

ALLMESH = '''#!/bin/sh
set -eu
cd "${0%/*}"
: "${WM_PROJECT_DIR:?source OpenFOAM Foundation 13 first}"
. "$WM_PROJECT_DIR/bin/tools/RunFunctions"
runApplication blockMesh
runApplication surfaceFeatures 2>/dev/null || true
runApplication snappyHexMesh -overwrite
runApplication topoSet
runApplication checkMesh
'''

ALLRUN = '''#!/bin/sh
set -eu
cd "${0%/*}"
: "${WM_PROJECT_DIR:?source OpenFOAM Foundation 13 first}"
. "$WM_PROJECT_DIR/bin/tools/RunFunctions"
if [ ! -d constant/polyMesh ]; then ./Allmesh.FoamPilot; fi
runApplication setFields
runApplication marineFoam -solver incompressibleVoF
runApplication postProcess -func rigidBodyForces
'''

ALLCLEAN = '''#!/bin/sh
set -eu
cd "${0%/*}"
rm -rf constant/polyMesh [0-9]* processor* postProcessing log.*
'''

README = '''# Turning35Foundation13

Cas de manœuvre marine construit pour **OpenFOAM Foundation 13** et `marineFoam`.

Le cas regroupe une géométrie hull/rudder, un mouvement rigide 6-DoF du navire,
un modèle de propulsion `actuationDiskSource`, une surface libre eau/air et les
sorties de forces et moments. Il s’agit d’abord d’un cas de validation courte et
reproductible ; les valeurs hydrodynamiques finales nécessitent une étude de
convergence en temps et en maillage.

## Reproduction

```sh
source /opt/openfoam13/etc/bashrc
python3 ../../build_turning35_foampilot.py
./Allmesh.FoamPilot
./Allrun
```

`Allclean` supprime uniquement les maillages, temps calculés, processeurs et
sorties de post-traitement. Aucun maillage généré n’est versionné.

## Donor/receveur

Ce cas est actuellement une validation mono-région du mouvement et de la
propulsion. Le couplage overset/inter-mailles est validé séparément par les
harnesses `marineInterMesh*` et le cas DTC multi-région ; il ne doit pas être
présenté ici comme un overset natif complet tant que la conservation de flux et
la classification hole/fringe n’ont pas été validées sur Turning35.
'''


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    if not (TARGET / "constant/geometry/hull.stl").is_file():
        raise SystemExit("missing constant/geometry/hull.stl")
    if not (TARGET / "constant/geometry/rudder.stl").is_file():
        raise SystemExit("missing constant/geometry/rudder.stl")
    write(TARGET / "system/controlDict", CONTROL)
    write(TARGET / "system/functions", FORCES)
    write(TARGET / "system/setFieldsDict", SETFIELDS)
    write(TARGET / "system/topoSetDict", TOPOSET)
    write(TARGET / "system/snappyHexMeshDict", MESH)
    write(TARGET / "system/blockMeshDict", BLOCK)
    write(TARGET / "system/meshQualityDict", MESH_QUALITY)
    write(TARGET / "constant/marineProperties", MARINE_PROPERTIES)
    write(TARGET / "constant/phaseProperties", PHASE_PROPERTIES)
    write(TARGET / "constant/dynamicMeshDict", DYNAMIC_MESH)
    write(TARGET / "constant/fvModels", FVMODELS)
    write(TARGET / "Allmesh.FoamPilot", ALLMESH)
    write(TARGET / "Allrun", ALLRUN)
    write(TARGET / "Allclean", ALLCLEAN)
    write(TARGET / "README.md", README)
    for script in (TARGET / "Allmesh.FoamPilot", TARGET / "Allrun", TARGET / "Allclean"):
        script.chmod(0o755)
    print(f"created {TARGET}")


if __name__ == "__main__":
    main()
