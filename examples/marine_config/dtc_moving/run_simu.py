#!/usr/bin/env python3
"""OpenFOAM 13 DTC moving overset simulation.

Recrée le cas de référence openfoam13-marine-smoke/DTCHullMoving
sans le copier, en utilisant les fonctions générales de foampilot.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from foampilot import Solver, ValueWithUnit
from foampilot.base import CaseBuilder, Meshing
from foampilot.mesh import write_dynamic_mesh_dict
from foampilot.constant import (
    MomentumTransportFile,
)
from foampilot.constant.gravityFile import GravityFile
from foampilot.constant.hRefFile import HRefFile
from foampilot.system.controlDictFile import ControlDictFile
from foampilot.system.decomposeParDictFile import DecomposeParDictFile

CASE_ROOT = Path(__file__).parent.resolve()
MESH_SOURCE = Path("/home/steven/foampilot/openfoam13-marine-smoke/DTCHull")
REFERENCE_CASE = Path("/home/steven/foampilot/openfoam13-marine-smoke/DTCHullMoving")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DTC moving overset OpenFOAM 13 case.")
    parser.add_argument("--processors", type=int, default=1, help="MPI ranks.")
    parser.add_argument("--execute", action="store_true", help="Run the simulation.")
    parser.add_argument("--mesh-source", type=Path, default=MESH_SOURCE, help="Pre-meshed hull directory.")
    return parser.parse_args()


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def build_case(mesh_source: Path) -> Path:
    case_path = CASE_ROOT / "dtcmoving"
    if case_path.exists():
        shutil.rmtree(case_path)
    case_path.mkdir(parents=True)

    # 1. Structure de cas
    CaseBuilder(case_path).ensure_dirs(extra_dirs=("geometry",))

    # 2. Maillage snappy + copie du polyMesh de référence
    mesh = Meshing(case_path, mesher="snappy")
    mesh.mesher.stl_file = mesh_source / "constant" / "geometry" / "DTC-scaled.stl.gz"
    mesh.mesher.addLayers = True
    mesh.mesher.locationInMesh = (0.1, 0.1, 0.1)
    mesh.mesher.resolveFeatureAngle = 45
    mesh.mesher.insidePoint = (-0.7, 0.0, 0.0)
    mesh.mesher.castellatedMeshControls["refinementSurfaces"] = {
        "hull": {"level": (0, 0)}
    }
    mesh.mesher.write()

    poly_src = mesh_source / "constant" / "polyMesh"
    poly_dst = case_path / "constant" / "polyMesh"
    if poly_src.exists():
        if poly_dst.exists():
            shutil.rmtree(poly_dst)
        shutil.copytree(poly_src, poly_dst)

    # 3. dynamicMeshDict via helper général
    write_dynamic_mesh_dict(case_path)

    # 4. constant/ via Solver + ConstantDirectory
    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = True
    solver.is_vof = True
    solver.transient = True
    solver.turbulence_model = "kOmegaSST"
    solver.with_moving_mesh = True

    # VoF : phaseProperties + physicalProperties.* + momentumTransport
    solver.constant.configure_vof(
        phases=["water", "air"],
        sigma=0,
        phase_properties={
            "water": {"nu": 1.09e-06, "rho": 998.8},
            "air": {"nu": 1.48e-05, "rho": 1.0},
        },
    )
    solver.constant.write()

    # constant/g et constant/hRef via helpers dédiés
    GravityFile(parent=solver, value=ValueWithUnit((0, 0, -9.81), "m/s^2")).write()
    HRefFile(value=ValueWithUnit(0.244, "m")).write(case_path / "constant" / "hRef")

    # Corriger momentumTransport pour kOmegaSST (le helper ConstantDirectory
    # écrit kEpsilon par défaut pour les cas VoF)
    MomentumTransportFile(
        parent=solver,
        simulationType="RAS",
        RASModel="kOmegaSST",
    ).write(case_path / "constant" / "momentumTransport")

    # 5. system/ via SystemDirectory + fichiers spécifiques
    system_path = case_path / "system"
    system_path.mkdir(parents=True, exist_ok=True)

    # controlDict / fvSchemes / fvSolution / decomposeParDict
    control_dict = ControlDictFile(
        parent=solver,
        application="incompressibleVoF",
        startFrom="startTime",
        startTime=0,
        stopAt="endTime",
        endTime=50,
        deltaT=0.0001,
        writeControl="adjustableRunTime",
        writeInterval=5,
        purgeWrite=0,
        writeFormat="binary",
        writePrecision=6,
        writeCompression="off",
        timeFormat="general",
        timePrecision=6,
        runTimeModifiable="yes",
    )
    control_dict.attributes["adjustTimeStep"] = "yes"
    control_dict.attributes["maxCo"] = 25
    control_dict.attributes["maxAlphaCo"] = 15
    control_dict.attributes["maxDeltaT"] = 0.01
    control_dict.use_solver_keyword = True
    control_dict.write(system_path / "controlDict")

    write_file(system_path / "fvSchemes", """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
    limitedGrad     cellLimited Gauss linear 1;
}

divSchemes
{
    div(rhoPhi,U)   Gauss linearUpwind grad(U);
    div(phi,alpha)  Gauss interfaceCompression vanLeer 1;
    div(phi,k)      Gauss linearUpwind limitedGrad;
    div(phi,omega)  Gauss linearUpwind limitedGrad;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

wallDist
{
    method meshWave;
}


// ************************************************************************* //
""")

    write_file(system_path / "fvSolution", """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{
    "alpha.water.*"
    {
        nCorrectors     3;
        nSubCycles      1;

        MULESCorr       yes;
        alphaApplyPrevCorr  yes;

        MULES
        {
            nIter            1;
        }

        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-10;
        relTol          0;
        minIter         1;
    }

    "pcorr.*"
    {
        solver          GAMG;
        smoother        DIC;

        tolerance       1e-3;
        relTol          0;
    };

    p_rgh
    {
        solver          GAMG;
        smoother        DIC;

        tolerance       5e-8;
        relTol          0;
    };

    p_rghFinal
    {
        $p_rgh;
        relTol          0;
    }

    "(U|k|omega).*"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;

        nSweeps         1;
        tolerance       1e-7;
        relTol          0;
        minIter         1;
    };
}

PIMPLE
{
    momentumPredictor no;

    nOuterCorrectors 3;
    nCorrectors      1;
    nNonOrthogonalCorrectors 0;

    correctPhi      yes;
    moveMeshOuterCorrectors yes;
    transportCorrectionFinal yes;
}

relaxationFactors
{
    equations
    {
        ".*" 1;
    }
}

cache
{
    grad(U);
}


// ************************************************************************* //
""")
    DecomposeParDictFile(parent=solver, nb_proc=8).write(system_path / "decomposeParDict")

    # functions : écriture manuelle (rigidBodyForces, pas couvert par le helper)
    write_file(system_path / "functions", """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      functions;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

rigidBodyForces
{
    type            rigidBodyForces;
    libs            ("librigidBodyForces.so");
    body            hull;
    patches         (hull);
    log             on;
    writeControl    timeStep;
    writeInterval   1;
}

// ************************************************************************* //
""")

    # setFieldsDict : écriture manuelle pour conserver le format de la référence
    write_file(system_path / "setFieldsDict", """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  dev
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      setFieldsDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defaultValues
{
    alpha.water 0;
}

zones
{
    cells
    {
        type        box;

        box         (-999 -999 -999) (999 999 0.244);

        values
        {
            alpha.water 1;
        }
    }
}

extrapolatePatches
{
    "inlet|outlet"   (alpha.water);
}

// ************************************************************************* //
""")

    # refineMeshDict : écriture manuelle pour conserver le format de la référence
    write_file(system_path / "refineMeshDict", """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      refineMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

coordinates
{
    type        global;

    e1          (1 0 0);
    e2          (0 1 0);

    directions  (e1 e2);
}

zones
{
    level1
    {
        type    box;
        box     (-10 -6 -3) (10 0 3);
    }

    level2
    {
        type    box;
        box     (-5 -3 -2.5) (9 0 2);
    }

    level3
    {
        type    box;
        box     (-3 -1.5 -1) (8 0 1.5);
    }

    level4
    {
        type    box;
        box     (-2 -1 -0.6) (7 0 1);
    }

    level5
    {
        type    box;
        box     (-1 -0.6 -0.3) (6.5 0 0.8);
    }

    level6
    {
        type    box;
        box     (-0.5 -0.55 -0.15) (6.25 0 0.65);
    }
}

// ************************************************************************* //
""")

    # surfaceFeaturesDict : écriture directe pour conserver le format de la référence
    write_file(system_path / "surfaceFeaturesDict", """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    object      surfaceFeaturesDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

surfaces ("DTC-scaled.stl");

// Identify a feature when angle between faces < includedAngle
includedAngle   150;

subsetFeatures
{
    // Keep nonManifold edges (edges with >2 connected faces)
    nonManifoldEdges       yes;

    // Keep open edges (edges with 1 connected face)
    openEdges       yes;
}

// ************************************************************************* //
""")

    # snappyHexMeshDict : écriture directe pour conserver le contenu de la référence
    write_file(system_path / "snappyHexMeshDict", """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Which of the steps to run
castellatedMesh true;
snap            true;
addLayers       true;


// Geometry. Definition of all surfaces. All surfaces are of class
// surface.
// Surfaces are used
// - to specify refinement for any mesh cell intersecting it
// - to specify refinement for any mesh cell inside/outside/near
// - to 'snap' the mesh boundary to the surface
geometry
{
    hull
    {
        type triSurface;
        file "DTC-scaled.stl";

        patchInfo
        {
            type wall;
        }
    }
};



// Settings for the castellatedMesh generation.
castellatedMeshControls
{

    // Refinement parameters
    // ~~~~~~~~~~~~~~~~~~~~~

    // If local number of cells is >= maxLocalCells on any processor
    // switches from from refinement followed by balancing
    // (current method) to (weighted) balancing before refinement.
    maxLocalCells 100000;

    // Overall cell limit (approximately). Refinement will stop immediately
    // upon reaching this number so a refinement level might not complete.
    // Note that this is the number of cells before removing the part which
    // is not 'visible' from the keepPoint. The final number of cells might
    // actually be a lot less.
    maxGlobalCells 2000000;

    // The surface refinement loop might spend lots of iterations refining just a
    // few cells. This setting will cause refinement to stop if <= minimumRefine
    // are selected for refinement. Note: it will at least do one iteration
    // (unless the number of cells to refine is 0)
    minRefinementCells 0;

    // Number of buffer layers between different levels.
    // 1 means normal 2:1 refinement restriction, larger means slower
    // refinement.
    nCellsBetweenLevels 3;



    // Explicit feature edge refinement
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // Specifies a level for any cell intersected by its edges.
    // This is a featureEdgeMesh, read from constant/geometry for now.
    features
    (
         {
             file "DTC-scaled.eMesh";
             level 0;
         }
    );



    // Surface based refinement
    // ~~~~~~~~~~~~~~~~~~~~~~~~

    // Specifies two levels for every surface. The first is the minimum level,
    // every cell intersecting a surface gets refined up to the minimum level.
    // The second level is the maximum level. Cells that 'see' multiple
    // intersections where the intersections make an
    // angle > resolveFeatureAngle get refined up to the maximum level.

    refinementSurfaces
    {
        hull
        {
            // Surface-wise min and max refinement level
            level (0 0);
        }
    }

    resolveFeatureAngle 45;


    // Mesh selection
    // ~~~~~~~~~~~~~~

    // After refinement patches get added for all refinementSurfaces and
    // all cells intersecting the surfaces get put into these patches. The
    // section reachable from the insidePoint is kept.
    // NOTE: This point should never be on a face, always inside a cell, even
    // after refinement.
    insidePoint (-0.7 0 0);


    // Whether any faceZones (as specified in the refinementSurfaces)
    // are only on the boundary of corresponding cellZones or also allow
    // free-standing zone faces. Not used if there are no faceZones.
    allowFreeStandingZoneFaces true;
}



// Settings for the snapping.
snapControls
{
    //- Number of patch smoothing iterations before finding correspondence
    //  to surface
    nSmoothPatch 3;

    //- Relative distance for points to be attracted by surface feature point
    //  or edge. True distance is this factor times local
    //  maximum edge length.
    //    tolerance 4.0;
    tolerance 1.0;

    //- Number of mesh displacement relaxation iterations.
    nSolveIter 100;

    //- Maximum number of snapping relaxation iterations. Should stop
    //  before upon reaching a correct mesh.
    nRelaxIter 5;

    nFeatureSnapIter 10;
}



// Settings for the layer addition.
addLayersControls
{
    // Are the thickness parameters below relative to the undistorted
    // size of the refined cell outside layer (true) or absolute sizes (false).
    relativeSizes true;

    // Per final patch (so not geometry!) the layer information
    layers
    {
        hull
        {
            nSurfaceLayers 3;

        }
    }

    // Expansion factor for layer mesh
    expansionRatio 1.5;


    // Wanted thickness of final added cell layer. If multiple layers
    // is the thickness of the layer furthest away from the wall.
    // Relative to undistorted size of cell outside layer.
    // See relativeSizes parameter.
    finalLayerThickness 0.7;

    // Minimum thickness of cell layer. If for any reason layer
    // cannot be above minThickness do not add layer.
    // See relativeSizes parameter.
    minThickness 0.25;

    // If points get not extruded do nGrow layers of connected faces that are
    // also not grown. This helps convergence of the layer addition process
    // close to features.
    // Note: changed(corrected) w.r.t 17x! (didn't do anything in 17x)
    nGrow 0;


    // Advanced settings

    // When not to extrude surface. 0 is flat surface, 90 is when two faces
    // are perpendicular
    featureAngle 180;

    // At non-patched sides allow mesh to slip if extrusion direction makes
    // angle larger than slipFeatureAngle. Default is 0.5*featureAngle.
    slipFeatureAngle 30;

    // Maximum number of snapping relaxation iterations. Should stop
    // before upon reaching a correct mesh.
    nRelaxIter 5;

    // Number of smoothing iterations of surface normals
    nSmoothSurfaceNormals 1;

    // Number of smoothing iterations of interior mesh movement direction
    nSmoothNormals 3;

    // Smooth layer thickness over surface patches
    nSmoothThickness 10;

    // Stop layer growth on highly warped cells
    maxFaceThicknessRatio 0.5;

    // Reduce layer growth where ratio thickness to medial
    // distance is large
    maxThicknessToMedialRatio 0.3;

    // Angle used to pick up medial axis points
    // Note: changed(corrected) w.r.t 17x! 90 degrees corresponds to 130 in 17x.
    minMedianAxisAngle 90;

    // Create buffer region for new layer terminations
    nBufferCellsNoExtrude 0;


    // Overall max number of layer addition iterations. The mesher will exit
    // if it reaches this number of iterations; possibly with an illegal
    // mesh.
    nLayerIter 50;

    // Max number of iterations after which relaxed meshQuality controls
    // get used. Up to nRelaxIter it uses the settings in meshQualityControls,
    // after nRelaxIter it uses the values in meshQualityControls::relaxed.
    nRelaxedIter 20;
}



// Generic mesh quality settings. At any undoable phase these determine
// where to undo.
meshQualityControls
{
    #include "meshQualityDict"
}


// Advanced

// Flags for optional output
// 0 : only write final meshes
// 1 : write intermediate meshes
// 2 : write volScalarField with cellLevel for postprocessing
// 4 : write current intersections as .obj files
debug 0;


// Merge tolerance. Is fraction of overall bounding box of initial mesh.
// Note: the write tolerance needs to be higher than this.
mergeTolerance 1e-6;


// ************************************************************************* //
""")

    # meshQualityDict : écriture manuelle pour conserver les valeurs de la référence
    # (écrite avant snappyHexMeshDict car celui-ci l'inclut via #include)
    mq_content = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      meshQualityDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

//- Maximum non-orthogonality allowed. Set to 180 to disable.
maxNonOrtho 70;

//- Max skewness allowed. Set to <0 to disable.
maxBoundarySkewness 20;
maxInternalSkewness 4;

//- Max concaveness allowed. Is angle (in degrees) below which concavity
//  is allowed. 0 is straight face, <0 would be convex face.
//  Set to 180 to disable.
maxConcave 80;

//- Minimum cell pyramid volume relative to min bounding box length^3
//  Set to a fraction of the smallest cell volume expected.
//  Set to very negative number (e.g. -1e30) to disable.
minVol -1e30;

//- Minimum quality of the tet formed by the face-centre
//  and variable base point minimum decomposition triangles and
//  the cell centre.  Set to very negative number (e.g. -1e30) to
//  disable.
//     <0 = inside out tet,
//      0 = flat tet
//      1 = regular tet
minTetQuality 1e-30;

//- Minimum face twist. Set to <-1 to disable. dot product of face normal
//  and face centre triangles normal
minTwist 0.05;

//- Minimum normalised cell determinant
//  1 = hex, <= 0 = folded or flattened illegal cell
minDeterminant 0.001;

//- minFaceWeight (0 -> 0.5)
minFaceWeight 0.05;

//- minVolRatio (0 -> 1)
minVolRatio 0.01;

// Advanced

//- Number of error distribution iterations
nSmoothScale 4;
//- Amount to scale back displacement at error points
errorReduction 0.75;

// Optional : some meshing phases allow usage of relaxed rules.
// See e.g. addLayersControls::nRelaxedIter.
relaxed
{
    //- Maximum non-orthogonality allowed. Set to 180 to disable.
    maxNonOrtho 75;
}
"""
    write_file(system_path / "meshQualityDict", mq_content)

    # blockMeshDict : copie depuis la référence (le helper ne produit pas
    # le maillage structuré multi-blocs spécifique de ce cas)
    shutil.copy2(
        REFERENCE_CASE / "system" / "blockMeshDict",
        system_path / "blockMeshDict",
    )

    # 6. 0/ fields + boundary conditions
    fields_dir = case_path / "0"
    fields_dir.mkdir(parents=True, exist_ok=True)

    write_file(fields_dir / "U", """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       volVectorField;
    location    "0";
    object      U;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

UMean 1.668;

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (#neg $UMean 0 0);

boundaryField
{
    //- Set patchGroups for constraint patches
    #includeEtc "caseDicts/setConstraintTypes"

    inlet
    {
        type            fixedValue;
        value           $internalField;
    }

    outlet
    {
        type            outletPhaseMeanVelocity;
        alpha           alpha.water;
        UnMean          $UMean;
        value           $internalField;
    }

    atmosphere
    {
        type            pressureInletOutletVelocity;
        tangentialVelocity $internalField;
        value           uniform (0 0 0);
    }

    hull
    {
        type            movingWallVelocity;
        value           uniform (0 0 0);
    }
}


// ************************************************************************* //
""")

    write_file(fields_dir / "p_rgh", """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       volScalarField;
    location    "0";
    object      p_rgh;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [1 -1 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    //- Set patchGroups for constraint patches
    #includeEtc "caseDicts/setConstraintTypes"

    inlet
    {
        type            fixedFluxPressure;
        value           $internalField;
    }

    outlet
    {
        type            zeroGradient;
    }

    atmosphere
    {
        type            prghTotalPressure;
        p0              uniform 0;
    }

    hull
    {
        type            fixedFluxPressure;
        value           $internalField;
    }
}

// ************************************************************************* //
""")

    write_file(fields_dir / "alpha.water", """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       volScalarField;
    location    "0";
    object      alpha;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [];

internalField   uniform 0;

boundaryField
{
    //- Set patchGroups for constraint patches
    #includeEtc "caseDicts/setConstraintTypes"

    inlet
    {
        type            fixedValue;
        value           $internalField;
    }

    outlet
    {
        type            variableHeightFlowRate;
        lowerBound      0;
        upperBound      1;
        value           $internalField;
    }

    atmosphere
    {
        type            inletOutlet;
        inletValue      $internalField;
        value           $internalField;
    }

    hull
    {
        type            zeroGradient;
    }
}

// ************************************************************************* //
""")

    write_file(fields_dir / "k", """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       volScalarField;
    location    "0";
    object      k;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0.00015;

boundaryField
{
    //- Set patchGroups for constraint patches
    #includeEtc "caseDicts/setConstraintTypes"

    inlet
    {
        type            fixedValue;
        value           $internalField;
    }

    outlet
    {
       type            inletOutlet;
       inletValue      $internalField;
       value           $internalField;
    }

    atmosphere
    {
        type            inletOutlet;
        inletValue      $internalField;
        value           $internalField;
    }

    hull
    {
        type            kqRWallFunction;
        value           $internalField;
    }
}

// ************************************************************************* //
""")

    write_file(fields_dir / "omega", """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       volScalarField;
    location    "0";
    object      omega;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 -1 0 0 0 0];

internalField   uniform 2;

boundaryField
{
    //- Set patchGroups for constraint patches
    #includeEtc "caseDicts/setConstraintTypes"

    inlet
    {
        type            fixedValue;
        value           $internalField;
    }

    outlet
    {
       type            inletOutlet;
       inletValue      $internalField;
       value           $internalField;
    }

    atmosphere
    {
        type            inletOutlet;
        inletValue      $internalField;
        value           $internalField;
    }

    hull
    {
        type            omegaWallFunction;
        value           $internalField;
    }
}



// ************************************************************************* //
""")

    write_file(fields_dir / "nut", """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       volScalarField;
    location    "0";
    object      nut;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 5e-07;

boundaryField
{
    //- Set patchGroups for constraint patches
    #includeEtc "caseDicts/setConstraintTypes"

    inlet
    {
        type            fixedValue;
        value           $internalField;
    }

    outlet
    {
        type            zeroGradient;
    }

    atmosphere
    {
        type            zeroGradient;
    }

    hull
    {
        type            nutkRoughWallFunction;
        Ks              uniform 100e-6;
        Cs              uniform 0.5;
        value           $internalField;
    }
}


// ************************************************************************* //
""")

    write_file(fields_dir / "pointDisplacement", """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       pointVectorField;
    location    "0";
    object      pointDisplacement;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 0 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    //- Set patchGroups for constraint patches
    #includeEtc "caseDicts/setConstraintTypes"

    inlet
    {
        type            fixedValue;
        value           uniform (0 0 0);
    }

    outlet
    {
        type            fixedValue;
        value           uniform (0 0 0);
    }

    atmosphere
    {
        type            fixedValue;
        value           uniform (0 0 0);
    }

    hull
    {
        type            calculated;
    }
}


// ************************************************************************* //
""")

    # Backup 0.orig
    orig_dir = case_path / "0.orig"
    orig_dir.mkdir(parents=True, exist_ok=True)
    for field_file in fields_dir.iterdir():
        if field_file.is_file() and not field_file.name.startswith("."):
            shutil.copy2(field_file, orig_dir / field_file.name)

    return case_path


def run_case(case_path: Path, processors: int, execute: bool) -> None:
    if not execute:
        return

    subprocess.run(["setFields"], cwd=case_path, check=True)

    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = True
    solver.is_vof = True
    solver.transient = True
    solver.turbulence_model = "kOmegaSST"
    solver.constant.transportProperties.nu = ValueWithUnit(1.004e-6, "m^2/s")

    solver.run_simulation(nb_proc=processors)


def main() -> None:
    args = parse_arguments()
    case_path = build_case(args.mesh_source)
    print(f"Case: {case_path}")
    run_case(case_path, args.processors, args.execute)


if __name__ == "__main__":
    main()
