#!/usr/bin/env python3
"""Steady propeller MRF simulation with rhoSimpleFoam.

Usage:
    python run_simu.py --processors 4 --execute
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from foampilot import Solver, ValueWithUnit
from foampilot.base import CaseBuilder, Meshing
from foampilot.mesh import write_rotating_zone
from foampilot.constant import MomentumTransportFile
from foampilot.system.controlDictFile import ControlDictFile
from foampilot.system.decomposeParDictFile import DecomposeParDictFile

CASE_ROOT = Path(__file__).parent.resolve()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Propeller MRF case.")
    parser.add_argument("--processors", type=int, default=4, help="MPI ranks.")
    parser.add_argument("--execute", action="store_true", help="Run the simulation.")
    return parser.parse_args()


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def build_case() -> Path:
    case_path = CASE_ROOT / "propeller_mrf"
    if case_path.exists():
        shutil.rmtree(case_path)
    case_path.mkdir(parents=True)

    CaseBuilder(case_path).ensure_dirs(extra_dirs=("geometry",))

    mesh = Meshing(case_path, mesher="snappy")
    mesh.mesher.addLayers = True
    mesh.mesher.locationInMesh = (0.0, 0.0, 0.0)
    mesh.mesher.resolveFeatureAngle = 45
    mesh.mesher.write()

    write_rotating_zone(
        case_path,
        cell_zone="rotor",
        origin=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
        omega=314.16,
        non_rotating_patches=("AMI1", "AMI2"),
    )

    solver = Solver(case_path, solver_name="rhoSimpleFoam")
    solver.compressible = False
    solver.with_gravity = False
    solver.is_vof = False
    solver.transient = False
    solver.turbulence_model = "kEpsilon"

    simulation_type, model = solver.get_turbulence_configuration()
    mt_file = MomentumTransportFile(
        parent=solver,
        simulationType=simulation_type,
        RASModel=model if simulation_type == "RAS" else None,
    )
    mt_file.write(case_path / "constant" / "momentumTransport")

    solver.constant.transportProperties.nu = ValueWithUnit(1.004e-6, "m^2/s")
    solver.constant.transportProperties.rho = ValueWithUnit(998.8, "kg/m^3")
    solver.constant.transportProperties.write(case_path / "constant" / "transportProperties")

    system_path = case_path / "system"
    system_path.mkdir(parents=True, exist_ok=True)

    control_dict = ControlDictFile(
        parent=solver,
        application="rhoSimpleFoam",
        startFrom="startTime",
        startTime=0,
        stopAt="endTime",
        endTime=1000,
        deltaT=1,
        writeControl="timeStep",
        writeInterval=100,
        purgeWrite=2,
        writeFormat="binary",
        writePrecision=6,
        writeCompression="off",
        timeFormat="general",
        timePrecision=6,
        runTimeModifiable="yes",
    )
    control_dict.use_solver_keyword = False
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
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    div(phi,U)       Gauss upwind;
    div(phi,k)       Gauss upwind;
    div(phi,epsilon) Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
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
    p
    {
        solver          GAMG;
        smoother        DICGaussSeidel;

        tolerance       1e-7;
        relTol          0.01;
        nCellsInCoarsestLevel 20;
        cacheAgglomeration true;
        agglomerator    faceAreaPair;
        mergeLevels     1;
    }

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;

        tolerance       1e-8;
        relTol          0;
        nSweeps         2;
    }

    k
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;

        tolerance       1e-8;
        relTol          0;
        nSweeps         2;
    }

    epsilon
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;

        tolerance       1e-8;
        relTol          0;
        nSweeps         2;
    }

    nut
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;

        tolerance       1e-8;
        relTol          0;
        nSweeps         2;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 2;
    residualControl
    {
        p               1e-4;
        U               1e-4;
        "(k|epsilon|nut)" 1e-4;
    }
    pRefCell         0;
    pRefValue        0;
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
        k               0.7;
        epsilon         0.7;
        nut             0.7;
    }
}


// ************************************************************************* //
""")
    DecomposeParDictFile(parent=solver, nb_proc=4).write(system_path / "decomposeParDict")

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

// ************************************************************************* //
""")

    write_file(system_path / "meshQualityDict", """/*--------------------------------*- C++ -*----------------------------------*\\
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
""")

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

castellatedMesh true;
snap            true;
addLayers       true;

geometry
{
    propeller
    {
        type triSurface;
        file "propeller.stl";

        patchInfo
        {
            type wall;
        }
    }
};

castellatedMeshControls
{
    maxLocalCells 100000;
    maxGlobalCells 2000000;
    minRefinementCells 0;
    nCellsBetweenLevels 3;

    features
    (
         {
             file "propeller.eMesh";
             level 0;
         }
    );

    refinementSurfaces
    {
        propeller
        {
            level (0 0);
        }
    }

    resolveFeatureAngle 45;
    insidePoint (0 0 0);
    allowFreeStandingZoneFaces true;
}

snapControls
{
    nSmoothPatch 3;
    tolerance 1.0;
    nSolveIter 100;
    nRelaxIter 5;
    nFeatureSnapIter 10;
}

addLayersControls
{
    relativeSizes true;
    layers
    {
        propeller
        {
            nSurfaceLayers 3;
        }
    }
    expansionRatio 1.5;
    finalLayerThickness 0.7;
    minThickness 0.25;
    nGrow 0;
    featureAngle 180;
    slipFeatureAngle 30;
    nRelaxIter 5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
    nRelaxedIter 20;
}

meshQualityControls
{
    #include "meshQualityDict"
}

debug 0;
mergeTolerance 1e-6;


// ************************************************************************* //
""")

    fields_dir = case_path / "0"
    fields_dir.mkdir(parents=True, exist_ok=True)

    write_file(fields_dir / "p", """/*--------------------------------*- C++ -*----------------------------------*\\
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
    object      p;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }

    outlet
    {
        type            zeroGradient;
    }

    wall
    {
        type            zeroGradient;
    }

    AMI1
    {
        type            cyclicAMI;
        value           $internalField;
    }

    AMI2
    {
        type            cyclicAMI;
        value           $internalField;
    }
}

// ************************************************************************* //
""")

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

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (10 0 0);

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           $internalField;
    }

    outlet
    {
        type            zeroGradient;
    }

    wall
    {
        type            noSlip;
    }

    AMI1
    {
        type            cyclicAMI;
        value           $internalField;
    }

    AMI2
    {
        type            cyclicAMI;
        value           $internalField;
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

internalField   uniform 0.375;

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           $internalField;
    }

    outlet
    {
        type            zeroGradient;
    }

    wall
    {
        type            kqRWallFunction;
        value           $internalField;
    }

    AMI1
    {
        type            cyclicAMI;
        value           $internalField;
    }

    AMI2
    {
        type            cyclicAMI;
        value           $internalField;
    }
}

// ************************************************************************* //
""")

    write_file(fields_dir / "epsilon", """/*--------------------------------*- C++ -*----------------------------------*\\
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
    object      epsilon;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -3 0 0 0 0];

internalField   uniform 0.125;

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           $internalField;
    }

    outlet
    {
        type            zeroGradient;
    }

    wall
    {
        type            epsilonWallFunction;
        value           $internalField;
    }

    AMI1
    {
        type            cyclicAMI;
        value           $internalField;
    }

    AMI2
    {
        type            cyclicAMI;
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

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           $internalField;
    }

    outlet
    {
        type            zeroGradient;
    }

    wall
    {
        type            nutkWallFunction;
        value           $internalField;
    }

    AMI1
    {
        type            cyclicAMI;
        value           $internalField;
    }

    AMI2
    {
        type            cyclicAMI;
        value           $internalField;
    }
}

// ************************************************************************* //
""")

    orig_dir = case_path / "0.orig"
    orig_dir.mkdir(parents=True, exist_ok=True)
    for field_file in fields_dir.iterdir():
        if field_file.is_file() and not field_file.name.startswith("."):
            shutil.copy2(field_file, orig_dir / field_file.name)

    return case_path


def run_case(case_path: Path, processors: int, execute: bool) -> None:
    if not execute:
        return

    solver = Solver(case_path, solver_name="rhoSimpleFoam")
    solver.compressible = False
    solver.with_gravity = False
    solver.is_vof = False
    solver.transient = False
    solver.turbulence_model = "kEpsilon"
    solver.constant.transportProperties.nu = ValueWithUnit(1.004e-6, "m^2/s")

    solver.run_simulation(nb_proc=processors)


def main() -> None:
    args = parse_arguments()
    case_path = build_case()
    print(f"Case: {case_path}")
    run_case(case_path, args.processors, args.execute)


if __name__ == "__main__":
    main()
