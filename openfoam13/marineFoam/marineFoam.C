/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Version: 13
    \\  /    A nd           | Website: https://openfoam.org
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Application
    marineFoam

Description
    Foundation OpenFOAM 13 marine runner.  The executable uses the standard
    OpenFOAM solver-module interface and is deliberately thin: the selected
    module performs the fluid physics while dynamic mesh, fvModels,
    fvConstraints, MRF and function objects remain native OpenFOAM inputs.

    By default the runner selects incompressibleVoF.  A different Foundation
    module can be selected with -solver or with the solver entry in controlDict.
    This allows the same executable to run moving-mesh marine cases without
    maintaining a fork of the incompressibleVoF equations.

Usage
    marineFoam [-solver <module>]
\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "solver.H"
#include "pimpleSingleRegionControl.H"
#include "setDeltaT.H"
#include "volFields.H"

using namespace Foam;

int main(int argc, char *argv[])
{
    argList::addOption("solver", "name", "Foundation solver module name");
    argList::addOption("donor-region", "name", "Optional inter-mesh donor region");
    #include "setRootCase.H"
    #include "createTime.H"

    word solverName
    (
        runTime.controlDict().lookupOrDefault("solver", word::null)
    );

    args.optionReadIfPresent("solver", solverName);

    if (solverName == word::null)
    {
        solverName = "incompressibleVoF";
    }

    solver::load(solverName);

    #include "createMesh.H"

    autoPtr<fvMesh> donorMeshPtr(nullptr);
    autoPtr<volVectorField> donorUPtr(nullptr);
    autoPtr<volScalarField> donorPRghPtr(nullptr);
    autoPtr<volScalarField> donorAlphaPtr(nullptr);
    autoPtr<volScalarField> donorKPtr(nullptr);
    autoPtr<volScalarField> donorOmegaPtr(nullptr);
    autoPtr<volScalarField> donorEpsilonPtr(nullptr);
    autoPtr<volScalarField> donorNutPtr(nullptr);

    word donorRegion;
    if (args.optionReadIfPresent("donor-region", donorRegion))
    {
        donorMeshPtr.reset
        (
            new fvMesh
            (
                IOobject
                (
                    donorRegion,
                    runTime.timePath().name(),
                    runTime,
                    IOobject::MUST_READ
                )
            )
        );
        donorUPtr.reset
        (
            new volVectorField
            (
                IOobject("U", runTime.timePath().name(), *donorMeshPtr, IOobject::MUST_READ),
                *donorMeshPtr
            )
        );
        donorPRghPtr.reset
        (
            new volScalarField
            (
                IOobject("p_rgh", runTime.timePath().name(), *donorMeshPtr, IOobject::MUST_READ),
                *donorMeshPtr
            )
        );
        donorAlphaPtr.reset
        (
            new volScalarField
            (
                IOobject("alpha.water", runTime.timePath().name(), *donorMeshPtr, IOobject::MUST_READ),
                *donorMeshPtr
            )
        );
        donorKPtr.reset
        (
            new volScalarField
            (
                IOobject("k", runTime.timePath().name(), *donorMeshPtr, IOobject::MUST_READ),
                *donorMeshPtr
            )
        );
        donorOmegaPtr.reset
        (
            new volScalarField
            (
                IOobject("omega", runTime.timePath().name(), *donorMeshPtr, IOobject::MUST_READ),
                *donorMeshPtr
            )
        );
        donorEpsilonPtr.reset
        (
            new volScalarField
            (
                IOobject("epsilon", runTime.timePath().name(), *donorMeshPtr, IOobject::MUST_READ),
                *donorMeshPtr
            )
        );
        donorNutPtr.reset
        (
            new volScalarField
            (
                IOobject("nut", runTime.timePath().name(), *donorMeshPtr, IOobject::MUST_READ),
                *donorMeshPtr
            )
        );
        Info<< "Loaded donor region " << donorRegion << " with "
            << donorMeshPtr->nCells() << " cells" << nl;
    }

    autoPtr<solver> solverPtr(solver::New(solverName, mesh));
    solver& marineSolver = solverPtr();
    pimpleSingleRegionControl pimple(marineSolver.pimple);

    setDeltaT(runTime, marineSolver);

    Info<< "\nStarting marineFoam with solver module: " << solverName
        << "\n" << endl;

    while (pimple.run(runTime))
    {
        marineSolver.preSolve();
        adjustDeltaT(runTime, marineSolver);
        runTime++;
        Info<< "Time = " << runTime.userTimeName() << nl << endl;

        while (pimple.loop())
        {
            if (marineSolver.pimple.flow())
            {
                marineSolver.moveMesh();
                marineSolver.motionCorrector();
            }

            if (marineSolver.pimple.models())
            {
                marineSolver.fvModels().correct();
            }

            marineSolver.prePredictor();

            if (marineSolver.pimple.predictTransport())
            {
                if (marineSolver.pimple.flow())
                {
                    marineSolver.momentumTransportPredictor();
                }
                if (marineSolver.pimple.thermophysics())
                {
                    marineSolver.thermophysicalTransportPredictor();
                }
            }

            if (marineSolver.pimple.flow())
            {
                marineSolver.momentumPredictor();
            }
            if (marineSolver.pimple.thermophysics())
            {
                marineSolver.thermophysicalPredictor();
            }
            if (marineSolver.pimple.flow())
            {
                marineSolver.pressureCorrector();
            }

            if (marineSolver.pimple.correctTransport())
            {
                if (marineSolver.pimple.flow())
                {
                    marineSolver.momentumTransportCorrector();
                }
                if (marineSolver.pimple.thermophysics())
                {
                    marineSolver.thermophysicalTransportCorrector();
                }
            }
        }

        marineSolver.postSolve();
        runTime.write();
        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;
    return 0;
}

// ************************************************************************* //
