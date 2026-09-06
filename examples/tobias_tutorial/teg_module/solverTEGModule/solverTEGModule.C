/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2022-2024 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "solverTEGModule.H"
#include "fvm.H"
#include "fvConstraints.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace solvers
{
    defineTypeNameAndDebug(solverTEGModule, 0);
    addToRunTimeSelectionTable(solver, solverTEGModule, fvMesh);
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::solvers::solverTEGModule::solverTEGModule
(
    fvMesh& mesh
)
:
    solver(mesh),

    T_
    (
        IOobject
        (
            "T",
            runTime.name(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    ),

    DT_
    (
        IOobject
        (
            "DT",
            runTime.name(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    ),

    T(T_),
    DT(DT_)
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::solvers::solverTEGModule::~solverTEGModule()
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

Foam::scalar Foam::solvers::solverTEGModule::maxDeltaT() const
{
    return 0;
}


void Foam::solvers::solverTEGModule::preSolve()
{}


void Foam::solvers::solverTEGModule::moveMesh()
{}


void Foam::solvers::solverTEGModule::motionCorrector()
{}


void Foam::solvers::solverTEGModule::prePredictor()
{}


void Foam::solvers::solverTEGModule::momentumTransportPredictor()
{}


void Foam::solvers::solverTEGModule::thermophysicalTransportPredictor()
{}


void Foam::solvers::solverTEGModule::momentumPredictor()
{}


void Foam::solvers::solverTEGModule::thermophysicalPredictor()
{
    while (pimple.correctNonOrthogonal())
    {
        fvScalarMatrix TEqn
        (
            fvm::ddt(T)
          - fvm::laplacian(DT, T)
          ==
            fvModels().source(T)
        );

        TEqn.relax();

        fvConstraints().constrain(TEqn);

        TEqn.solve();

        fvConstraints().constrain(T);
    }
}


void Foam::solvers::solverTEGModule::momentumTransportCorrector()
{}


void Foam::solvers::solverTEGModule::thermophysicalTransportCorrector()
{}


void Foam::solvers::solverTEGModule::pressureCorrector()
{}


void Foam::solvers::solverTEGModule::postCorrector()
{}


void Foam::solvers::solverTEGModule::postSolve()
{}


// ************************************************************************* //
