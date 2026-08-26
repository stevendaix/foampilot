/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2022-2023 OpenFOAM Foundation
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

Class
    Foam::solvers::solverTEGModule

Description
    Simple solver module to solve the temperature equation

SourceFiles
    solverTEGModule.C

\*---------------------------------------------------------------------------*/

#ifndef solverTEGModule_H
#define solverTEGModule_H

#include "solver.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace solvers
{

/*---------------------------------------------------------------------------*\
                          Class solverTEGModule Declaration
\*---------------------------------------------------------------------------*/

class solverTEGModule
:
    public solver
{

protected:

    // Fields

        //- Temperature field
        volScalarField T_;

        //- Thermal conductivity field
        volScalarField DT_;


public:

    // Public Data

        //- Reference to the temperature field
        volScalarField& T;

        //- Const reference to the thermal conductivity field
        const volScalarField& DT;


    //- Runtime type information
    TypeName("solverTEGModule");


    // Constructors

        //- Construct from region mesh
        solverTEGModule(fvMesh& mesh);

        //- Disallow default bitwise copy construction
        solverTEGModule(const solverTEGModule&) = delete;


    //- Destructor
    virtual ~solverTEGModule();


    // Member Functions

        //- Do nothing
        virtual scalar maxDeltaT() const;

        //- Do nothing
        virtual void preSolve();

        //- Do nothing
        virtual void moveMesh();

        //- Do nothing
        virtual void motionCorrector();

        //- Do nothing
        virtual void prePredictor();

        //- Do nothing
        virtual void momentumTransportPredictor();

        //- Do nothing; the custom equation is solved below
        virtual void thermophysicalTransportPredictor();

        //- Do nothing
        virtual void momentumPredictor();

        //- Construct and solve the temperature equation
        virtual void thermophysicalPredictor();

        //- Do nothing
        virtual void momentumTransportCorrector();

        //- Do nothing
        virtual void thermophysicalTransportCorrector();

        //- Do nothing
        virtual void pressureCorrector();

        //- Do nothing
        virtual void postCorrector();

        //- Do nothing
        virtual void postSolve();


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const solverTEGModule&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace solvers
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
