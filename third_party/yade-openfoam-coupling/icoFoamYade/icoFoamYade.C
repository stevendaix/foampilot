/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
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

Application
    icoFoam

Description
    Transient solver for incompressible, laminar flow of Newtonian fluids.

\*---------------------------------------------------------------------------*/

// Check development branch, the foundation version reorganized classes after version 6
#if foamVersion < 1000 && foamVersion > 10
    #define newFoundationVersion 1
#else
    #define isFoundationVersion 0
#endif

#if (newFoundationVersion)
   // The content of "fvCFD.H" has been splitted, pick what's necessary below
    #include "argList.H"
    #include "volFields.H"
    #include "fvMesh.H"
    #include "fvSchemes.H"
    #include "fvSolution.H"
    #include "surfaceFields.H"
    #include "fvm.H"

    #include "pressureReference.H"
    #include "findRefCell.H"
    #include "constrainPressure.H"
    #include "constrainHbyA.H"
    #include "adjustPhi.H"

    #include "fvcDdt.H"
    #include "fvcGrad.H"
    #include "fvcFlux.H"

    #include "fvmDdt.H"
    #include "fvmDiv.H"
    #include "fvmLaplacian.H"
    #include "fvcLaplacian.H"
    #include "polyMesh.H"



#endif

#include "pisoControl.H"
#include "../FoamYade/FoamYade.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    pisoControl piso(mesh);

    #include "createFields.H"
    #include "initContinuityErrs.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    bool gaussianInterp = false;


    FoamYade yadeCoupling(mesh,U, gradP, vGrad, divT,ddtU_f,g,uSourceDrag,alphac, uSource, uParticle, uCoeff,uInterp, gaussianInterp);

    yadeCoupling.setScalarProperties(partDensity.value(), fluidDensity.value(), nu.value());
    std::cout << "done set of part properties" << std::endl;


    while (runTime.loop())
    {
        Info<< "Time = " << runTime.name() << nl << endl;
        #include "CourantNo.H"


        vGrad = fvc::grad(U);


        yadeCoupling.setParticleAction(runTime.deltaT().value());


        // Momentum predictor

        fvVectorMatrix UEqn
        (
            fvm::ddt(U)
          + fvm::div(phi, U)
          - fvm::laplacian(nu, U)
           ==uSource
          );



        if (piso.momentumPredictor())
        {
            solve(UEqn == -fvc::grad(p));
        }

        // --- PISO loop
        while (piso.correct())
        {
            volScalarField rAU(1.0/UEqn.A());
            volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));
            surfaceScalarField phiHbyA
            (
                "phiHbyA",
                fvc::flux(HbyA)
              + fvc::interpolate(rAU)*fvc::ddtCorr(U, phi)
            );

            adjustPhi(phiHbyA, U, p);

            // Update the pressure BCs to ensure flux consistency
            constrainPressure(p, U, phiHbyA, rAU);

            // Non-orthogonal pressure corrector loop
            while (piso.correctNonOrthogonal())
            {
                // Pressure corrector

                fvScalarMatrix pEqn
                (
                    fvm::laplacian(rAU, p) == fvc::div(phiHbyA)
                );

                pEqn.setReference(pRefCell, pRefValue);

            #if (newFoundationVersion)
              pEqn.solve();
            #else
              pEqn.solve(mesh.solver(p.select(piso.finalInnerIter())));
            #endif

                if (piso.finalNonOrthogonalIter())
                {
                    phi = phiHbyA - pEqn.flux();
                }
            }


            #include "continuityErrs.H"

            U = HbyA - rAU*fvc::grad(p);
            U.correctBoundaryConditions();


        }

        runTime.write();

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
	yadeCoupling.setSourceZero();

    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
