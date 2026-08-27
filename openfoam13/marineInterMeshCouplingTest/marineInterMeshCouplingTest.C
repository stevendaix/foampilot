#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"
#include "fvmDdt.H"
#include "MarineInterMeshStencilState.H"
#include "MarineInterMeshMatrix.H"

using namespace Foam;

int main(int argc, char* argv[])
{
    argList::noParallel();
    argList args(argc, argv);
    Time runTime(Time::controlDictName, args);

    fvMesh hullMesh
    (
        IOobject
        (
            "region0",
            runTime.timePath().name(),
            runTime,
            IOobject::MUST_READ
        )
    );
    fvMesh backgroundMesh
    (
        IOobject
        (
            "background",
            runTime.timePath().name(),
            runTime,
            IOobject::MUST_READ
        )
    );

    volScalarField donor
    (
        IOobject
        (
            "donorProbe",
            runTime.timePath().name(),
            backgroundMesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        backgroundMesh,
        dimensionedScalar(dimless, 0)
    );
    donor.primitiveFieldRef() = scalar(2);
    donor.correctBoundaryConditions();

    volScalarField target
    (
        IOobject
        (
            "targetProbe",
            runTime.timePath().name(),
            hullMesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        hullMesh,
        dimensionedScalar(dimless, 0)
    );
    fvScalarMatrix equation(fvm::ddt(target));
    MarineInterMeshStencilState state(hullMesh, backgroundMesh.nCells());
    MarineInterMeshMatrix operator_(state);
    operator_.applyScalar(equation, donor);

    const labelList& acceptorIndices = state.acceptorIndices();
    forAll(acceptorIndices, stencilI)
    {
        const label celli = acceptorIndices[stencilI];
        const scalar expected = 2;
        const scalar reconstructed = equation.source()[celli]
            / equation.diag()[celli];
        if (mag(reconstructed - expected) > 1e-10)
        {
            FatalErrorInFunction
                << "Unexpected interpolated value at cell " << celli
                << ": " << reconstructed << nl
                << exit(FatalError);
        }
    }

    Info<< "inter-mesh matrix coupling passed: target=" << hullMesh.nCells()
        << ", donor=" << backgroundMesh.nCells()
        << ", stencils=" << state.size() << nl;
    return 0;
}
