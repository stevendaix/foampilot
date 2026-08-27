#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"
#include "fvMatrices.H"
#include "fvConstraints.H"
#include "fvmDdt.H"
#include "MarineOversetCellState.H"
#include "MarineOversetInterpolation.H"
#include "MarineOversetMatrix.H"

using namespace Foam;

int main(int argc, char* argv[])
{
    argList::noParallel();
    argList args(argc, argv);
    Time runTime(Time::controlDictName, args);
    fvMesh mesh
    (
        IOobject
        (
            "region0",
            runTime.timePath().name(),
            runTime,
            IOobject::MUST_READ
        )
    );

    volScalarField p
    (
        IOobject
        (
            "p",
            runTime.timePath().name(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        mesh
    );

    fvScalarMatrix matrix(fvm::ddt(p));
    fvConstraints& constraints = fvConstraints::New(mesh);
    if (!constraints.constrain(matrix))
    {
        FatalErrorInFunction
            << "The marine overset fvConstraint did not constrain p" << nl
            << exit(FatalError);
    }

    const scalarField& matrixDiag = matrix.diag();
    const scalarField& matrixSource = matrix.source();
    const scalar interpolatedValue = matrixSource[2]/matrixDiag[2];
    const scalar holeValue = matrixSource[3]/matrixDiag[3];

    if
    (
        mag(interpolatedValue - scalar(1.5)) > SMALL
     || mag(holeValue) > SMALL
     || mag(p.internalField()[2] - scalar(1.5)) > SMALL
     || mag(p.internalField()[3]) > SMALL
    )
    {
        FatalErrorInFunction
            << "Unexpected constrained values: interpolated="
            << interpolatedValue << ", hole=" << holeValue << nl
            << exit(FatalError);
    }

    Info<< "matrix overset application passed: constrained cells = "
        << 2
        << ", interpolatedValue=" << interpolatedValue
        << ", holeValue=" << holeValue << nl;
    return 0;
}
