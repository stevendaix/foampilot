#include "argList.H"
#include "MarineOversetInterpolation.H"
#include "scalarField.H"
#include "vectorField.H"
#include "error.H"

using namespace Foam;

int main(int argc, char* argv[])
{
    argList::noParallel();
    argList args(argc, argv);

    List<labelList> donorIndices(1);
    donorIndices[0] = labelList(4);
    donorIndices[0][0] = 0;
    donorIndices[0][1] = 1;
    donorIndices[0][2] = 2;
    donorIndices[0][3] = 3;

    List<scalarList> weights(1);
    weights[0] = scalarList(4, 0.25);

    MarineOversetInterpolation interpolation(donorIndices, weights, 4);

    scalarField scalarDonors(4);
    scalarDonors[0] = 2;
    scalarDonors[1] = 4;
    scalarDonors[2] = 6;
    scalarDonors[3] = 8;
    const scalar scalarValue = interpolation.interpolateScalar(0, scalarDonors);

    vectorField vectorDonors(4, vector::zero);
    vectorDonors[0] = vector(0, 1, 2);
    vectorDonors[1] = vector(2, 3, 4);
    vectorDonors[2] = vector(4, 5, 6);
    vectorDonors[3] = vector(6, 7, 8);
    const vector vectorValue = interpolation.interpolateVector(0, vectorDonors);

    const scalar scalarError = mag(scalarValue - 5.0);
    const scalar vectorError = mag(vectorValue - vector(3, 4, 5));

    Info<< "scalarValue=" << scalarValue << " scalarError=" << scalarError << nl
        << "vectorValue=" << vectorValue << " vectorError=" << vectorError << nl;

    if (scalarError > 1e-12 || vectorError > 1e-12)
    {
        FatalErrorInFunction
            << "Analytical interpolation test failed" << nl
            << exit(FatalError);
    }

    Info<< "MarineOversetInterpolation analytical test passed" << nl;
    return 0;
}
