#include "marineOversetProbe.H"
#include "MarineOversetCellState.H"
#include "MarineOversetInterpolation.H"
#include "volFields.H"
#include "addToRunTimeSelectionTable.H"
#include "IOobject.H"
#include "IFstream.H"
#include "fileName.H"

namespace Foam
{
namespace fvMeshMovers
{
    defineTypeNameAndDebug(marineOversetProbe, 0);
    addToRunTimeSelectionTable(fvMeshMover, marineOversetProbe, fvMesh);
}
}

Foam::fvMeshMovers::marineOversetProbe::marineOversetProbe(fvMesh& mesh)
:
    fvMeshMover(mesh),
    state_(new MarineOversetCellState(mesh)),
    interpolation_
    (
        new MarineOversetInterpolation
        (
            state_->donorIndices(),
            state_->weights(),
            mesh.nCells()
        )
    ),
    matrixOperator_(new MarineOversetMatrix(*state_, *interpolation_))
{}

Foam::fvMeshMovers::marineOversetProbe::marineOversetProbe
(
    fvMesh& mesh,
    const dictionary& dict
)
:
    fvMeshMover(mesh),
    state_(new MarineOversetCellState(mesh)),
    interpolation_
    (
        new MarineOversetInterpolation
        (
            state_->donorIndices(),
            state_->weights(),
            mesh.nCells()
        )
    ),
    matrixOperator_(new MarineOversetMatrix(*state_, *interpolation_))
{
    (void)dict;
    const fileName stencilFile
    (
        mesh.time().constant() / "marineOversetStencils"
    );

    IFstream stencilStream(stencilFile);
    if (!stencilStream.good())
    {
        WarningInFunction
            << "No constant/marineOversetStencils file found. "
            << "The probe validates only zoneID; matrix application "
            << "requires an explicit solver call." << nl;
    }

    Info<< "marineOversetProbe: zoneID validated for "
        << state_->nZones() << " zone(s), cells: calculated="
        << state_->nCalculated() << ", interpolated="
        << state_->nInterpolated() << ", holes=" << state_->nHoles()
        << ", stencils=" << state_->nStencils() << nl;

    if (state_->nStencils())
    {
        IOobject pObject
        (
            "p",
            Time::timeName(mesh.time().value()),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false
        );
        volScalarField p(pObject, mesh);

        Info<< "marineOversetProbe: stencil 0 interpolated p="
            << interpolation_->interpolateScalar(0, p.internalField()) << nl;
    }

    Info<< "marineOversetProbe: matrix operator ready for scalar/vector equations" << nl;
}

Foam::fvMeshMovers::marineOversetProbe::~marineOversetProbe()
{}

bool Foam::fvMeshMovers::marineOversetProbe::update()
{
    // The mover lifecycle updates geometry only. Matrix constraints are
    // applied explicitly by the solver after each equation is assembled.
    return false;
}

void Foam::fvMeshMovers::marineOversetProbe::applyScalar
(
    fvMatrix<scalar>& matrix
) const
{
    matrixOperator_->applyScalar(matrix);
}

void Foam::fvMeshMovers::marineOversetProbe::applyVector
(
    fvMatrix<vector>& matrix
) const
{
    matrixOperator_->applyVector(matrix);
}

// ************************************************************************* //
