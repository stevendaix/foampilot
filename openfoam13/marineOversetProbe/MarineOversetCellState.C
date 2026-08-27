#include "MarineOversetCellState.H"
#include "IOobject.H"
#include "IOdictionary.H"
#include "Time.H"
#include "mathematicalConstants.H"
#include <cmath>

Foam::MarineOversetCellState::MarineOversetCellState(const fvMesh& mesh)
:
    mesh_(mesh),
    zoneId_(mesh.nCells(), 0),
    status_(mesh.nCells(), calculated),
    donorIndices_(),
    weights_(),
    acceptorIndices_(),
    nStencils_(0)
{
    readAndValidate();
    readAndValidateStencils();
}

void Foam::MarineOversetCellState::readAndValidateStencils()
{
    IOdictionary stencilDict
    (
        IOobject
        (
            "marineOversetStencils",
            mesh_.time().constant(),
            mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE,
            false
        )
    );

    if (!stencilDict.headerOk())
    {
        return;
    }

    const List<dictionary> acceptors(stencilDict.lookup("acceptors"));
    donorIndices_.setSize(acceptors.size());
    weights_.setSize(acceptors.size());
    acceptorIndices_.setSize(acceptors.size());

    forAll(acceptors, acceptorI)
    {
        acceptors[acceptorI].lookup("index") >> acceptorIndices_[acceptorI];
        acceptors[acceptorI].lookup("donorIndices") >> donorIndices_[acceptorI];
        acceptors[acceptorI].lookup("weights") >> weights_[acceptorI];

        const labelList& donorIndices = donorIndices_[acceptorI];
        const scalarList& weights = weights_[acceptorI];

        if (donorIndices.empty() || donorIndices.size() != weights.size())
        {
            FatalErrorInFunction
                << "Stencil " << acceptorI
                << " must have aligned non-empty donorIndices and weights" << nl
                << exit(FatalError);
        }

        scalar weightSum = 0;
        forAll(donorIndices, donorI)
        {
            if (donorIndices[donorI] < 0 || donorIndices[donorI] >= mesh_.nCells())
            {
                FatalErrorInFunction
                    << "Stencil " << acceptorI << " contains donor cell "
                    << donorIndices[donorI] << " outside [0, "
                    << mesh_.nCells() - 1 << "]" << nl
                    << exit(FatalError);
            }
            if (!std::isfinite(weights[donorI]) || weights[donorI] < 0)
            {
                FatalErrorInFunction
                    << "Stencil " << acceptorI
                    << " contains an invalid weight " << weights[donorI] << nl
                    << exit(FatalError);
            }
            weightSum += weights[donorI];
        }

        if (mag(weightSum - 1.0) > 1e-10)
        {
            FatalErrorInFunction
                << "Stencil " << acceptorI << " weights sum to "
                << weightSum << ", expected 1" << nl
                << exit(FatalError);
        }
    }

    nStencils_ = acceptors.size();
}


void Foam::MarineOversetCellState::readAndValidate()
{
    IOobject zoneObject
    (
        "zoneID",
        Time::timeName(mesh_.time().value()),
        mesh_,
        IOobject::MUST_READ,
        IOobject::NO_WRITE,
        false
    );

    Foam::volScalarField zoneField(zoneObject, mesh_);

    label maximumZone = -1;
    forAll(zoneField, celli)
    {
        const scalar value = zoneField[celli];
        const label zone = label(value + (value >= 0 ? 0.5 : -0.5));
        if (mag(value - scalar(zone)) > SMALL || zone < 0)
        {
            FatalErrorInFunction
                << "zoneID must contain finite non-negative integer IDs; "
                << "cell " << celli << " contains " << value << nl
                << exit(FatalError);
        }
        zoneId_[celli] = zone;
        maximumZone = max(maximumZone, zone);
    }

    for (label zone = 0; zone <= maximumZone; ++zone)
    {
        bool found = false;
        forAll(zoneId_, celli)
        {
            if (zoneId_[celli] == zone)
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            FatalErrorInFunction
                << "zoneID IDs must be consecutive; missing zone "
                << zone << nl
                << exit(FatalError);
        }
    }

    IOobject statusObject
    (
        "oversetCellStatus",
        Time::timeName(mesh_.time().value()),
        mesh_,
        IOobject::READ_IF_PRESENT,
        IOobject::NO_WRITE,
        false
    );

    if (statusObject.headerOk())
    {
        Foam::volScalarField statusField(statusObject, mesh_);
        forAll(statusField, celli)
        {
            const scalar value = statusField[celli];
            const label state = label(value + (value >= 0 ? 0.5 : -0.5));
            if (mag(value - scalar(state)) > SMALL || state < calculated || state > hole)
            {
                FatalErrorInFunction
                    << "oversetCellStatus must be 0, 1 or 2; cell "
                    << celli << " contains " << value << nl
                    << exit(FatalError);
            }
            status_[celli] = state;
        }
    }
}

Foam::label Foam::MarineOversetCellState::nZones() const
{
    label result = 0;
    forAll(zoneId_, celli)
    {
        result = max(result, zoneId_[celli] + 1);
    }
    return result;
}

Foam::label Foam::MarineOversetCellState::nCalculated() const
{
    label result = 0;
    forAll(status_, celli)
    {
        result += status_[celli] == calculated;
    }
    return result;
}

Foam::label Foam::MarineOversetCellState::nInterpolated() const
{
    label result = 0;
    forAll(status_, celli)
    {
        result += status_[celli] == interpolated;
    }
    return result;
}

Foam::label Foam::MarineOversetCellState::nHoles() const
{
    label result = 0;
    forAll(status_, celli)
    {
        result += status_[celli] == hole;
    }
    return result;
}

// ************************************************************************* //
