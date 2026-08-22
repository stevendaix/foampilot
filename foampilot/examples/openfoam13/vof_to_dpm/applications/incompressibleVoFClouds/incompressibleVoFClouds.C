/*---------------------------------------------------------------------------*\
  OpenFOAM 13 incompressible VoF / parcel-cloud coupling model
\*---------------------------------------------------------------------------*/
#include "incompressibleVoFClouds.H"
#include "addToRunTimeSelectionTable.H"

namespace Foam
{
namespace fv
{
    defineTypeNameAndDebug(incompressibleVoFClouds, 0);
    addToRunTimeSelectionTable(fvModel, incompressibleVoFClouds, dictionary);
}
}

Foam::fv::incompressibleVoFClouds::incompressibleVoFClouds
(
    const word& sourceName,
    const word& modelType,
    const fvMesh& mesh,
    const dictionary& dict
)
:
    fvModel(sourceName, modelType, mesh, dict),
    mixture_
    (
        mesh.lookupObject<incompressibleTwoPhaseVoFMixture>
        (
            dict.lookupOrDefault<word>("mixture", "phaseProperties")
        )
    ),
    g_
    (
        IOobject
        (
            "g",
            mesh.time().constant(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE
        ),
        dimensionedVector(dimAcceleration, Zero)
    ),
    cloudNames_
    (
        dict.lookupOrDefault<wordList>
        (
            "clouds",
            parcelCloudList::defaultCloudNames
        )
    ),
    mu_
    (
        IOobject
        (
            "mu",
            mesh.time().name(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimDynamicViscosity, 0)
    ),
    clouds_
    (
        cloudNames_,
        mixture_.rho(),
        mesh.lookupObject<volVectorField>("U"),
        mu_,
        g_
    ),
    curTimeIndex_(-1)
{
    mu_ = mixture_.rho()*mixture_.nu();
}

Foam::wordList Foam::fv::incompressibleVoFClouds::addSupFields() const
{
    return wordList({"U"});
}

void Foam::fv::incompressibleVoFClouds::correct()
{
    if (curTimeIndex_ == mesh().time().timeIndex())
    {
        return;
    }

    mu_ = mixture_.rho()*mixture_.nu();
    clouds_.evolve();
    curTimeIndex_ = mesh().time().timeIndex();
}

void Foam::fv::incompressibleVoFClouds::addSup
(
    const volScalarField&,
    fvMatrix<scalar>&
) const
{
    // Parcel mass is currently supplied by the injection model.  Alpha
    // consumption is deliberately not hidden in this hook: a future VOF
    // transfer model must provide a bounded, conservative source.
}

void Foam::fv::incompressibleVoFClouds::addSup
(
    const volScalarField& rho,
    const volVectorField& U,
    fvMatrix<vector>& eqn
) const
{
    if (U.name() != "U" || &rho != &mixture_.rho())
    {
        FatalErrorInFunction
            << "incompressibleVoFClouds supports only the mixture rho and U fields"
            << exit(FatalError);
    }

    eqn += clouds_.SU(eqn.psi());
}

void Foam::fv::incompressibleVoFClouds::preUpdateMesh()
{
    clouds_.storeGlobalPositions();
}

void Foam::fv::incompressibleVoFClouds::topoChange
(
    const polyTopoChangeMap& map
)
{
    clouds_.topoChange(map);
}

void Foam::fv::incompressibleVoFClouds::mapMesh
(
    const polyMeshMap& map
)
{
    clouds_.mapMesh(map);
}

void Foam::fv::incompressibleVoFClouds::distribute
(
    const polyDistributionMap& map
)
{
    clouds_.distribute(map);
}

bool Foam::fv::incompressibleVoFClouds::movePoints()
{
    return true;
}

// ************************************************************************* //
