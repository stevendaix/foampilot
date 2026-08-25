/*---------------------------------------------------------------------------*\
  OpenFOAM 13 incompressible VoF / parcel-cloud coupling model
\*---------------------------------------------------------------------------*/
#include "incompressibleVoFClouds.H"
#include "fvmSup.H"
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
    fragmentMask_
    (
        IOobject
        (
            "vofFragmentMask",
            mesh.time().name(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar(dimless, 0)
    ),
    alphaConsumptionRate_
    (
        IOobject
        (
            "vofAlphaConsumptionRate",
            mesh.time().name(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimless/dimTime, 0)
    ),
    consumeAlpha_(dict.lookupOrDefault<Switch>("consumeAlpha", false)),
    consumptionPending_(false),
    transitionApplied_(false),
    alphaThreshold_(dict.lookupOrDefault<scalar>("alphaThreshold", 0.5)),
    minCells_(dict.lookupOrDefault<label>("minCells", 1)),
    minVolume_(dict.lookupOrDefault<scalar>("minVolume", 0)),
    detectFragments_(dict.lookupOrDefault<Switch>("detectFragments", true)),
    curTimeIndex_(-1)
{
    mu_ = mixture_.rho()*mixture_.nu();
}

Foam::wordList Foam::fv::incompressibleVoFClouds::addSupFields() const
{
    return wordList
    ({mixture_.alpha1().name(), mixture_.alpha2().name(), "U"});
}

void Foam::fv::incompressibleVoFClouds::correct()
{
    if (curTimeIndex_ == mesh().time().timeIndex())
    {
        return;
    }

    mu_ = mixture_.rho()*mixture_.nu();
    if (detectFragments_)
    {
        const volVectorField& U = mesh().lookupObject<volVectorField>("U");
        const List<vofFragmentTransitionRecord> fragments =
            vofFragmentTransition::detect
            (
                mixture_.alpha1(),
                U,
                alphaThreshold_,
                minCells_,
                minVolume_
            );
        scalar detectedVolume = 0;
        fragmentMask_.internalFieldRef() = scalar(0);
        forAll(fragments, fragmentI)
        {
            detectedVolume += fragments[fragmentI].volume;
            const labelList& cells = fragments[fragmentI].cells;
            forAll(cells, cellI)
            {
                fragmentMask_[cells[cellI]] = scalar(1);
            }
        }
        Info<< "VOF fragments detected: " << fragments.size()
            << ", convertible volume: " << detectedVolume << nl;
        forAll(fragments, fragmentI)
        {
            Info<< "  fragment " << fragmentI
                << " id " << fragments[fragmentI].id
                << " volume " << fragments[fragmentI].volume << nl;
        }
        if (consumeAlpha_ && !transitionApplied_)
        {
            alphaConsumptionRate_ =
                dimensionedScalar(dimless/dimTime, scalar(0));
            const scalar rate = 1/mesh().time().deltaTValue();
            forAll(fragments, fragmentI)
            {
                const labelList& cells = fragments[fragmentI].cells;
                forAll(cells, cellI)
                {
                    alphaConsumptionRate_[cells[cellI]] = rate;
                }
            }
            consumptionPending_ = true;
            transitionApplied_ = true;
            Info<< "VOF alpha consumption armed for "
                << detectedVolume << " m3" << nl;
        }
    }
    clouds_.evolve();
    curTimeIndex_ = mesh().time().timeIndex();
}

void Foam::fv::incompressibleVoFClouds::addSup
(
    const volScalarField& alpha,
    fvMatrix<scalar>& eqn
) const
{
    if
    (
        &alpha != &mixture_.alpha1()
     && &alpha != &mixture_.alpha2()
    )
    {
        FatalErrorInFunction
            << "incompressibleVoFClouds supports alpha fields "
            << mixture_.alpha1().name() << " and "
            << mixture_.alpha2().name()
            << exit(FatalError);
    }

    if (consumptionPending_ && &eqn.psi() == &alpha)
    {
        if (&alpha == &mixture_.alpha1())
        {
            eqn += -fvm::Sp(alphaConsumptionRate_, eqn.psi());
        }
        else
        {
            tmp<volScalarField::Internal> tSu
            (
                volScalarField::Internal::New
                (
                    "vofAlphaTransfer",
                    mesh(),
                    dimensionedScalar(dimless/dimTime, 0)
                )
            );
            forAll(tSu(), celli)
            {
                tSu.ref()[celli] =
                    alphaConsumptionRate_[celli]
                   *mixture_.alpha1()[celli]
                   ;
            }
            eqn += tSu;
        }

        if (&alpha == &mixture_.alpha2())
        {
            consumptionPending_ = false;
        }
        Info<< "Applied conservative VOF alpha consumption to "
            << alpha.name() << nl;
    }
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
