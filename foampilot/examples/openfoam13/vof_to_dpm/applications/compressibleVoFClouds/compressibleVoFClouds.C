/* OpenFOAM 13 compressible VoF / parcel-cloud coupling model */
#include "compressibleVoFClouds.H"
#include "addToRunTimeSelectionTable.H"

namespace Foam
{
namespace fv
{
namespace compressible
{
    defineTypeNameAndDebug(compressibleVoFClouds, 0);
    addToRunTimeSelectionTable(fvModel, compressibleVoFClouds, dictionary);
}
}
}

Foam::fv::compressible::compressibleVoFClouds::compressibleVoFClouds
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
        mesh.lookupObject<compressibleTwoPhaseVoFMixture>
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
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
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
        mixture_.rho()*mixture_.nu()
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
    alphaRhoTransferRate_
    (
        IOobject
        (
            "vofAlphaRhoTransferRate",
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
    liquidPhase_
    (
        dict.lookupOrDefault<word>
        (
            "liquidPhase",
            mixture_.alpha1().group()
        )
    ),
    liquidAlpha_
    (
        liquidPhase_ == mixture_.alpha1().group()
      ? mixture_.alpha1()
      : mixture_.alpha2()
    ),
    alphaThreshold_(dict.lookupOrDefault<scalar>("alphaThreshold", 0.5)),
    minCells_(dict.lookupOrDefault<label>("minCells", 1)),
    minVolume_(dict.lookupOrDefault<scalar>("minVolume", 0)),
    detectFragments_(dict.lookupOrDefault<Switch>("detectFragments", true)),
    curTimeIndex_(-1)
{
    if
    (
        liquidPhase_ != mixture_.alpha1().group()
     && liquidPhase_ != mixture_.alpha2().group()
    )
    {
        FatalErrorInFunction
            << "liquidPhase must be one of "
            << mixture_.alpha1().group() << " or "
            << mixture_.alpha2().group() << exit(FatalError);
    }
}

Foam::wordList
Foam::fv::compressible::compressibleVoFClouds::addSupFields() const
{
    return
    {
        mixture_.rho1().name(),
        mixture_.rho2().name(),
        "U"
    };
}

void Foam::fv::compressible::compressibleVoFClouds::addSup
(
    const volScalarField& alpha,
    const volScalarField& rho,
    fvMatrix<scalar>& eqn
) const
{
    if
    (
        (&alpha != &mixture_.alpha1() || &rho != &mixture_.rho1())
     && (&alpha != &mixture_.alpha2() || &rho != &mixture_.rho2())
    )
    {
        FatalErrorInFunction
            << "compressibleVoFClouds received an unsupported alpha/rho pair"
            << exit(FatalError);
    }

    if (consumptionPending_ && &eqn.psi() == &alpha)
    {
        tmp<volScalarField::Internal> tSu
        (
            volScalarField::Internal::New
            (
                "vofAlphaRhoTransfer",
                mesh(),
                                    dimensionedScalar(dimMass/dimVolume/dimTime, 0)

            )
        );
        const bool liquidIsPhase1 = liquidPhase_ == mixture_.alpha1().group();
        const scalar sign =
            (&alpha == &mixture_.alpha1()) == liquidIsPhase1 ? -1 : 1;
        forAll(tSu(), celli)
        {
            tSu.ref()[celli] =
                sign*alphaRhoTransferRate_[celli]
               *liquidAlpha_[celli]
               *rho[celli]
                                  ;

        }
        eqn += tSu;
        if (&alpha == &mixture_.alpha2())
        {
            consumptionPending_ = false;
        }
        Info<< "Applied compressible alphaRho transfer to "
            << alpha.name() << nl;
    }
}

void Foam::fv::compressible::compressibleVoFClouds::correct()
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
                liquidAlpha_,
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
        if (consumeAlpha_ && !transitionApplied_)
        {
            alphaRhoTransferRate_ =
                dimensionedScalar(dimless/dimTime, scalar(0));
            const scalar rate = 1/mesh().time().deltaTValue();
            forAll(fragments, fragmentI)
            {
                const labelList& cells = fragments[fragmentI].cells;
                forAll(cells, cellI)
                {
                    alphaRhoTransferRate_[cells[cellI]] = rate;
                }
            }
            consumptionPending_ = true;
            transitionApplied_ = true;
            Info<< "Compressible alphaRho transfer armed for "
                << detectedVolume << " m3" << nl;
        }
        forAll(fragments, fragmentI)
        {
            Info<< "  fragment " << fragmentI
                << " id " << fragments[fragmentI].id
                << " volume " << fragments[fragmentI].volume << nl;
        }
    }
    clouds_.evolve();
    curTimeIndex_ = mesh().time().timeIndex();
}

void Foam::fv::compressible::compressibleVoFClouds::addSup
(
    const volScalarField& rho,
    const volVectorField& U,
    fvMatrix<vector>& eqn
) const
{
    if (U.name() != "U" || &rho != &mixture_.rho())
    {
        FatalErrorInFunction
            << "compressibleVoFClouds supports only the mixture rho and U fields"
            << exit(FatalError);
    }

    eqn += clouds_.SU(eqn.psi());
}

void Foam::fv::compressible::compressibleVoFClouds::preUpdateMesh()
{
    clouds_.storeGlobalPositions();
}

void Foam::fv::compressible::compressibleVoFClouds::topoChange
(
    const polyTopoChangeMap& map
)
{
    clouds_.topoChange(map);
}

void Foam::fv::compressible::compressibleVoFClouds::mapMesh
(
    const polyMeshMap& map
)
{
    clouds_.mapMesh(map);
}

void Foam::fv::compressible::compressibleVoFClouds::distribute
(
    const polyDistributionMap& map
)
{
    clouds_.distribute(map);
}

bool Foam::fv::compressible::compressibleVoFClouds::movePoints()
{
    return true;
}

// ************************************************************************* //
