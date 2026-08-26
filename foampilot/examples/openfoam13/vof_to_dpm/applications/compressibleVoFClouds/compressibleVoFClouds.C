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
    useThermoCloud_
    (
        dict.lookupOrDefault<Switch>("thermoCloud", false)
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
    confirmedTransferRate_
    (
        IOobject
        (
            "vofConfirmedTransferRate",
            mesh.time().name(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimless/dimTime, 0)
    ),
    cloudsPtr_
    (
        useThermoCloud_
      ? new parcelCloudList
        (
            cloudNames_,
            mixture_.rho(),
            mesh.lookupObject<volVectorField>("U"),
            g_,
            dict.lookupOrDefault<word>
            (
                "liquidPhase",
                mixture_.alpha1().group()
            ) == mixture_.alpha1().group()
          ? mixture_.thermo1()
          : mixture_.thermo2()
        )
      : new parcelCloudList
        (
            cloudNames_,
            mixture_.rho(),
            mesh.lookupObject<volVectorField>("U"),
            mu_,
            g_
        )
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
    energyTransferPending_(false),
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
    curTimeIndex_(-1),
    transitionManagerPtr_
    (
        new vofFragmentTransitionManager
        (
            mesh,
            liquidAlpha_,
            mesh.lookupObject<volVectorField>("U"),
            mixture_.rho(),
            alphaThreshold_,
            minCells_,
            minVolume_,
            dict.lookupOrDefault<scalar>("rhoLiquid", 0)
        )
    ),
    transitionBatch_()
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
    wordList fields
    {
        mixture_.rho1().name(),
        mixture_.rho2().name(),
        "U"
    };

    if (useThermoCloud_)
    {
        fields.append
        (
            liquidPhase_ == mixture_.alpha1().group()
          ? mixture_.thermo1().he().name()
          : mixture_.thermo2().he().name()
        );
    }

    return fields;
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
                dimensionedScalar(dimDensity/dimTime, 0)
            )
        );
        const bool liquidIsPhase1 = liquidPhase_ == mixture_.alpha1().group();
        const scalar sign =
            (&alpha == &mixture_.alpha1()) == liquidIsPhase1 ? -1 : 1;
        forAll(tSu(), celli)
        {
            tSu.ref()[celli] =
                sign*confirmedTransferRate_[celli]
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

Foam::List<Foam::vofParcelConfirmation>
Foam::fv::compressible::compressibleVoFClouds::collectLocalInjectionConfirmations
(
    const label timeIndex
) const
{
    List<vofParcelConfirmation> result;
    forAll(cloudNames_, cloudI)
    {
        const word name = "vofConfirmations." + cloudNames_[cloudI];
        if (!mesh().foundObject<vofLocalConfirmationStore>(name))
        {
            continue;
        }
        const vofLocalConfirmationStore& store =
            mesh().lookupObject<vofLocalConfirmationStore>(name);
        if (store.timeIndex() != timeIndex)
        {
            FatalErrorInFunction
                << "Stale confirmation store for cloud "
                << cloudNames_[cloudI] << exit(FatalError);
        }
        forAll(store.confirmations(), confirmationI)
        {
            result.append(store.confirmations()[confirmationI]);
        }
    }
    return result;
}


void Foam::fv::compressible::compressibleVoFClouds::applyConfirmedResults
(
    const List<vofParcelConfirmation>& results
) const
{
    const scalar rate = 1/mesh().time().deltaTValue();
    forAll(results, resultI)
    {
        if (!results[resultI].success)
        {
            continue;
        }
        forAll(transitionBatch_.fragments, fragmentI)
        {
            const vofGlobalFragment& fragment =
                transitionBatch_.fragments[fragmentI];
            if
            (
                fragment.id == results[resultI].fragmentId
             && fragment.ownerProc == Pstream::myProcNo()
            )
            {
                forAll(fragment.localCells, cellI)
                {
                    confirmedTransferRate_
                    [fragment.localCells[cellI]] = rate;
                }
                break;
            }
        }
    }
}


void Foam::fv::compressible::compressibleVoFClouds::publishLocalBatchForEachCloud
(
    const label timeIndex
) const
{
    forAll(cloudNames_, cloudI)
    {
        const word objectName =
            "vofLocalTransitionBatch." + cloudNames_[cloudI];

        if (!mesh().foundObject<vofLocalTransitionBatch>(objectName))
        {
            autoPtr<vofLocalTransitionBatch> object
            (
                new vofLocalTransitionBatch(mesh(), objectName)
            );
            regIOobject::store(object.ptr());
        }

        vofLocalTransitionBatch& object =
            const_cast<fvMesh&>(mesh()).lookupObjectRef
            <vofLocalTransitionBatch>(objectName);
        object.reset(transitionBatch_, timeIndex);

        const word confirmationName =
            "vofConfirmations." + cloudNames_[cloudI];
        if (!mesh().foundObject<vofLocalConfirmationStore>(confirmationName))
        {
            autoPtr<vofLocalConfirmationStore> confirmationStore
            (
                new vofLocalConfirmationStore(mesh(), confirmationName)
            );
            regIOobject::store(confirmationStore.ptr());
        }
        const_cast<fvMesh&>(mesh()).lookupObjectRef
        <vofLocalConfirmationStore>(confirmationName).clear(timeIndex);
    }
}


void Foam::fv::compressible::compressibleVoFClouds::commitDirectLocalParcels
(
    List<vofParcelConfirmation>& localConfirmations
) const
{
    if (cloudNames_.empty())
    {
        return;
    }

    const scalar pi = 3.14159265358979323846;
    const scalar minValue = SMALL;

    forAll(transitionBatch_.fragments, fragmentI)
    {
        const vofGlobalFragment& fragment =
            transitionBatch_.fragments[fragmentI];

        if
        (
            fragment.ownerProc != Pstream::myProcNo()
         || fragment.localCells.empty()
         || fragment.volume <= minValue
         || fragment.mass <= minValue
        )
        {
            continue;
        }

        parcelCloud::directParcelData data;
        data.position = fragment.centroid;
        data.celli = fragment.localCells[0];
        data.diameter = cbrt(6*fragment.volume/pi);
        data.density = fragment.mass/fragment.volume;
        data.velocity = fragment.velocity;
        data.nParticle = 1;
        data.temperature = fragment.temperature;
        data.Cp = -GREAT;

        const bool committed =
            cloudsPtr_().commitDirect(cloudNames_[0], data, -1);

        vofParcelConfirmation confirmation;
        confirmation.fragmentId = fragment.id;
        confirmation.ownerProc = Pstream::myProcNo();
        confirmation.parcelsAdded = committed ? 1 : 0;
        confirmation.massAdded = committed ? fragment.mass : scalar(0);
        confirmation.expectedMass = fragment.mass;
        confirmation.success = committed;
        localConfirmations.append(confirmation);

        Info<< "VOF direct commit fragmentId=" << fragment.id
            << " success=" << committed
            << " mass=" << confirmation.massAdded << nl;
    }
}


void Foam::fv::compressible::compressibleVoFClouds::correct()
{
    if (curTimeIndex_ == mesh().time().timeIndex())
    {
        return;
    }

    mu_ = mixture_.rho()*mixture_.nu();
    transitionApplied_ = false;
    confirmedTransferRate_.internalFieldRef() =
        dimensionedScalar(dimless/dimTime, 0);
    consumptionPending_ = false;
    energyTransferPending_ = false;
    transitionBatch_ = vofFragmentBatch();
    transitionBatch_.timeIndex = mesh().time().timeIndex();
    fragmentMask_.internalFieldRef() = scalar(0);

    scalar detectedMass = 0;
    scalar preparedMass = 0;
    scalar detectedEnthalpy = 0;
    scalar preparedEnthalpy = 0;
    const volScalarField& auditRho =
        liquidPhase_ == mixture_.alpha1().group()
      ? mixture_.rho1()
      : mixture_.rho2();
    const volScalarField& auditHe =
        liquidPhase_ == mixture_.alpha1().group()
      ? mixture_.thermo1().he()
      : mixture_.thermo2().he();

    if (detectFragments_)
    {
        transitionBatch_ =
            transitionManagerPtr_().reconcileMPI
            (
                mesh().time().timeIndex()
            );
        publishLocalBatchForEachCloud
        (
            mesh().time().timeIndex()
        );

        scalar detectedVolume = 0;
        forAll(transitionBatch_.fragments, fragmentI)
        {
            const vofGlobalFragment& fragment =
                transitionBatch_.fragments[fragmentI];

            if (fragment.ownerProc != Pstream::myProcNo())
            {
                continue;
            }

            detectedVolume += fragment.volume;
            detectedMass += fragment.mass;
            preparedMass += fragment.mass;
            forAll(fragment.localCells, cellI)
            {
                const label celli = fragment.localCells[cellI];
                fragmentMask_[celli] = scalar(1);
                const scalar cellMass =
                    fragment.mass*liquidAlpha_[celli]
                   *mesh().V()[celli]
                   /max(fragment.volume, SMALL);
                detectedEnthalpy += cellMass*auditHe[celli];
                preparedEnthalpy += cellMass*auditHe[celli];
            }
        }

        Info<< "VOF fragments detected globally: "
            << transitionBatch_.fragments.size()
            << ", local convertible volume: " << detectedVolume << nl
            << "massDetected=" << detectedMass
            << " massPrepared=" << preparedMass
            << " enthalpyDetected=" << detectedEnthalpy
            << " enthalpyPrepared=" << preparedEnthalpy << nl;
        transitionApplied_ = true;
    }
    else
    {
        publishLocalBatchForEachCloud
        (
            mesh().time().timeIndex()
        );
    }

    List<vofParcelConfirmation> localConfirmations;
    commitDirectLocalParcels(localConfirmations);

    List<vofParcelConfirmation> localResults;
    transitionManagerPtr_().reconcileConfirmationsMPI
    (
        transitionBatch_,
        localConfirmations,
        localResults
    );
    scalar createdMass = 0;
    scalar confirmedMass = 0;
    scalar createdEnthalpy = 0;
    scalar confirmedEnthalpy = 0;

    forAll(localResults, resultI)
    {
        const vofParcelConfirmation& result =
            localResults[resultI];

        if (!result.success)
        {
            continue;
        }

        createdMass += result.massAdded;
        confirmedMass += result.massAdded;

        forAll(transitionBatch_.fragments, fragmentI)
        {
            const vofGlobalFragment& fragment =
                transitionBatch_.fragments[fragmentI];

            if
            (
                fragment.id == result.fragmentId
             && fragment.ownerProc == Pstream::myProcNo()
            )
            {
                forAll(fragment.localCells, cellI)
                {
                    const label celli = fragment.localCells[cellI];
                    const scalar cellMass =
                        fragment.mass*liquidAlpha_[celli]
                       *mesh().V()[celli]
                       /max(fragment.volume, SMALL);
                    createdEnthalpy += cellMass*auditHe[celli];
                    confirmedEnthalpy += cellMass*auditHe[celli];
                }
                break;
            }
        }
    }

    applyConfirmedResults(localResults);

    Info<< "massCreated=" << createdMass
        << " massConfirmed=" << confirmedMass
        << " enthalpyCreated=" << createdEnthalpy
        << " enthalpyConfirmed=" << confirmedEnthalpy << nl;

    consumptionPending_ =
        consumeAlpha_
     && gMax(confirmedTransferRate_.internalField()) > SMALL;
    energyTransferPending_ =
        useThermoCloud_
     && consumptionPending_;
    curTimeIndex_ = mesh().time().timeIndex();
}

void Foam::fv::compressible::compressibleVoFClouds::addSup
(
    const volScalarField& alpha,
    const volScalarField& rho,
    const volScalarField& field,
    fvMatrix<scalar>& eqn
) const
{
    if
    (
        !useThermoCloud_
     || &alpha != &liquidAlpha_
     || &rho != &(
            liquidPhase_ == mixture_.alpha1().group()
          ? mixture_.rho1()
          : mixture_.rho2()
        )
     || &field != &(
            liquidPhase_ == mixture_.alpha1().group()
          ? mixture_.thermo1().he()
          : mixture_.thermo2().he()
        )
    )
    {
        return;
    }

    eqn += cloudsPtr_().Sh(eqn.psi());

    if (energyTransferPending_)
    {
        const volScalarField& liquidRho =
            liquidPhase_ == mixture_.alpha1().group()
          ? mixture_.rho1()
          : mixture_.rho2();
        const volScalarField& liquidHe =
            liquidPhase_ == mixture_.alpha1().group()
          ? mixture_.thermo1().he()
          : mixture_.thermo2().he();

        tmp<volScalarField::Internal> tSu
        (
            volScalarField::Internal::New
            (
                "vofEnthalpyTransfer",
                mesh(),
                dimensionedScalar(dimEnergy/dimVolume/dimTime, 0)
            )
        );

        forAll(tSu(), celli)
        {
            tSu.ref()[celli] =
                -confirmedTransferRate_[celli]
               *liquidAlpha_[celli]
               *liquidRho[celli]
               *liquidHe[celli];
        }

        eqn += tSu;
        energyTransferPending_ = false;
        Info<< "Applied compressible enthalpy transfer to "
            << eqn.psi().name() << nl;
    }
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

    eqn += cloudsPtr_().SU(eqn.psi());
}

void Foam::fv::compressible::compressibleVoFClouds::preUpdateMesh()
{
    cloudsPtr_().storeGlobalPositions();
}

void Foam::fv::compressible::compressibleVoFClouds::topoChange
(
    const polyTopoChangeMap& map
)
{
    cloudsPtr_().topoChange(map);
}

void Foam::fv::compressible::compressibleVoFClouds::mapMesh
(
    const polyMeshMap& map
)
{
    cloudsPtr_().mapMesh(map);
}

void Foam::fv::compressible::compressibleVoFClouds::distribute
(
    const polyDistributionMap& map
)
{
    cloudsPtr_().distribute(map);
}

bool Foam::fv::compressible::compressibleVoFClouds::movePoints()
{
    return true;
}

// ************************************************************************* //
