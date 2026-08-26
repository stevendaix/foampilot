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
    speciesFractions_
    (
        dict.lookupOrDefault<scalarList>
        (
            "speciesMassFractions",
            scalarList()
        )
    ),
    mu_
    (
        IOobject
        (
            sourceName + ":mu",
            mesh.time().name(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimDynamicViscosity, 0)
    ),
    confirmedTransferRate_
    (
        IOobject
        (
            sourceName + ":vofConfirmedTransferRate",
            mesh.time().name(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimless/dimTime, 0)
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
            sourceName + ":vofFragmentMask",
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
            sourceName + ":vofAlphaConsumptionRate",
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
    curTimeIndex_(-1),
    transitionManagerPtr_
    (
        new vofFragmentTransitionManager
        (
            mesh,
            mixture_.alpha1(),
            mesh.lookupObject<volVectorField>("U"),
            mixture_.rho(),
            alphaThreshold_,
            minCells_,
            minVolume_,
            dict.lookupOrDefault<scalar>("rhoLiquid", 0),
            cloudNames_[0],
            mixture_.alpha1().name(),
            speciesFractions_
        )
    ),
    transitionBatch_()
{
    mu_ = mixture_.rho()*mixture_.nu();
}

Foam::wordList Foam::fv::incompressibleVoFClouds::addSupFields() const
{
    return wordList
    ({mixture_.alpha1().name(), mixture_.alpha2().name(), "U"});
}

Foam::List<Foam::vofParcelConfirmation>
Foam::fv::incompressibleVoFClouds::collectLocalInjectionConfirmations
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


void Foam::fv::incompressibleVoFClouds::applyConfirmedResults
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


void Foam::fv::incompressibleVoFClouds::publishLocalBatchForEachCloud
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


void Foam::fv::incompressibleVoFClouds::commitDirectLocalParcels
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
        data.speciesMassFractions = speciesFractions_;
        data.Cp = -GREAT;

        const bool committed =
            clouds_.commitDirect(cloudNames_[0], data, -1);

        vofParcelConfirmation confirmation;
        confirmation.cloudName = cloudNames_[0];
        confirmation.alphaFieldName = mixture_.alpha1().name();
        confirmation.fragmentId = fragment.id;
        confirmation.ownerProc = Pstream::myProcNo();
        confirmation.parcelsAdded = committed ? 1 : 0;
        confirmation.massAdded = committed ? fragment.mass : scalar(0);
        confirmation.expectedMass = fragment.mass;
        confirmation.speciesMassAdded.setSize(speciesFractions_.size(), 0);
        confirmation.expectedSpeciesMass.setSize(speciesFractions_.size(), 0);
        forAll(speciesFractions_, speciesI)
        {
            confirmation.speciesMassAdded[speciesI] =
                committed ? fragment.mass*speciesFractions_[speciesI] : scalar(0);
            confirmation.expectedSpeciesMass[speciesI] =
                fragment.mass*speciesFractions_[speciesI];
        }
        confirmation.success = committed;
        localConfirmations.append(confirmation);

        Info<< "VOF direct commit cloud=" << confirmation.cloudName
            << " fragmentId=" << fragment.id
            << " success=" << committed
            << " mass=" << confirmation.massAdded
            << " speciesMass=" << confirmation.speciesMassAdded << nl;
    }
}


void Foam::fv::incompressibleVoFClouds::correct()
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
    transitionBatch_ = vofFragmentBatch();
    transitionBatch_.timeIndex = mesh().time().timeIndex();
    fragmentMask_.internalFieldRef() = scalar(0);

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
            forAll(fragment.localCells, cellI)
            {
                fragmentMask_[fragment.localCells[cellI]] = scalar(1);
            }
        }

        Info<< "VOF fragments detected globally: "
            << transitionBatch_.fragments.size()
            << ", local convertible volume: " << detectedVolume << nl;
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
    applyConfirmedResults(localResults);

    consumptionPending_ =
        consumeAlpha_
     && gMax(confirmedTransferRate_.internalField()) > SMALL;
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
            eqn += -fvm::Sp(confirmedTransferRate_, eqn.psi());
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
                    confirmedTransferRate_[celli]
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
