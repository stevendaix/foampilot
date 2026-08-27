#include "vofFragmentInjection.H"
#include "Pstream.H"

namespace Foam
{

namespace
{

inline void vofInjectionTrace
(
    const word& cloudName,
    const word& phase,
    const label timeIndex,
    const string& message
)
{
    Pout<< "[vofFragmentInjection] rank=" << Pstream::myProcNo()
        << " cloud=" << cloudName
        << " phase=" << phase
        << " timeIndex=" << timeIndex
        << " " << message << nl << flush;
}

template<class Parcel>
auto setInjectedTemperature(Parcel& parcel, const scalar T, int)
-> decltype(parcel.T(), void())
{
    parcel.T() = T;
}

template<class Parcel>
void setInjectedTemperature(Parcel&, const scalar, long)
{}
}


template<class CloudType>

vofFragmentInjection<CloudType>::vofFragmentInjection
(
    const dictionary& dict,
    CloudType& owner,
    const word& modelName
)
:
    InjectionModel<CloudType>(dict, owner, modelName, typeName),
    alpha_
    (
        owner.db().template lookupObject<volScalarField>
        (
            this->coeffDict().template lookupOrDefault<word>
            (
                "alpha",
                "alpha.water"
            )
        )
    ),
    U_(owner.U()),
    rho_(owner.rho()),
    threshold_(this->coeffDict().template lookupOrDefault<scalar>("alphaThreshold", 0.5)),
    minCells_(this->coeffDict().template lookupOrDefault<label>("minCells", 1)),
    minVolume_(this->coeffDict().template lookupOrDefault<scalar>("minVolume", 0)),
    rhoLiquid_(this->coeffDict().template lookupOrDefault<scalar>("rhoLiquid", -1)),
    cloudPhase_(this->coeffDict().template lookupOrDefault<word>("phase", "water")),
    prepared_(false),
    emitted_(false),
    lastTimeIndex_(-1),
    expectedMass_(0),
    injectedIds_(),
    injectedCellSets_(),
    confirmationStoreName_("vofConfirmations." + owner.name())
{}


template<class CloudType>
vofFragmentInjection<CloudType>::vofFragmentInjection
(
    const vofFragmentInjection<CloudType>& other
)
:
    InjectionModel<CloudType>(other),
    alpha_(other.alpha_),
    U_(other.U_),
    rho_(other.rho_),
    threshold_(other.threshold_),
    minCells_(other.minCells_),
    minVolume_(other.minVolume_),
    rhoLiquid_(other.rhoLiquid_),
    cloudPhase_(other.cloudPhase_),
    fragments_(other.fragments_),
    coordinates_(other.coordinates_),
    cells_(other.cells_),
    tetFaces_(other.tetFaces_),
    tetPts_(other.tetPts_),
    diameters_(other.diameters_),
    prepared_(other.prepared_),
    emitted_(other.emitted_),
    lastTimeIndex_(other.lastTimeIndex_),
    expectedMass_(other.expectedMass_),
    injectedIds_(other.injectedIds_),
    injectedCellSets_(other.injectedCellSets_),
    confirmationStoreName_(other.confirmationStoreName_)
{}


template<class CloudType>
void vofFragmentInjection<CloudType>::prepare()
{
    const label timeIndex = this->owner().db().time().timeIndex();

    vofInjectionTrace
    (
        this->owner().name(),
        "prepare.begin",
        timeIndex,
        "prepared=" + name(prepared_)
      + " lastTimeIndex=" + name(lastTimeIndex_)
      + " injectedIds=" + name(injectedIds_.size())
    );

    if (prepared_ && lastTimeIndex_ == timeIndex)
    {
        vofInjectionTrace
        (
            this->owner().name(),
            "prepare.cached-return",
            timeIndex,
            "fragments=" + name(fragments_.size())
        );
        return;
    }

    const word batchName =
        "vofLocalTransitionBatch." + this->owner().name();

    fragments_.clear();
    prepared_ = false;
    emitted_ = false;

    if
    (
        !this->owner().db().template foundObject
        <vofLocalTransitionBatch>(batchName)
    )
    {
        lastTimeIndex_ = timeIndex;
        prepared_ = true;

        vofInjectionTrace
        (
            this->owner().name(),
            "prepare.no-batch-return",
            timeIndex,
            "batch=" + batchName
        );
        return;
    }

    const vofLocalTransitionBatch& published =
        this->owner().db().template lookupObject
        <vofLocalTransitionBatch>(batchName);

    vofInjectionTrace
    (
        this->owner().name(),
        "prepare.batch-found",
        timeIndex,
        "publishedTimeIndex=" + name(published.timeIndex())
      + " fragments=" + name(published.fragments().size())
    );

    if (published.timeIndex() != timeIndex)
    {
        FatalErrorInFunction
            << "Stale local transition batch for cloud "
            << this->owner().name()
            << ": batch timeIndex=" << published.timeIndex()
            << ", current timeIndex=" << timeIndex
            << exit(FatalError);
    }

    DynamicList<std::uint64_t> activeIds(published.fragments().size());

    forAll(published.fragments(), fragmentI)
    {
        const vofGlobalFragment& global =
            published.fragments()[fragmentI];

        if (global.ownerProc != Pstream::myProcNo())
        {
            continue;
        }

        if (global.localCells.empty())
        {
            FatalErrorInFunction
                << "Owner fragment " << global.id
                << " has no localCells"
                << exit(FatalError);
        }

        vofFragmentTransitionRecord local;
        local.id = global.id;
        local.cells = global.localCells;
        local.globalCells = global.globalCells;
        local.volume = global.volume;
        local.centroid = global.centroid;
        local.velocity = global.velocity;

        activeIds.append(local.id);

        vofInjectionTrace
        (
            this->owner().name(),
            "prepare.fragment",
            timeIndex,
            "id=" + name(local.id)
          + " globalCells=" + name(local.globalCells.size())
          + " localCells=" + name(local.cells.size())
        );

        bool alreadyInjected = false;
        forAll(injectedIds_, idI)
        {
            if (injectedIds_[idI] == local.id)
            {
                alreadyInjected = true;
                break;
            }
        }

        if (!alreadyInjected)
        {
            fragments_.append(local);
        }
    }

    coordinates_.setSize(fragments_.size());
    cells_.setSize(fragments_.size(), -1);
    tetFaces_.setSize(fragments_.size(), -1);
    tetPts_.setSize(fragments_.size(), -1);
    diameters_.setSize(fragments_.size(), 0);

    vofInjectionTrace
    (
        this->owner().name(),
        "prepare.before-mesh-search",
        timeIndex,
        "ownedFragments=" + name(fragments_.size())
    );

    const meshSearch& searchEngine = meshSearch::New(this->owner().mesh());
    forAll(fragments_, fragmentI)
    {
        this->findCellAtPosition
        (
            searchEngine,
            fragments_[fragmentI].centroid,
            coordinates_[fragmentI],
            cells_[fragmentI],
            tetFaces_[fragmentI],
            tetPts_[fragmentI]
        );

        bool cellBelongsToFragment = false;
        forAll(fragments_[fragmentI].cells, cellI)
        {
            if (fragments_[fragmentI].cells[cellI] == cells_[fragmentI])
            {
                cellBelongsToFragment = true;
                break;
            }
        }

        if (!cellBelongsToFragment)
        {
            FatalErrorInFunction
                << "Centroid of fragment " << fragments_[fragmentI].id
                << " was located outside its localCells"
                << exit(FatalError);
        }

        diameters_[fragmentI] =
            pow
            (
                6*fragments_[fragmentI].volume
               /constant::mathematical::pi,
                scalar(1)/3
            );
    }

    lastTimeIndex_ = timeIndex;
    prepared_ = true;

    vofInjectionTrace
    (
        this->owner().name(),
        "prepare.end",
        timeIndex,
        "fragments=" + name(fragments_.size())
      + " expectedMass=" + name(expectedMass_)
    );
}


template<class CloudType>
vofLocalConfirmationStore&
vofFragmentInjection<CloudType>::confirmationStore() const
{
    const objectRegistry& db = this->owner().db();

    if (!db.template foundObject<vofLocalConfirmationStore>
        (confirmationStoreName_))
    {
        autoPtr<vofLocalConfirmationStore> store
        (
            new vofLocalConfirmationStore(db, confirmationStoreName_)
        );
        regIOobject::store(store.ptr());
    }

    return const_cast<objectRegistry&>(db).template lookupObjectRef
    <vofLocalConfirmationStore>(confirmationStoreName_);
}


template<class CloudType>
const DynamicList<vofParcelConfirmation>&
vofFragmentInjection<CloudType>::confirmations() const
{
    return confirmationStore().confirmations();
}


template<class CloudType>
void vofFragmentInjection<CloudType>::clearConfirmations() const
{
    confirmationStore().clear
    (
        this->owner().db().time().timeIndex()
    );
}


template<class CloudType>
void vofFragmentInjection<CloudType>::postInject
(
    const label parcelsAdded,
    const scalar massAdded,
    typename CloudType::parcelType::trackingData& td
)
{
    InjectionModel<CloudType>::postInject(parcelsAdded, massAdded, td);

    const scalar massTol = 1e-8*max(mag(expectedMass_), scalar(1));
    const bool massConfirmed =
        expectedMass_ <= SMALL
      ? massAdded > SMALL
      : mag(massAdded - expectedMass_) <= massTol;
    const bool confirmed =
        !fragments_.empty()
     && parcelsAdded == fragments_.size()
     && massConfirmed;

    vofLocalConfirmationStore& store = confirmationStore();
    forAll(fragments_, fragmentI)
    {
        vofParcelConfirmation confirmation;
        confirmation.fragmentId = fragments_[fragmentI].id;
        confirmation.ownerProc = Pstream::myProcNo();
        confirmation.parcelsAdded = confirmed ? 1 : 0;
        confirmation.expectedMass =
            rhoLiquid_ > 0
          ? rhoLiquid_*fragments_[fragmentI].volume
          : rho_[cells_[fragmentI]]*fragments_[fragmentI].volume;
        confirmation.massAdded =
            confirmed ? confirmation.expectedMass : scalar(0);
        confirmation.success = confirmed;
        store.append(confirmation);
    }

    if (!confirmed)
    {
        emitted_ = false;
        prepared_ = false;
        expectedMass_ = 0;
        return;
    }

    forAll(fragments_, fragmentI)
    {
        injectedIds_.append(fragments_[fragmentI].id);
        injectedCellSets_.append(fragments_[fragmentI].cells);
    }

    expectedMass_ = 0;
}


template<class CloudType>
void vofFragmentInjection<CloudType>::topoChange()
{
    prepared_ = false;
    prepare();
}


template<class CloudType>
Foam::scalar vofFragmentInjection<CloudType>::timeEnd() const
{
    return vGreat;
}


template<class CloudType>
Foam::scalar vofFragmentInjection<CloudType>::nParcelsToInject
(
    const scalar,
    const scalar
)
{
    const label timeIndex = this->owner().db().time().timeIndex();

    vofInjectionTrace
    (
        this->owner().name(),
        "nParcelsToInject.begin",
        timeIndex,
        "prepared=" + name(prepared_)
      + " emitted=" + name(emitted_)
      + " lastTimeIndex=" + name(lastTimeIndex_)
    );
    if (timeIndex != lastTimeIndex_)
    {
        lastTimeIndex_ = timeIndex;
        prepared_ = false;
        emitted_ = false;
    }

    vofInjectionTrace
    (
        this->owner().name(),
        "nParcelsToInject.before-prepare",
        timeIndex,
        "calling prepare"
    );

    prepare();

    vofInjectionTrace
    (
        this->owner().name(),
        "nParcelsToInject.after-prepare",
        timeIndex,
        "fragments=" + name(fragments_.size())
    );

    // A spray fragment may appear only after the liquid jet has entered
    // the domain.  Do not cache an empty first scan forever.
    const scalar nParcels = emitted_ ? 0 : fragments_.size();
    if (!emitted_ && fragments_.empty())
    {
        prepared_ = false;
    }
    vofInjectionTrace
    (
        this->owner().name(),
        "nParcelsToInject.end",
        timeIndex,
        "return=" + name(nParcels)
      + " emitted=" + name(emitted_)
      + " fragments=" + name(fragments_.size())
    );

    return nParcels;
}


template<class CloudType>
Foam::scalar vofFragmentInjection<CloudType>::massToInject
(
    const scalar,
    const scalar
)
{
    prepare();
    if (emitted_)
    {
        return 0;
    }

    scalar totalMass = 0;
    forAll(fragments_, fragmentI)
    {
        const scalar rhoCell =
            rhoLiquid_ > 0 ? rhoLiquid_ : rho_[cells_[fragmentI]];
        totalMass += rhoCell*fragments_[fragmentI].volume;
    }
    expectedMass_ = totalMass;
    return totalMass;
}


template<class CloudType>
void vofFragmentInjection<CloudType>::setPositionAndCell
(
    const meshSearch&,
    const label parcelI,
    const label,
    const scalar,
    barycentric& coordinates,
    label& celli,
    label& tetFacei,
    label& tetPti,
    label& facei
)
{
    prepare();
    coordinates = coordinates_[parcelI];
    celli = cells_[parcelI];
    tetFacei = tetFaces_[parcelI];
    tetPti = tetPts_[parcelI];
    facei = -1;
    if (parcelI == fragments_.size() - 1)
    {
        emitted_ = true;
    }
}


template<class CloudType>
bool vofFragmentInjection<CloudType>::fullyDescribed() const
{
    return true;
}


template<class CloudType>
void vofFragmentInjection<CloudType>::setProperties
(
    const label parcelI,
    const label,
    const scalar,
    typename CloudType::parcelType::trackingData&,
    typename CloudType::parcelType& parcel
)
{
    parcel.U() = fragments_[parcelI].velocity;
    parcel.d() = diameters_[parcelI];

    if (this->owner().db().template foundObject<volScalarField>("T"))
    {
        const volScalarField& T =
            this->owner().db().template lookupObject<volScalarField>("T");
        setInjectedTemperature(parcel, T[cells_[parcelI]], 0);
    }
}

} // End namespace Foam
