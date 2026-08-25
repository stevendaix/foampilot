#include "vofFragmentInjection.H"

namespace Foam
{

namespace
{

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
    injectedCellSets_()
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
    injectedCellSets_(other.injectedCellSets_)
{}


template<class CloudType>
void vofFragmentInjection<CloudType>::prepare()
{
    if (prepared_)
    {
        return;
    }

    const List<vofFragmentTransitionRecord> detected =
        vofFragmentTransition::detect
        (
            alpha_,
            U_,
            threshold_,
            minCells_,
            minVolume_
        );

    DynamicList<std::uint64_t> activeIds(detected.size());
    forAll(detected, fragmentI)
    {
        activeIds.append(detected[fragmentI].id);
    }
    DynamicList<std::uint64_t> retainedIds(activeIds.size());
    forAll(injectedIds_, idI)
    {
        forAll(activeIds, activeI)
        {
            if (activeIds[activeI] == injectedIds_[idI])
            {
                retainedIds.append(injectedIds_[idI]);
                break;
            }
        }
    }
    injectedIds_.transfer(retainedIds);

    DynamicList<vofFragmentTransitionRecord> fresh(detected.size());
    forAll(detected, fragmentI)
    {
        bool alreadyInjected = false;
        forAll(injectedIds_, idI)
        {
            if (injectedIds_[idI] == detected[fragmentI].id)
            {
                alreadyInjected = true;
                break;
            }
        }
        forAll(injectedCellSets_, setI)
        {
            if (alreadyInjected)
            {
                break;
            }
            forAll(detected[fragmentI].cells, cellI)
            {
                forAll(injectedCellSets_[setI], oldCellI)
                {
                    if (detected[fragmentI].cells[cellI]
                     == injectedCellSets_[setI][oldCellI])
                    {
                        alreadyInjected = true;
                        break;
                    }
                }
                if (alreadyInjected)
                {
                    break;
                }
            }
        }
        if (!alreadyInjected)
        {
            fresh.append(detected[fragmentI]);
        }
    }
    fragments_.transfer(fresh);

    coordinates_.setSize(fragments_.size());
    cells_.setSize(fragments_.size(), -1);
    tetFaces_.setSize(fragments_.size(), -1);
    tetPts_.setSize(fragments_.size(), -1);
    diameters_.setSize(fragments_.size(), 0);

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
        diameters_[fragmentI] =
            pow
            (
                6*fragments_[fragmentI].volume
               /constant::mathematical::pi,
                scalar(1)/3
            );
    }
    prepared_ = true;
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

    if (!confirmed)
    {
        emitted_ = false;
        prepared_ = false;
        expectedMass_ = 0;
        return;
    }

    if (this->owner().db().template foundObject<volScalarField>
        ("vofConfirmedTransferRate"))
    {
        volScalarField& confirmedRate =
            this->owner().db().template lookupObjectRef<volScalarField>
            ("vofConfirmedTransferRate");
        const scalar rate = 1/this->owner().db().time().deltaTValue();
        forAll(fragments_, fragmentI)
        {
            const labelList& cells = fragments_[fragmentI].cells;
            forAll(cells, cellI)
            {
                confirmedRate.internalFieldRef()[cells[cellI]] = rate;
            }
        }
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
    if (timeIndex != lastTimeIndex_)
    {
        lastTimeIndex_ = timeIndex;
        prepared_ = false;
        emitted_ = false;
    }

    prepare();

    // A spray fragment may appear only after the liquid jet has entered
    // the domain.  Do not cache an empty first scan forever.
    const scalar nParcels = emitted_ ? 0 : fragments_.size();
    if (!emitted_ && fragments_.empty())
    {
        prepared_ = false;
    }
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
