#include "vofFragmentInjection.H"

namespace Foam
{

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
    cloudPhase_(this->coeffDict().template lookupOrDefault<word>("phase", "water")),
    prepared_(false),
    emitted_(false)
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
    cloudPhase_(other.cloudPhase_),
    fragments_(other.fragments_),
    coordinates_(other.coordinates_),
    cells_(other.cells_),
    tetFaces_(other.tetFaces_),
    tetPts_(other.tetPts_),
    diameters_(other.diameters_),
    prepared_(other.prepared_),
    emitted_(other.emitted_)
{}


template<class CloudType>
void vofFragmentInjection<CloudType>::prepare()
{
    if (prepared_)
    {
        return;
    }

    fragments_ = vofFragmentTransition::detect
    (
        alpha_,
        U_,
        threshold_,
        minCells_,
        minVolume_
    );

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
    prepare();
    return emitted_ ? 0 : fragments_.size();
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
        totalMass += rho_[cells_[fragmentI]]*fragments_[fragmentI].volume;
    }
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
}

} // End namespace Foam
