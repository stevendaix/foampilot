#ifndef vofFragmentInjection_H
#define vofFragmentInjection_H

#include "InjectionModel.H"
#include "vofFragmentTransition.H"
#include "meshSearch.H"
#include "mathematicalConstants.H"

namespace Foam
{

template<class CloudType>
class vofFragmentInjection
:
    public InjectionModel<CloudType>
{
    const volScalarField& alpha_;
    const volVectorField& U_;
    const volScalarField& rho_;
    const scalar threshold_;
    const label minCells_;
    const scalar minVolume_;
    const word cloudPhase_;

    List<vofFragmentTransitionRecord> fragments_;
    List<barycentric> coordinates_;
    labelList cells_;
    labelList tetFaces_;
    labelList tetPts_;
    scalarList diameters_;
    bool prepared_;
    bool emitted_;

    void prepare();

public:
    TypeName("vofFragmentInjection");

    vofFragmentInjection
    (
        const dictionary& dict,
        CloudType& owner,
        const word& modelName
    );

    vofFragmentInjection(const vofFragmentInjection<CloudType>&);

    virtual autoPtr<InjectionModel<CloudType>> clone() const
    {
        return autoPtr<InjectionModel<CloudType>>
        (
            new vofFragmentInjection<CloudType>(*this)
        );
    }

    virtual ~vofFragmentInjection() {}
    virtual void topoChange();
    virtual scalar timeEnd() const;
    virtual scalar nParcelsToInject(const scalar, const scalar);
    virtual scalar massToInject(const scalar, const scalar);

    virtual void setPositionAndCell
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
    );

    virtual void setProperties
    (
        const label parcelI,
        const label,
        const scalar,
        typename CloudType::parcelType::trackingData&,
        typename CloudType::parcelType& parcel
    );

    virtual bool fullyDescribed() const { return false; }
};

} // End namespace Foam

#ifdef NoRepository
    #include "vofFragmentInjection.C"
#endif

#endif
