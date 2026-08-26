/*---------------------------------------------------------------------------*\
  OpenFOAM 13 incompressible VoF / parcel-cloud coupling model
\*---------------------------------------------------------------------------*/
#ifndef incompressibleVoFClouds_H
#define incompressibleVoFClouds_H

#include "fvModel.H"
#include "parcelCloudList.H"
#include "incompressibleTwoPhaseVoFMixture.H"
#include "uniformDimensionedFields.H"
#include "volFields.H"
#include "fvMatrices.H"
#include "vofFragmentTransitionManager.H"

namespace Foam
{
namespace fv
{

class incompressibleVoFClouds
:
    public fvModel
{
    const incompressibleTwoPhaseVoFMixture& mixture_;
    const uniformDimensionedVectorField g_;
    const wordList cloudNames_;
    const scalarList speciesFractions_;
    mutable volScalarField mu_;
    mutable volScalarField confirmedTransferRate_;
    mutable parcelCloudList clouds_;
    mutable volScalarField fragmentMask_;
    mutable volScalarField alphaConsumptionRate_;
    const Switch consumeAlpha_;
    mutable bool consumptionPending_;
    mutable bool transitionApplied_;
    const scalar alphaThreshold_;
    const label minCells_;
    const scalar minVolume_;
    const Switch detectFragments_;
    mutable label curTimeIndex_;
    mutable autoPtr<vofFragmentTransitionManager> transitionManagerPtr_;
    mutable vofFragmentBatch transitionBatch_;

    void publishLocalBatchForEachCloud(const label timeIndex) const;

    void commitDirectLocalParcels
    (
        List<vofParcelConfirmation>& localConfirmations
    ) const;

    List<vofParcelConfirmation> collectLocalInjectionConfirmations
    (
        const label timeIndex
    ) const;

    void applyConfirmedResults
    (
        const List<vofParcelConfirmation>& results
    ) const;

public:
    TypeName("incompressibleVoFClouds");

    incompressibleVoFClouds
    (
        const word& sourceName,
        const word& modelType,
        const fvMesh& mesh,
        const dictionary& dict
    );

    incompressibleVoFClouds(const incompressibleVoFClouds&) = delete;

    virtual wordList addSupFields() const;

    virtual void correct();

    virtual void addSup
    (
        const volScalarField& alpha,
        fvMatrix<scalar>& eqn
    ) const;

    virtual void addSup
    (
        const volScalarField& rho,
        const volVectorField& U,
        fvMatrix<vector>& eqn
    ) const;

    virtual void preUpdateMesh();
    virtual void topoChange(const polyTopoChangeMap&);
    virtual void mapMesh(const polyMeshMap&);
    virtual void distribute(const polyDistributionMap&);
    virtual bool movePoints();

    void operator=(const incompressibleVoFClouds&) = delete;
};

} // End namespace fv
} // End namespace Foam

#endif
