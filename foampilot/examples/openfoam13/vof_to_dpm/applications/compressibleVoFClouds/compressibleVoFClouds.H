/* OpenFOAM 13 compressible VoF / parcel-cloud coupling model */
#ifndef compressibleVoFClouds_H
#define compressibleVoFClouds_H

#include "fvModel.H"
#include "parcelCloudList.H"
#include "compressibleTwoPhaseVoFMixture.H"
#include "uniformDimensionedFields.H"
#include "volFields.H"
#include "fvMatrices.H"
#include "vofFragmentTransition.H"

namespace Foam
{
namespace fv
{
namespace compressible
{
class compressibleVoFClouds : public fvModel
{
    const compressibleTwoPhaseVoFMixture& mixture_;
    const uniformDimensionedVectorField g_;
    const wordList cloudNames_;
    const Switch useThermoCloud_;
    mutable volScalarField mu_;
    mutable volScalarField confirmedTransferRate_;
    mutable autoPtr<parcelCloudList> cloudsPtr_;
    mutable volScalarField fragmentMask_;
    mutable volScalarField alphaRhoTransferRate_;
    const Switch consumeAlpha_;
    mutable bool consumptionPending_;
    mutable bool energyTransferPending_;
    mutable bool transitionApplied_;
    const word liquidPhase_;
    const volScalarField& liquidAlpha_;
    const scalar alphaThreshold_;
    const label minCells_;
    const scalar minVolume_;
    const Switch detectFragments_;
    mutable label curTimeIndex_;

public:
    TypeName("compressibleVoFClouds");

    compressibleVoFClouds
    (
        const word& sourceName,
        const word& modelType,
        const fvMesh& mesh,
        const dictionary& dict
    );

    compressibleVoFClouds(const compressibleVoFClouds&) = delete;

    virtual wordList addSupFields() const;

    virtual void correct();

    virtual void addSup
    (
        const volScalarField& alpha,
        const volScalarField& rho,
        fvMatrix<scalar>& eqn
    ) const;

    virtual void addSup
    (
        const volScalarField& alpha,
        const volScalarField& rho,
        const volScalarField& field,
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

    void operator=(const compressibleVoFClouds&) = delete;
};
}
}
}

#endif
