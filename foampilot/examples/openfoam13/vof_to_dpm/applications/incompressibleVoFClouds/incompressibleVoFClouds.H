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
#include "vofFragmentTransition.H"

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
    mutable volScalarField mu_;
    mutable parcelCloudList clouds_;
    const scalar alphaThreshold_;
    const label minCells_;
    const scalar minVolume_;
    const Switch detectFragments_;
    mutable label curTimeIndex_;

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
