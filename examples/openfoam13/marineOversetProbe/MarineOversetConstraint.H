#ifndef MarineOversetConstraint_H
#define MarineOversetConstraint_H

#include "fvConstraint.H"
#include "MarineOversetCellState.H"
#include "MarineOversetInterpolation.H"
#include "MarineOversetMatrix.H"
#include "MarineInterMeshStencilState.H"
#include "MarineInterMeshMatrix.H"
#include "wordList.H"
#include "autoPtr.H"

namespace Foam
{
namespace fv
{
class marineOversetConstraint : public fvConstraint
{
    wordList fields_;
    autoPtr<MarineOversetCellState> state_;
    autoPtr<MarineOversetInterpolation> interpolation_;
    autoPtr<MarineOversetMatrix> matrixOperator_;
    autoPtr<MarineInterMeshStencilState> interMeshState_;
    autoPtr<MarineInterMeshMatrix> interMeshOperator_;
    const fvMesh* donorMesh_;

    bool selected(const word& fieldName) const;

public:
    TypeName("marineOversetConstraint");

    marineOversetConstraint
    (
        const word& name,
        const word& modelType,
        const fvMesh& mesh,
        const dictionary& dict
    );

    virtual ~marineOversetConstraint() {}

    virtual wordList constrainedFields() const { return fields_; }

    virtual bool constrain
    (
        fvMatrix<scalar>& eqn,
        const word& fieldName
    ) const;

    virtual bool constrain
    (
        fvMatrix<vector>& eqn,
        const word& fieldName
    ) const;

    virtual bool movePoints() { return true; }
    virtual void topoChange(const polyTopoChangeMap&) {}
    virtual void mapMesh(const polyMeshMap&) {}
    virtual void distribute(const polyDistributionMap&) {}
};
}
}

#endif
