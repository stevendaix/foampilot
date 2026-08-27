#ifndef MarineOversetInterpolation_H
#define MarineOversetInterpolation_H

#include "labelList.H"
#include "scalarField.H"
#include "vectorField.H"
#include "List.H"

namespace Foam
{

class MarineOversetInterpolation
{
    List<labelList> donorIndices_;
    List<scalarList> weights_;

    void validate(const label nCells) const;

public:
    MarineOversetInterpolation
    (
        const List<labelList>& donorIndices,
        const List<scalarList>& weights,
        const label nCells
    );

    label size() const { return donorIndices_.size(); }

    scalar interpolateScalar
    (
        const label stencilI,
        const scalarField& donorField
    ) const;

    vector interpolateVector
    (
        const label stencilI,
        const vectorField& donorField
    ) const;
};

}

#endif
