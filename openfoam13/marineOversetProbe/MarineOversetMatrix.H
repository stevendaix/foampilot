#ifndef MarineOversetMatrix_H
#define MarineOversetMatrix_H

#include "fvMatrix.H"
#include "MarineOversetCellState.H"
#include "MarineOversetInterpolation.H"

namespace Foam
{

class MarineOversetMatrix
{
    const MarineOversetCellState& state_;
    const MarineOversetInterpolation& interpolation_;

public:
    MarineOversetMatrix
    (
        const MarineOversetCellState& state,
        const MarineOversetInterpolation& interpolation
    );

    void applyScalar(fvMatrix<scalar>& matrix) const;
    void applyVector(fvMatrix<vector>& matrix) const;
};

}

#endif
