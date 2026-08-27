#ifndef MarineInterMeshMatrix_H
#define MarineInterMeshMatrix_H

#include "MarineInterMeshStencilState.H"
#include "fvMatrix.H"
#include "volFields.H"

namespace Foam
{
class MarineInterMeshMatrix
{
    const MarineInterMeshStencilState& state_;

public:
    explicit MarineInterMeshMatrix(const MarineInterMeshStencilState& state)
    : state_(state) {}

    void applyScalar
    (
        fvMatrix<scalar>& eqn,
        const volScalarField& donor
    ) const;

    void applyVector
    (
        fvMatrix<vector>& eqn,
        const volVectorField& donor
    ) const;
};
}

#endif
