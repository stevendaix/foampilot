#include "MarineInterMeshMatrix.H"

void Foam::MarineInterMeshMatrix::applyScalar
(
    fvMatrix<scalar>& eqn,
    const volScalarField& donor
) const
{
    scalarField values(state_.size(), 0);
    forAll(state_.acceptorIndices(), stencilI)
    {
        const labelList& donors = state_.donorIndices()[stencilI];
        const scalarList& weights = state_.weights()[stencilI];
        forAll(donors, donorI)
        {
            values[stencilI] += weights[donorI] * donor[donors[donorI]];
        }
    }
    eqn.setValues(state_.acceptorIndices(), values);
}

void Foam::MarineInterMeshMatrix::applyVector
(
    fvMatrix<vector>& eqn,
    const volVectorField& donor
) const
{
    vectorField values(state_.size(), Zero);
    forAll(state_.acceptorIndices(), stencilI)
    {
        const labelList& donors = state_.donorIndices()[stencilI];
        const scalarList& weights = state_.weights()[stencilI];
        forAll(donors, donorI)
        {
            values[stencilI] += weights[donorI] * donor[donors[donorI]];
        }
    }
    eqn.setValues(state_.acceptorIndices(), values);
}

// ************************************************************************* //
