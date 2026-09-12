#include "MarineOversetInterpolation.H"
#include "error.H"

Foam::MarineOversetInterpolation::MarineOversetInterpolation
(
    const List<labelList>& donorIndices,
    const List<scalarList>& weights,
    const label nCells
)
:
    donorIndices_(donorIndices),
    weights_(weights)
{
    validate(nCells);
}

void Foam::MarineOversetInterpolation::validate(const label nCells) const
{
    if (donorIndices_.size() != weights_.size())
    {
        FatalErrorInFunction
            << "The number of donor stencils and weight stencils differs" << nl
            << exit(FatalError);
    }

    forAll(donorIndices_, stencilI)
    {
        if (donorIndices_[stencilI].empty())
        {
            FatalErrorInFunction
                << "Stencil " << stencilI << " is empty" << nl
                << exit(FatalError);
        }

        if (donorIndices_[stencilI].size() != weights_[stencilI].size())
        {
            FatalErrorInFunction
                << "Stencil " << stencilI
                << " has mismatched donors and weights" << nl
                << exit(FatalError);
        }

        scalar sum = 0;
        forAll(donorIndices_[stencilI], donorI)
        {
            const label celli = donorIndices_[stencilI][donorI];
            const scalar weight = weights_[stencilI][donorI];

            if (celli < 0 || celli >= nCells)
            {
                FatalErrorInFunction
                    << "Stencil " << stencilI << " donor cell " << celli
                    << " is outside [0, " << nCells - 1 << "]" << nl
                    << exit(FatalError);
            }

            if (!std::isfinite(weight) || weight < 0)
            {
                FatalErrorInFunction
                    << "Stencil " << stencilI << " has invalid weight "
                    << weight << nl
                    << exit(FatalError);
            }

            sum += weight;
        }

        if (mag(sum - 1.0) > 1e-10)
        {
            FatalErrorInFunction
                << "Stencil " << stencilI << " weights sum to " << sum
                << ", expected 1" << nl
                << exit(FatalError);
        }
    }
}

Foam::scalar Foam::MarineOversetInterpolation::interpolateScalar
(
    const label stencilI,
    const scalarField& donorField
) const
{
    if (stencilI < 0 || stencilI >= size())
    {
        FatalErrorInFunction << "Invalid stencil index " << stencilI << nl
            << exit(FatalError);
    }

    scalar result = 0;
    forAll(donorIndices_[stencilI], donorI)
    {
        result += weights_[stencilI][donorI]
            * donorField[donorIndices_[stencilI][donorI]];
    }
    return result;
}

Foam::vector Foam::MarineOversetInterpolation::interpolateVector
(
    const label stencilI,
    const vectorField& donorField
) const
{
    if (stencilI < 0 || stencilI >= size())
    {
        FatalErrorInFunction << "Invalid stencil index " << stencilI << nl
            << exit(FatalError);
    }

    vector result = vector::zero;
    forAll(donorIndices_[stencilI], donorI)
    {
        result += weights_[stencilI][donorI]
            * donorField[donorIndices_[stencilI][donorI]];
    }
    return result;
}

// ************************************************************************* //
