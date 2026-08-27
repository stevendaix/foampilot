#include "MarineOversetMatrix.H"
#include "error.H"
#include "DynamicList.H"

Foam::MarineOversetMatrix::MarineOversetMatrix
(
    const MarineOversetCellState& state,
    const MarineOversetInterpolation& interpolation
)
:
    state_(state),
    interpolation_(interpolation)
{}

void Foam::MarineOversetMatrix::applyScalar(fvMatrix<scalar>& matrix) const
{
    labelList cells(state_.nInterpolated() + state_.nHoles());
    scalarField values(cells.size());
    label targetI = 0;

    forAll(state_.status(), celli)
    {
        if (state_.status()[celli] == MarineOversetCellState::hole)
        {
            cells[targetI] = celli;
            values[targetI++] = scalar(0);
        }
        else if (state_.status()[celli] == MarineOversetCellState::interpolated)
        {
            label stencilI = -1;
            forAll(state_.acceptorIndices(), candidateI)
            {
                if (state_.acceptorIndices()[candidateI] == celli)
                {
                    stencilI = candidateI;
                    break;
                }
            }

            if (stencilI < 0)
            {
                FatalErrorInFunction
                    << "No stencil found for interpolated cell " << celli << nl
                    << exit(FatalError);
            }

            cells[targetI] = celli;
            values[targetI++] = interpolation_.interpolateScalar
                (
                    stencilI,
                    matrix.psi().internalField()
                )
            ;
        }
    }

    matrix.setValues(cells, values);
}

void Foam::MarineOversetMatrix::applyVector(fvMatrix<vector>& matrix) const
{
    labelList cells(state_.nInterpolated() + state_.nHoles());
    vectorField values(cells.size());
    label targetI = 0;

    forAll(state_.status(), celli)
    {
        if (state_.status()[celli] == MarineOversetCellState::hole)
        {
            cells[targetI] = celli;
            values[targetI++] = vector::zero;
        }
        else if (state_.status()[celli] == MarineOversetCellState::interpolated)
        {
            label stencilI = -1;
            forAll(state_.acceptorIndices(), candidateI)
            {
                if (state_.acceptorIndices()[candidateI] == celli)
                {
                    stencilI = candidateI;
                    break;
                }
            }

            if (stencilI < 0)
            {
                FatalErrorInFunction
                    << "No stencil found for interpolated cell " << celli << nl
                    << exit(FatalError);
            }

            cells[targetI] = celli;
            values[targetI++] = interpolation_.interpolateVector
                (
                    stencilI,
                    matrix.psi().internalField()
                )
            ;
        }
    }

    matrix.setValues(cells, values);
}

// ************************************************************************* //
