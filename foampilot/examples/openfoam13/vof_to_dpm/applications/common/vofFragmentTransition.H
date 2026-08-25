/*---------------------------------------------------------------------------*\
  Native OpenFOAM 13 VOF fragment detection shared by solver-side models.
\*---------------------------------------------------------------------------*/
#ifndef vofFragmentTransition_H
#define vofFragmentTransition_H

#include "DynamicList.H"
#include "fvMesh.H"
#include "volFields.H"
#include <cstdint>

namespace Foam
{

struct vofFragmentTransitionRecord
{
    labelList cells;
    std::uint64_t id;
    scalar volume;
    vector centroid;
    vector velocity;
};

class vofFragmentTransition
{
public:
    static List<vofFragmentTransitionRecord> detect
    (
        const volScalarField& alpha,
        const volVectorField& U,
        const scalar threshold,
        const label minCells,
        const scalar minVolume
    )
    {
        const fvMesh& mesh = alpha.mesh();
        const scalarField& alphaField = alpha.internalField();
        const scalarField& cellVolumes = mesh.V();
        const vectorField& cellCentres = mesh.C();
        const labelListList& cellNeighbours = mesh.cellCells();

        boolList visited(mesh.nCells(), false);
        DynamicList<vofFragmentTransitionRecord> fragments;

        forAll(alphaField, seed)
        {
            if (visited[seed] || alphaField[seed] < threshold)
            {
                continue;
            }

            DynamicList<label> stack(32);
            DynamicList<label> component(32);
            stack.append(seed);
            visited[seed] = true;
            scalar fragmentVolume = 0;
            vector fragmentCentroid = vector::zero;
            vector fragmentVelocity = vector::zero;

            while (stack.size())
            {
                const label celli = stack.remove();
                component.append(celli);
                const scalar liquidWeight = alphaField[celli]*cellVolumes[celli];
                fragmentVolume += liquidWeight;
                fragmentCentroid += liquidWeight*cellCentres[celli];
                fragmentVelocity += liquidWeight*U[celli];

                const labelList& neighbours = cellNeighbours[celli];
                forAll(neighbours, nbrI)
                {
                    const label nbr = neighbours[nbrI];
                    if (!visited[nbr] && alphaField[nbr] >= threshold)
                    {
                        visited[nbr] = true;
                        stack.append(nbr);
                    }
                }
            }

            if
            (
                component.size() < minCells
             || fragmentVolume < minVolume
             || fragmentVolume <= SMALL
            )
            {
                continue;
            }

            vofFragmentTransitionRecord fragment;
            fragment.cells.transfer(component);
            Foam::sort(fragment.cells);
            fragment.id = 1469598103934665603ULL;
            forAll(fragment.cells, cellI)
            {
                fragment.id ^=
                    static_cast<std::uint64_t>(fragment.cells[cellI] + 1);
                fragment.id *= 1099511628211ULL;
            }
            fragment.volume = fragmentVolume;
            fragment.centroid = fragmentCentroid/fragmentVolume;
            fragment.velocity = fragmentVelocity/fragmentVolume;
            fragments.append(fragment);
        }

        return fragments;
    }
};

} // End namespace Foam

#endif

// ************************************************************************* //
