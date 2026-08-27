#include "MarineInterMeshStencilState.H"
#include "IOdictionary.H"
#include "error.H"

Foam::MarineInterMeshStencilState::MarineInterMeshStencilState
(
    const fvMesh& mesh,
    const label donorCellCount
)
:
    mesh_(mesh),
    donorRegion_(word::null),
    acceptorRegion_(word::null),
    acceptorIndices_(),
    donorIndices_(),
    weights_()
{
    IOdictionary dict
    (
        IOobject
        (
            "marineInterMeshStencils",
            mesh_.time().constant(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false
        )
    );

    dict.lookup("donorRegion") >> donorRegion_;
    dict.lookup("acceptorRegion") >> acceptorRegion_;
    const List<dictionary> acceptors(dict.lookup("acceptors"));
    acceptorIndices_.setSize(acceptors.size());
    donorIndices_.setSize(acceptors.size());
    weights_.setSize(acceptors.size());

    forAll(acceptors, stencilI)
    {
        acceptors[stencilI].lookup("index") >> acceptorIndices_[stencilI];
        acceptors[stencilI].lookup("donorIndices")
            >> donorIndices_[stencilI];
        acceptors[stencilI].lookup("weights") >> weights_[stencilI];

        if
        (
            acceptorIndices_[stencilI] < 0
         || acceptorIndices_[stencilI] >= mesh_.nCells()
        )
        {
            FatalErrorInFunction
                << "Acceptor index " << acceptorIndices_[stencilI]
                << " is outside target mesh cell range" << nl
                << exit(FatalError);
        }
        if
        (
            donorIndices_[stencilI].empty()
         || donorIndices_[stencilI].size() != weights_[stencilI].size()
        )
        {
            FatalErrorInFunction
                << "Stencil " << stencilI
                << " has invalid donor/weight list sizes" << nl
                << exit(FatalError);
        }

        scalar sum = 0;
        forAll(donorIndices_[stencilI], donorI)
        {
            const label donor = donorIndices_[stencilI][donorI];
            const scalar weight = weights_[stencilI][donorI];
            if (donor < 0 || donor >= donorCellCount)
            {
                FatalErrorInFunction
                    << "Donor index " << donor << " is outside donor mesh"
                    << nl << exit(FatalError);
            }
            if (!std::isfinite(weight) || weight < 0)
            {
                FatalErrorInFunction
                    << "Invalid inter-mesh weight " << weight << nl
                    << exit(FatalError);
            }
            sum += weight;
        }
        if (mag(sum - scalar(1)) > 1e-10)
        {
            FatalErrorInFunction
                << "Stencil " << stencilI << " weights sum to " << sum
                << ", expected 1" << nl << exit(FatalError);
        }
    }
}

// ************************************************************************* //
