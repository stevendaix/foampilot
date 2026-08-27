#include "MarineOversetConstraint.H"
#include "addToRunTimeSelectionTable.H"
#include "dictionary.H"
#include "IOdictionary.H"
#include "volFields.H"

namespace Foam
{
namespace fv
{
    defineTypeNameAndDebug(marineOversetConstraint, 0);
    addToRunTimeSelectionTable
    (
        fvConstraint,
        marineOversetConstraint,
        dictionary
    );
}
}

bool Foam::fv::marineOversetConstraint::selected
(
    const word& fieldName
) const
{
    forAll(fields_, fieldI)
    {
        if (fields_[fieldI] == fieldName)
        {
            return true;
        }
    }
    return false;
}

Foam::fv::marineOversetConstraint::marineOversetConstraint
(
    const word& name,
    const word& modelType,
    const fvMesh& mesh,
    const dictionary& dict
)
:
    fvConstraint(name, modelType, mesh, dict),
    fields_(coeffs(dict).lookupOrDefault<wordList>
    (
        "fields",
        wordList({"U", "p", "p_rgh", "alpha.water"})
    )),
    state_(new MarineOversetCellState(mesh)),
    interpolation_
    (
        new MarineOversetInterpolation
        (
            state_->donorIndices(),
            state_->weights(),
            mesh.nCells()
        )
    ),
    matrixOperator_(new MarineOversetMatrix(*state_, *interpolation_)),
    interMeshState_(nullptr),
    interMeshOperator_(nullptr),
    donorMesh_(nullptr)
{
    IOobject interMeshObject
    (
        "marineInterMeshStencils",
        mesh.time().constant(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::NO_WRITE,
        false
    );
    if (interMeshObject.headerOk())
    {
        IOdictionary interMeshDict(interMeshObject);
        word donorRegion;
        interMeshDict.lookup("donorRegion") >> donorRegion;
        if (mesh.time().foundObject<fvMesh>(donorRegion))
        {
            const fvMesh& donor = mesh.time().lookupObject<fvMesh>(donorRegion);
            interMeshState_.reset
            (
                new MarineInterMeshStencilState(mesh, donor.nCells())
            );
            interMeshOperator_.reset
            (
                new MarineInterMeshMatrix(*interMeshState_)
            );
            donorMesh_ = &donor;
            Info<< "marineOversetConstraint: inter-mesh donor region="
                << donorRegion << ", donor cells=" << donor.nCells() << nl;
        }
        else
        {
            Info<< "marineOversetConstraint: donor region " << donorRegion
                << " is not registered; using local overset fallback" << nl;
        }
    }

    Info<< "marineOversetConstraint: fields=" << fields_
        << ", calculated=" << state_->nCalculated()
        << ", interpolated=" << state_->nInterpolated()
        << ", holes=" << state_->nHoles() << nl;
}

bool Foam::fv::marineOversetConstraint::constrain
(
    fvMatrix<scalar>& eqn,
    const word& fieldName
) const
{
    if (!selected(fieldName))
    {
        return false;
    }
    if
    (
        interMeshOperator_.valid()
     && donorMesh_
     && donorMesh_->foundObject<volScalarField>(fieldName)
    )
    {
        interMeshOperator_->applyScalar
        (
            eqn,
            donorMesh_->lookupObject<volScalarField>(fieldName)
        );
        return interMeshState_->size();
    }
    matrixOperator_->applyScalar(eqn);
    return state_->nInterpolated() + state_->nHoles();
}

bool Foam::fv::marineOversetConstraint::constrain
(
    fvMatrix<vector>& eqn,
    const word& fieldName
) const
{
    if (!selected(fieldName))
    {
        return false;
    }
    if
    (
        interMeshOperator_.valid()
     && donorMesh_
     && donorMesh_->foundObject<volVectorField>(fieldName)
    )
    {
        interMeshOperator_->applyVector
        (
            eqn,
            donorMesh_->lookupObject<volVectorField>(fieldName)
        );
        return interMeshState_->size();
    }
    matrixOperator_->applyVector(eqn);
    return state_->nInterpolated() + state_->nHoles();
}

// ************************************************************************* //
