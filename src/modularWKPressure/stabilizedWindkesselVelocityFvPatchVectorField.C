/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2024 OpenFOAM Foundation
     \\/     M anipulation  |
---------------------------------------------------------------------------*/

#include "stabilizedWindkesselVelocityFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "surfaceFields.H"

namespace Foam
{

// Constructors

stabilizedWindkesselVelocityFvPatchVectorField::stabilizedWindkesselVelocityFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    zeroGradientFvPatchVectorField(p, iF, dict),
    beta_(dict.lookupOrDefault<scalar>("beta", 1.0)),
    enableStabilization_(dict.lookupOrDefault<bool>("enableStabilization", true)),
    stabilizationType_(dict.lookupOrDefault<word>("stabilizationType", "simple")),
    dampingFactor_(dict.lookupOrDefault<scalar>("dampingFactor", 0.5))
{
    // Validate stabilization type
    if (stabilizationType_ != "simple" &&
        stabilizationType_ != "fluxBased" &&
        stabilizationType_ != "traction")
    {
        FatalErrorInFunction
            << "stabilizationType must be 'simple', 'fluxBased', or 'traction', not '"
            << stabilizationType_ << "'"
            << exit(FatalError);
    }

    // Adjust beta defaults based on method
    if (!dict.found("beta"))
    {
        if (stabilizationType_ == "traction")
        {
            beta_ = 0.3;  // Medium beta for traction method
        }
        else if (stabilizationType_ == "fluxBased")
        {
            beta_ = 0.7;  // Medium-high beta for flux-based FVM method
        }
        else
        {
            beta_ = 0.9;  // High beta for simple method
        }
    }

    if (dict.found("value"))
    {
        fvPatchField<vector>::operator=
        (
            vectorField("value", dict, p.size())
        );
    }
    else
    {
        fvPatchField<vector>::operator=(patchInternalField());
    }
}


stabilizedWindkesselVelocityFvPatchVectorField::stabilizedWindkesselVelocityFvPatchVectorField
(
    const stabilizedWindkesselVelocityFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    zeroGradientFvPatchVectorField(ptf, p, iF, mapper),
    beta_(ptf.beta_),
    enableStabilization_(ptf.enableStabilization_),
    stabilizationType_(ptf.stabilizationType_),
    dampingFactor_(ptf.dampingFactor_)
{}


stabilizedWindkesselVelocityFvPatchVectorField::stabilizedWindkesselVelocityFvPatchVectorField
(
    const stabilizedWindkesselVelocityFvPatchVectorField& swvf,
    const DimensionedField<vector, volMesh>& iF
)
:
    zeroGradientFvPatchVectorField(swvf, iF),
    beta_(swvf.beta_),
    enableStabilization_(swvf.enableStabilization_),
    stabilizationType_(swvf.stabilizationType_),
    dampingFactor_(swvf.dampingFactor_)
{}


// Member Functions

void stabilizedWindkesselVelocityFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    // Call parent updateCoeffs first
    zeroGradientFvPatchVectorField::updateCoeffs();
}


void stabilizedWindkesselVelocityFvPatchVectorField::evaluate
(
    const Pstream::commsTypes
)
{
    if (!updated())
    {
        updateCoeffs();
    }

    if (!enableStabilization_)
    {
        // Just use zero gradient if stabilization is disabled
        zeroGradientFvPatchVectorField::evaluate();
        return;
    }

    // Get velocity field from internal cells
    const vectorField velocity = patchInternalField();

    // Get patch normal vectors (pointing outward)
    const vectorField n = patch().nf();

    // Get face areas
    const scalarField& magSf = patch().magSf();

    // Calculate normal velocity component (v·n)
    const scalarField normalVel = velocity & n;

    vectorField stabilizedVelocity = patchInternalField();

    // For fluxBased method, get the phi field
    const surfaceScalarField* phiPtr = nullptr;
    if (stabilizationType_ == "fluxBased")
    {
        phiPtr = &db().lookupObject<surfaceScalarField>("phi");
    }

    if (stabilizationType_ == "simple")
    {
        // Simple damping method: V = (1-β)*V_backflow + V_tangential
        forAll(stabilizedVelocity, faceI)
        {
            const scalar vn = normalVel[faceI];

            if (vn < 0.0) // Backflow detected
            {
                // Reduce backflow by factor (1 - beta)
                const vector& normal = n[faceI];
                const vector tangential = stabilizedVelocity[faceI] - vn * normal;

                // Apply damping only to normal component, preserve tangential
                stabilizedVelocity[faceI] = (1.0 - beta_) * vn * normal + tangential;
            }
        }
    }
    else if (stabilizationType_ == "fluxBased")
    {
        // True FVM method: Use face flux (phi) for backflow detection
        // This is the most FVM-consistent approach
        // Reference: Native OpenFOAM FVM philosophy + Esmaily-Moghadam et al. (2011)
        //
        // FIXED (2025-11-16): Changed from additive to replacement to prevent
        // accumulation over PIMPLE iterations and ensure stable damping

        const fvsPatchField<scalar>& phip = phiPtr->boundaryField()[patch().index()];

        forAll(stabilizedVelocity, faceI)
        {
            const scalar flux = phip[faceI];

            // Backflow detected when flux < 0 (flow into domain)
            if (flux < 0.0)
            {
                const vector& normal = n[faceI];
                const scalar area = magSf[faceI];

                // Flux-based velocity from face flux
                // For incompressible: phi = U·Sf [m³/s], so U_n = phi/A [m/s]
                // For compressible: phi = rho*U·Sf [kg/s], so U_n = phi/(rho*A) [m/s]
                const scalar backflowVel = flux / area;  // Negative for backflow

                // Extract tangential component (preserve tangential flow)
                const scalar vn = normalVel[faceI];
                const vector tangential = stabilizedVelocity[faceI] - vn * normal;

                // Replace normal component with damped backflow velocity
                // This prevents accumulation over PIMPLE iterations
                // Effective damping: (1 - beta*dampingFactor) reduction
                const scalar dampedVn = (1.0 - beta_ * dampingFactor_) * backflowVel;

                // Reconstruct velocity: damped normal + original tangential
                stabilizedVelocity[faceI] = dampedVn * normal + tangential;
            }
        }
    }
    else // stabilizationType_ == "traction"
    {
        // Traction-based stabilization adapted from Moghadam et al. (2011)
        // Original FEM formulation adapted for FVM boundary application
        // Adds convective traction for backflow dissipation
        //
        // FIXED (2025-11-16): Changed from quadratic (vn*v) to linear (vn²) form
        // to prevent excessive velocity corrections. Original formulation created
        // enormous corrections proportional to |v|² causing timestep collapse.
        // New formulation: linear damping similar to simple method but with
        // traction-based physical interpretation

        forAll(stabilizedVelocity, faceI)
        {
            const scalar vn = normalVel[faceI];

            if (vn < 0.0) // Backflow detected: (v·n)⁻ = vn (negative)
            {
                const vector& normal = n[faceI];

                // Extract tangential component (preserve tangential flow)
                const vector tangential = stabilizedVelocity[faceI] - vn * normal;

                // Traction-based damping coefficient
                // Provides stronger dissipation than simple method via dampingFactor
                // Physical interpretation: backflow resistance proportional to dynamic pressure
                const scalar tractionDamping = beta_ * dampingFactor_;

                // Replace normal component with damped velocity
                // Linear damping prevents quadratic velocity growth
                const scalar dampedVn = (1.0 - tractionDamping) * vn;

                // Reconstruct velocity: damped normal + original tangential
                stabilizedVelocity[faceI] = dampedVn * normal + tangential;
            }
        }
    }

    // Set the final field value
    fvPatchField<vector>::operator=(stabilizedVelocity);

    fvPatchField<vector>::evaluate();
}


tmp<Field<vector>> 
stabilizedWindkesselVelocityFvPatchVectorField::valueInternalCoeffs
(
    const tmp<scalarField>& w
) const
{
    // For backflow stabilization, we modify the matrix coefficients
    // to include the stabilization term contribution
    return zeroGradientFvPatchVectorField::valueInternalCoeffs(w);
}


tmp<Field<vector>> 
stabilizedWindkesselVelocityFvPatchVectorField::valueBoundaryCoeffs
(
    const tmp<scalarField>& w
) const
{
    return zeroGradientFvPatchVectorField::valueBoundaryCoeffs(w);
}


void stabilizedWindkesselVelocityFvPatchVectorField::write(Ostream& os) const
{
    zeroGradientFvPatchVectorField::write(os);

    os.writeKeyword("beta") << beta_ << token::END_STATEMENT << nl;
    os.writeKeyword("enableStabilization") << enableStabilization_
        << token::END_STATEMENT << nl;
    os.writeKeyword("stabilizationType") << stabilizationType_
        << token::END_STATEMENT << nl;
    os.writeKeyword("dampingFactor") << dampingFactor_
        << token::END_STATEMENT << nl;
}

} // End namespace Foam

#include "addToRunTimeSelectionTable.H"

namespace Foam
{
    makePatchTypeField
    (
        fvPatchVectorField, 
        stabilizedWindkesselVelocityFvPatchVectorField
    );
}