/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2024 OpenFOAM Foundation
     \\/     M anipulation  |
---------------------------------------------------------------------------*/

#include "vectorFittingImpedanceFvPatchScalarField.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "surfaceFields.H"

namespace Foam
{

// Constructors

vectorFittingImpedanceFvPatchScalarField::vectorFittingImpedanceFvPatchScalarField
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchScalarField(p, iF, dict),
    phiName_(dict.lookupOrDefault<word>("phi", "phi")),
    UName_(dict.lookupOrDefault<word>("U", "U")),
    couplingMode_(dict.lookupOrDefault<word>("couplingMode", "explicit")),
    order_(readLabel(dict.lookup("order"))),
    residues_(order_, 0.0),
    poles_(order_, 0.0),
    directTerm_(readScalar(dict.lookup("directTerm"))),
    stateVariables_(order_, 0.0),
    stateVariables_old_(order_, 0.0),
    rho_(dict.lookupOrDefault<scalar>("rho", 1060.0)),
    q0_(0.0),
    q_1_(dict.lookupOrDefault<scalar>("q_1", 0.0)),
    lastUpdateTime_(-GREAT),
    patchArea_(gSum(p.magSf()))
{
    // Validate coupling mode
    if (couplingMode_ != "explicit" && couplingMode_ != "implicit")
    {
        FatalErrorInFunction
            << "couplingMode must be 'explicit' or 'implicit', not '"
            << couplingMode_ << "'"
            << exit(FatalError);
    }

    // Read poles
    dict.lookup("poles") >> poles_;
    if (poles_.size() != order_)
    {
        FatalErrorInFunction
            << "poles list size (" << poles_.size()
            << ") must equal order (" << order_ << ")"
            << exit(FatalError);
    }

    // Read residues
    dict.lookup("residues") >> residues_;
    if (residues_.size() != order_)
    {
        FatalErrorInFunction
            << "residues list size (" << residues_.size()
            << ") must equal order (" << order_ << ")"
            << exit(FatalError);
    }

    // Validate poles for stability
    validatePoles();

    // Initialize state variables from dictionary if present (for restart)
    if (dict.found("stateVariables"))
    {
        dict.lookup("stateVariables") >> stateVariables_;
        if (stateVariables_.size() != order_)
        {
            WarningInFunction
                << "stateVariables list size mismatch, reinitializing to zero"
                << endl;
            stateVariables_ = scalarList(order_, 0.0);
        }
    }

    stateVariables_old_ = stateVariables_;

    if (dict.found("value"))
    {
        fvPatchField<scalar>::operator=
        (
            scalarField("value", dict, p.size())
        );
    }
    else
    {
        fvPatchField<scalar>::operator=(patchInternalField());
    }
}


vectorFittingImpedanceFvPatchScalarField::vectorFittingImpedanceFvPatchScalarField
(
    const vectorFittingImpedanceFvPatchScalarField& ptf,
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchScalarField(ptf, p, iF, mapper),
    phiName_(ptf.phiName_),
    UName_(ptf.UName_),
    couplingMode_(ptf.couplingMode_),
    order_(ptf.order_),
    residues_(ptf.residues_),
    poles_(ptf.poles_),
    directTerm_(ptf.directTerm_),
    stateVariables_(ptf.stateVariables_),
    stateVariables_old_(ptf.stateVariables_old_),
    rho_(ptf.rho_),
    q0_(ptf.q0_),
    q_1_(ptf.q_1_),
    lastUpdateTime_(ptf.lastUpdateTime_),
    patchArea_(ptf.patchArea_)
{}


vectorFittingImpedanceFvPatchScalarField::vectorFittingImpedanceFvPatchScalarField
(
    const vectorFittingImpedanceFvPatchScalarField& vfipsf,
    const DimensionedField<scalar, volMesh>& iF
)
:
    fixedValueFvPatchScalarField(vfipsf, iF),
    phiName_(vfipsf.phiName_),
    UName_(vfipsf.UName_),
    couplingMode_(vfipsf.couplingMode_),
    order_(vfipsf.order_),
    residues_(vfipsf.residues_),
    poles_(vfipsf.poles_),
    directTerm_(vfipsf.directTerm_),
    stateVariables_(vfipsf.stateVariables_),
    stateVariables_old_(vfipsf.stateVariables_old_),
    rho_(vfipsf.rho_),
    q0_(vfipsf.q0_),
    q_1_(vfipsf.q_1_),
    lastUpdateTime_(vfipsf.lastUpdateTime_),
    patchArea_(vfipsf.patchArea_)
{}


// Member Functions

void vectorFittingImpedanceFvPatchScalarField::validatePoles() const
{
    forAll(poles_, i)
    {
        if (poles_[i] >= 0.0)
        {
            FatalErrorInFunction
                << "Pole " << i << " has value " << poles_[i]
                << " but must be negative for stability"
                << nl
                << "All poles must satisfy: poles[i] < 0"
                << exit(FatalError);
        }

        // Warn about very stiff poles (may cause numerical issues)
        if (poles_[i] < -1000.0)
        {
            WarningInFunction
                << "Pole " << i << " is very negative (" << poles_[i] << " rad/s)"
                << nl
                << "This may lead to stiff ODE requiring very small timesteps"
                << endl;
        }
    }
}


void vectorFittingImpedanceFvPatchScalarField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    // Get current simulation time
    const scalar currentTime = db().time().value();

    // Only update once per timestep - prevent oscillations from multiple PISO/PIMPLE iterations
    if (mag(currentTime - lastUpdateTime_) < SMALL)
    {
        // Already updated this timestep - keep pressure fixed during iterations
        fixedValueFvPatchScalarField::updateCoeffs();
        return;
    }

    // Record that we're updating at this timestep
    lastUpdateTime_ = currentTime;

    // --- 1. Get the flux from the previous timestep's result ---
    const surfaceScalarField& phi =
        db().lookupObject<surfaceScalarField>(phiName_);

    // Sum the flux over the patch to get the flow rate Q [m³/s]
    q0_ = sum(phi.boundaryField()[this->patch().index()]);

    // --- 2. Get timestep ---
    const scalar dt = db().time().deltaTValue();

    // --- 3. Initialize pressure with direct term contribution ---
    //  P = d·Q (high-frequency/instantaneous resistance)
    scalar P = directTerm_ * q0_;

    // --- 4. Update state variables using recursive convolution algorithm ---
    //  For each pole-residue pair: zᵢⁿ⁺¹ = exp(pᵢ·Δt)·zᵢⁿ + rᵢ·Qⁿ⁺¹·[exp(pᵢ·Δt)-1]/pᵢ
    //
    //  This recursive formula eliminates need for full Q(t) history
    //  Memory: O(N) instead of O(M) where M = number of timesteps
    //  Key advantage for long cardiovascular simulations

    forAll(poles_, i)
    {
        const scalar p = poles_[i];  // Must be negative for stability
        const scalar r = residues_[i];

        // Compute exponential term exp(pᵢ·Δt)
        // Since pᵢ < 0, this decay factor is between 0 and 1
        const scalar expPdt = exp(p * dt);

        // Previous state value
        const scalar zOld = stateVariables_[i];

        // Convolution integral term: [exp(pᵢ·Δt) - 1] / pᵢ
        // Handle special case when |pᵢ·Δt| is very small (avoid division by near-zero)
        scalar convolutionTerm;

        if (mag(p * dt) < 1e-6)
        {
            // Taylor series expansion: (exp(x)-1)/x ≈ 1 + x/2 + x²/6 + ...
            // Use 2nd order approximation for better accuracy
            const scalar pdt = p * dt;
            convolutionTerm = dt * (1.0 + 0.5*pdt + pdt*pdt/6.0);
        }
        else
        {
            // Standard formula
            convolutionTerm = (expPdt - 1.0) / p;
        }

        // Recursive update: decay of old state + contribution from current flow
        stateVariables_[i] = expPdt * zOld + r * q0_ * convolutionTerm;

        // Add this pole's contribution to total pressure
        P += stateVariables_[i];
    }

    // --- 5. Set the boundary condition value for this timestep ---
    //  Convert dynamic pressure → kinematic pressure for incompressible solver
    //  P_dynamic [Pa] → p_kinematic [m²/s²] = P/ρ
    this->operator==(P / rho_);

    // --- 6. Update historical values for the next timestep ---
    stateVariables_old_ = stateVariables_;
    q_1_ = q0_;

    fixedValueFvPatchScalarField::updateCoeffs();
}


scalar vectorFittingImpedanceFvPatchScalarField::calculateEffectiveImpedance() const
{
    // Calculate effective impedance for implicit coupling
    // Z_eff represents the instantaneous sensitivity dp_kin/dQ
    //
    // Derivation (dynamic):
    //   P = d·Q + Σᵢ zᵢ
    //   zᵢⁿ⁺¹ = exp(pᵢ·Δt)·zᵢⁿ + rᵢ·Qⁿ⁺¹·[exp(pᵢ·Δt)-1]/pᵢ
    //   ∂P/∂Q = d + Σᵢ rᵢ·[exp(pᵢ·Δt)-1]/pᵢ
    //         = d + Σᵢ rᵢ·Δt/(1 - exp(pᵢ·Δt))  (alternate form)
    //
    // For incompressible (kinematic): Z_eff_kin = Z_eff_dyn / ρ
    // Units: [Pa·s/m³]/[kg/m³] = [1/(m·s)]

    const scalar dt = db().time().deltaTValue();
    scalar Z_eff_dyn = directTerm_;

    forAll(poles_, i)
    {
        const scalar p = poles_[i];
        const scalar r = residues_[i];
        const scalar expPdt = exp(p * dt);

        // Contribution from each pole-residue pair
        // Since p < 0, we have 0 < expPdt < 1, so (1 - expPdt) > 0
        if (mag(1.0 - expPdt) > SMALL)
        {
            Z_eff_dyn += r * dt / (1.0 - expPdt);
        }
        else
        {
            // Very small dt or very negative pole: use approximation
            Z_eff_dyn += r * dt;
        }
    }

    // Convert dynamic → kinematic impedance for incompressible solver
    return Z_eff_dyn / rho_;
}


tmp<Field<scalar>> vectorFittingImpedanceFvPatchScalarField::snGrad() const
{
    // Standard gradient for fixed value BC
    // Same for both explicit and implicit modes
    return fixedValueFvPatchScalarField::snGrad();
}


tmp<Field<scalar>>
vectorFittingImpedanceFvPatchScalarField::valueInternalCoeffs
(
    const tmp<scalarField>& w
) const
{
    if (couplingMode_ == "implicit")
    {
        // For implicit coupling, we modify the matrix diagonal to include
        // the impedance contribution
        // This adds resistance term: -Z_eff/A_patch to the momentum equation

        const scalar Z_eff = calculateEffectiveImpedance();

        // Internal coefficient adds to matrix diagonal
        tmp<Field<scalar>> tcoeff =
            fixedValueFvPatchScalarField::valueInternalCoeffs(w);

        // Add implicit impedance contribution
        // This stabilizes the coupling by penalizing rapid flow rate changes
        const scalar impedanceFactor = Z_eff / (patchArea_ + SMALL);
        tcoeff.ref() -= impedanceFactor * w;

        return tcoeff;
    }
    else
    {
        // Explicit mode: standard fixed value behavior
        return fixedValueFvPatchScalarField::valueInternalCoeffs(w);
    }
}


tmp<Field<scalar>>
vectorFittingImpedanceFvPatchScalarField::valueBoundaryCoeffs
(
    const tmp<scalarField>& w
) const
{
    if (couplingMode_ == "implicit")
    {
        // Boundary coefficient: adds source term contribution
        // This includes the historical state variable terms

        tmp<Field<scalar>> tcoeff =
            fixedValueFvPatchScalarField::valueBoundaryCoeffs(w);

        // Add contribution from previous state variables
        // These represent the "memory" of the impedance function
        scalar historicalSource = 0.0;

        const scalar dt = db().time().deltaTValue();
        forAll(poles_, i)
        {
            const scalar p = poles_[i];
            const scalar expPdt = exp(p * dt);

            // Contribution from previous timestep's state
            // This maintains continuity of the convolution integral
            historicalSource += expPdt * stateVariables_old_[i];
        }

        // Add to boundary source (distributed over patch area)
        tcoeff.ref() += historicalSource * w / (patchArea_ + SMALL);

        return tcoeff;
    }
    else
    {
        // Explicit mode: standard fixed value behavior
        return fixedValueFvPatchScalarField::valueBoundaryCoeffs(w);
    }
}


void vectorFittingImpedanceFvPatchScalarField::write(Ostream& os) const
{
    // Use the base class to write the "type" and "value" entries
    fixedValueFvPatchScalarField::write(os);

    // Write the parameters
    os.writeKeyword("phi") << phiName_ << token::END_STATEMENT << nl;
    os.writeKeyword("U") << UName_ << token::END_STATEMENT << nl;
    os.writeKeyword("couplingMode") << couplingMode_ << token::END_STATEMENT << nl;
    os.writeKeyword("order") << order_ << token::END_STATEMENT << nl;

    // Write vector fitting parameters
    os.writeKeyword("directTerm") << directTerm_ << token::END_STATEMENT << nl;

    // Force ASCII format for list entries (poles, residues, stateVariables)
    // This ensures proper parsing even when decomposePar uses binary format
    // Binary representation of lists cannot be read by OpenFOAM parser
    if (os.format() == IOstream::BINARY)
    {
        // Temporarily switch to ASCII for list entries
        IOstream::streamFormat oldFormat = os.format();
        const_cast<Ostream&>(os).format(IOstream::ASCII);

        os.writeKeyword("poles") << poles_ << token::END_STATEMENT << nl;
        os.writeKeyword("residues") << residues_ << token::END_STATEMENT << nl;

        // Restore binary format for scalar entries
        const_cast<Ostream&>(os).format(oldFormat);
    }
    else
    {
        // Already in ASCII mode, write normally
        os.writeKeyword("poles") << poles_ << token::END_STATEMENT << nl;
        os.writeKeyword("residues") << residues_ << token::END_STATEMENT << nl;
    }

    // Write optional density
    os.writeKeyword("rho") << rho_ << token::END_STATEMENT << nl;

    // Write the historical state for robust restarts
    // Critical for maintaining convolution continuity across restart
    os.writeKeyword("q_1") << q_1_ << token::END_STATEMENT << nl;

    // stateVariables is also a list, handle binary format the same way
    if (os.format() == IOstream::BINARY)
    {
        IOstream::streamFormat oldFormat = os.format();
        const_cast<Ostream&>(os).format(IOstream::ASCII);

        os.writeKeyword("stateVariables") << stateVariables_
            << token::END_STATEMENT << nl;

        const_cast<Ostream&>(os).format(oldFormat);
    }
    else
    {
        os.writeKeyword("stateVariables") << stateVariables_
            << token::END_STATEMENT << nl;
    }
}

} // End namespace Foam

#include "addToRunTimeSelectionTable.H"

namespace Foam
{
    makePatchTypeField
    (
        fvPatchScalarField,
        vectorFittingImpedanceFvPatchScalarField
    );
}

// ************************************************************************* //
