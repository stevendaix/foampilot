/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2024 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Description
    3-Element Windkessel boundary condition using KINEMATIC units throughout.

    All parameters are in kinematic units (consistent with OpenFOAM p field):
    - p, p0, p_1: [m²/s²] kinematic pressure
    - R, Z: [s/m] kinematic resistance
    - C: [m] kinematic compliance
    - Q: [m³/s] volumetric flow rate

    Conversion from dynamic (SI) units:
    - R_kin = R_dyn / rho  (Pa·s/m³ → s/m)
    - C_kin = C_dyn * rho  (m³/Pa → m)
    - p_kin = p_dyn / rho  (Pa → m²/s²)

---------------------------------------------------------------------------*/

#include "modularWKPressureFvPatchScalarField.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "surfaceFields.H"

namespace Foam
{

// Constructors

modularWKPressureFvPatchScalarField::modularWKPressureFvPatchScalarField
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchScalarField(p, iF, dict),
    phiName_(dict.lookupOrDefault<word>("phi", "phi")),
    UName_(dict.lookupOrDefault<word>("U", "U")),
    order_(readLabel(dict.lookup("order"))),
    couplingMode_(dict.lookupOrDefault<word>("couplingMode", "explicit")),
    // All parameters read directly - already in kinematic units
    R_(readScalar(dict.lookup("R"))),
    C_(readScalar(dict.lookup("C"))),
    Z_(readScalar(dict.lookup("Z"))),
    p1_(0.0),
    p0_(readScalar(dict.lookup("p0"))),      // Kinematic [m²/s²]
    p_1_(dict.lookupOrDefault("p_1", p0_)),  // Kinematic [m²/s²]
    p_2_(dict.lookupOrDefault("p_2", p_1_)), // Kinematic [m²/s²]
    q0_(0.0),
    q_1_(readScalar(dict.lookup("q_1"))),    // [m³/s]
    q_2_(dict.lookupOrDefault("q_2", q_1_)),
    q_3_(dict.lookupOrDefault("q_3", q_2_)),
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

    // Set the initial pressure value of the patch from p0 (already kinematic)
    fixedValueFvPatchScalarField::operator==(p0_);
}


modularWKPressureFvPatchScalarField::modularWKPressureFvPatchScalarField
(
    const modularWKPressureFvPatchScalarField& ptf,
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchScalarField(ptf, p, iF, mapper),
    phiName_(ptf.phiName_),
    UName_(ptf.UName_),
    order_(ptf.order_),
    couplingMode_(ptf.couplingMode_),
    R_(ptf.R_),
    C_(ptf.C_),
    Z_(ptf.Z_),
    p1_(ptf.p1_),
    p0_(ptf.p0_),
    p_1_(ptf.p_1_),
    p_2_(ptf.p_2_),
    q0_(ptf.q0_),
    q_1_(ptf.q_1_),
    q_2_(ptf.q_2_),
    q_3_(ptf.q_3_),
    lastUpdateTime_(ptf.lastUpdateTime_),
    patchArea_(ptf.patchArea_)
{}

modularWKPressureFvPatchScalarField::modularWKPressureFvPatchScalarField
(
    const modularWKPressureFvPatchScalarField& fvmpsf,
    const DimensionedField<scalar, volMesh>& iF
)
:
    fixedValueFvPatchScalarField(fvmpsf, iF),
    phiName_(fvmpsf.phiName_),
    UName_(fvmpsf.UName_),
    order_(fvmpsf.order_),
    couplingMode_(fvmpsf.couplingMode_),
    R_(fvmpsf.R_),
    C_(fvmpsf.C_),
    Z_(fvmpsf.Z_),
    p1_(fvmpsf.p1_),
    p0_(fvmpsf.p0_),
    p_1_(fvmpsf.p_1_),
    p_2_(fvmpsf.p_2_),
    q0_(fvmpsf.q0_),
    q_1_(fvmpsf.q_1_),
    q_2_(fvmpsf.q_2_),
    q_3_(fvmpsf.q_3_),
    lastUpdateTime_(fvmpsf.lastUpdateTime_),
    patchArea_(fvmpsf.patchArea_)
{}


// Member Functions

void modularWKPressureFvPatchScalarField::updateCoeffs()
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
        fixedValueFvPatchScalarField::updateCoeffs();
        return;
    }

    lastUpdateTime_ = currentTime;

    // --- 1. Get the flux from the previous timestep's result ---
    const surfaceScalarField& phi =
        db().lookupObject<surfaceScalarField>(phiName_);

    // Sum the flux over the patch to get the flow rate Q [m³/s]
    q0_ = sum(phi.boundaryField()[this->patch().index()]);

    // --- 2. Solve the Windkessel ODE for the new pressure ---
    // All units are kinematic - no rho conversion needed!
    //
    // 3-Element Windkessel equation (kinematic form):
    //   p = Z*Q + p_c
    //   dp_c/dt = Q/C - p_c/(R*C)
    //
    // Using BDF discretization for dp/dt and dQ/dt

    const scalar dt = db().time().deltaTValue();
    scalar Q_source = 0.0;
    scalar Pgrad_part = 0.0;
    scalar Pdenom = 1.0;

    switch (order_)
    {
        case 1:
            // 1st order backward difference
            Q_source = (q0_/C_)*(1.0 + Z_/R_) + (Z_/dt)*(q0_ - q_1_);
            Pgrad_part = -p0_/dt;
            Pdenom = 1.0/dt + 1.0/(R_*C_);
            break;

        case 2:
            // 2nd order backward difference
            Q_source = (q0_/C_)*(1.0 + Z_/R_) + (Z_/dt)*(1.5*q0_ - 2.0*q_1_ + 0.5*q_2_);
            Pgrad_part = (-2.0*p0_ + 0.5*p_1_)/dt;
            Pdenom = 1.5/dt + 1.0/(R_*C_);
            break;

        case 3:
            // 3rd order backward difference (default, most accurate)
            Q_source = (q0_/C_)*(1.0 + Z_/R_) + (Z_/dt)*((11.0/6.0)*q0_ - 3.0*q_1_ + 1.5*q_2_ - (1.0/3.0)*q_3_);
            Pgrad_part = (-3.0*p0_ + 1.5*p_1_ - (1.0/3.0)*p_2_)/dt;
            Pdenom = (11.0/6.0)/dt + 1.0/(R_*C_);
            break;

        default:
            // Default to 1st order
            Q_source = (q0_/C_)*(1.0 + Z_/R_) + (Z_/dt)*(q0_ - q_1_);
            Pgrad_part = -p0_/dt;
            Pdenom = 1.0/dt + 1.0/(R_*C_);
            break;
    }

    // Calculate new pressure [m²/s²] (kinematic)
    p1_ = (Q_source - Pgrad_part) / Pdenom;

    // --- 3. Set the boundary condition value ---
    this->operator==(p1_);

    // --- 4. Update historical values for the next timestep ---
    q_3_ = q_2_;
    q_2_ = q_1_;
    q_1_ = q0_;

    p_2_ = p_1_;
    p_1_ = p0_;
    p0_ = p1_;

    fixedValueFvPatchScalarField::updateCoeffs();
}


scalar modularWKPressureFvPatchScalarField::calculateImpedance() const
{
    // Calculate effective Windkessel impedance [s/m] (kinematic)
    // Z_eff = dP/dQ for implicit coupling
    //
    // From 3-element Windkessel with BDF:
    // Z_eff = Z + R*dt/(alpha*R*C + dt)
    //
    // Where alpha is the BDF coefficient (1.0, 1.5, or 11/6)

    const scalar dt = db().time().deltaTValue();
    scalar alpha = 1.0;

    switch (order_)
    {
        case 1:
            alpha = 1.0;
            break;
        case 2:
            alpha = 1.5;
            break;
        case 3:
            alpha = 11.0/6.0;
            break;
        default:
            alpha = 1.0;
            break;
    }

    // RC circuit effective impedance (all kinematic, no rho needed)
    const scalar RC_eff = (R_ * dt) / (alpha * R_ * C_ + dt);

    return Z_ + RC_eff;
}


tmp<Field<scalar>> modularWKPressureFvPatchScalarField::snGrad() const
{
    // Standard behavior for both explicit and implicit modes
    return fixedValueFvPatchScalarField::snGrad();
}


tmp<Field<scalar>> modularWKPressureFvPatchScalarField::valueInternalCoeffs
(
    const tmp<scalarField>& w
) const
{
    if (couplingMode_ == "implicit")
    {
        // For implicit coupling, modify the matrix diagonal to include
        // the Windkessel impedance contribution
        const scalar Z_eff = calculateImpedance();

        tmp<Field<scalar>> tcoeff = fixedValueFvPatchScalarField::valueInternalCoeffs(w);

        // Add implicit resistance contribution
        const scalar impedanceFactor = Z_eff / (patchArea_ + SMALL);
        tcoeff.ref() -= impedanceFactor * w;

        return tcoeff;
    }
    else
    {
        return fixedValueFvPatchScalarField::valueInternalCoeffs(w);
    }
}


tmp<Field<scalar>> modularWKPressureFvPatchScalarField::valueBoundaryCoeffs
(
    const tmp<scalarField>& w
) const
{
    if (couplingMode_ == "implicit")
    {
        tmp<Field<scalar>> tcoeff = fixedValueFvPatchScalarField::valueBoundaryCoeffs(w);

        const scalar dt = db().time().deltaTValue();
        scalar historicalSource = 0.0;

        // Historical contribution (all kinematic, no rho conversion)
        switch (order_)
        {
            case 1:
                historicalSource = -Z_ * q_1_ / dt - p0_ / dt;
                break;
            case 2:
                historicalSource = -Z_ * (-2.0*q_1_ + 0.5*q_2_) / dt
                                  + (-2.0*p0_ + 0.5*p_1_) / dt;
                break;
            case 3:
                historicalSource = -Z_ * (-3.0*q_1_ + 1.5*q_2_ - (1.0/3.0)*q_3_) / dt
                                  + (-3.0*p0_ + 1.5*p_1_ - (1.0/3.0)*p_2_) / dt;
                break;
            default:
                historicalSource = -Z_ * q_1_ / dt - p0_ / dt;
                break;
        }

        // Compliance contribution: Q/C (already kinematic)
        const scalar complianceSource = q0_ / (C_ + SMALL);

        // Add to boundary source
        tcoeff.ref() += (historicalSource + complianceSource) * w / (patchArea_ + SMALL);

        return tcoeff;
    }
    else
    {
        return fixedValueFvPatchScalarField::valueBoundaryCoeffs(w);
    }
}


void modularWKPressureFvPatchScalarField::write(Ostream& os) const
{
    fixedValueFvPatchScalarField::write(os);

    os.writeKeyword("phi") << phiName_ << token::END_STATEMENT << nl;
    os.writeKeyword("U") << UName_ << token::END_STATEMENT << nl;
    os.writeKeyword("couplingMode") << couplingMode_ << token::END_STATEMENT << nl;
    os.writeKeyword("order") << order_ << token::END_STATEMENT << nl;

    // Write kinematic Windkessel parameters
    os.writeKeyword("R") << R_ << token::END_STATEMENT << nl;
    os.writeKeyword("C") << C_ << token::END_STATEMENT << nl;
    os.writeKeyword("Z") << Z_ << token::END_STATEMENT << nl;

    // Write state variables (all kinematic, no conversion)
    os.writeKeyword("p0") << p0_ << token::END_STATEMENT << nl;
    os.writeKeyword("p_1") << p_1_ << token::END_STATEMENT << nl;
    os.writeKeyword("p_2") << p_2_ << token::END_STATEMENT << nl;
    os.writeKeyword("q_1") << q_1_ << token::END_STATEMENT << nl;
    os.writeKeyword("q_2") << q_2_ << token::END_STATEMENT << nl;
    os.writeKeyword("q_3") << q_3_ << token::END_STATEMENT << nl;
}

} // End namespace Foam

#include "addToRunTimeSelectionTable.H"

namespace Foam
{
    makePatchTypeField(fvPatchScalarField, modularWKPressureFvPatchScalarField);
}
