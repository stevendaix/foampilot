/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2024 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Class
    Foam::modularWKPressureFvPatchScalarField

Description
    A modular, 3-element Windkessel pressure boundary condition with optional
    implicit coupling for enhanced stability and performance.

    ALL UNITS ARE KINEMATIC (consistent with OpenFOAM incompressible solvers):
    - Pressure: m²/s² (kinematic) - same as OpenFOAM p field
    - Resistance R, Z: s/m (kinematic) = Pa·s/m³ / rho
    - Compliance C: m (kinematic) = m³/Pa * rho
    - Flow rate Q: m³/s (unchanged)

    User provides rho for reference/documentation only - no internal conversion.

    Coupling modes:
    - "explicit" (default): Uses flux from previous timestep (lagged coupling)
      Simple, stable, but may require small timesteps

    - "implicit": Adds matrix source term contribution for same-timestep coupling
      Up to 4-5x larger stable timesteps, better convergence
      Recommended for production simulations

    Example usage (all kinematic units):
        outlet1
        {
            type            modularWKPressure;
            phi             phi;
            U               U;
            couplingMode    implicit;
            order           3;
            // Windkessel parameters (KINEMATIC units)
            // Convert from dynamic: R_kin = R_dyn/rho, C_kin = C_dyn*rho
            R               268765.33;      // s/m (= 2.85e8 Pa·s/m³ / 1060)
            C               3.72e-6;        // m (= 3.51e-9 m³/Pa * 1060)
            Z               268936.63;      // s/m (= 2.85e8 Pa·s/m³ / 1060)
            // Initial/reference pressure [m²/s²] (kinematic)
            p0              12.577;         // = 13332 Pa / 1060
            // State variables
            p_1             12.577;
            q_1             1.97e-5;        // m³/s (expected steady-state flow)
            q_2             1.97e-5;
            q_3             1.97e-5;
            // Initial field value (kinematic)
            value           uniform 12.577;
        }

\*---------------------------------------------------------------------------*/

#ifndef modularWKPressureFvPatchScalarField_H
#define modularWKPressureFvPatchScalarField_H

#include "fixedValueFvPatchFields.H"

namespace Foam
{

class modularWKPressureFvPatchScalarField
:
    public fixedValueFvPatchScalarField
{
    // Private Data

        //- Name of the flux field
        word phiName_;

        //- Name of the velocity field (for implicit coupling)
        word UName_;

        //- Finite difference order for dQ/dt
        label order_;

        //- Coupling mode: "explicit" (default) or "implicit"
        word couplingMode_;

        //- Windkessel parameters (KINEMATIC units - no conversion needed)
        //  R [s/m] = resistance (kinematic)
        //  C [m] = compliance (kinematic)
        //  Z [s/m] = proximal/characteristic impedance (kinematic)
        scalar R_;
        scalar C_;
        scalar Z_;

        //- Historical pressure values [m²/s²] (kinematic)
        //  p1_ is current timestep, p0_ is t-dt, p_1_ is t-2*dt, etc.
        scalar p1_;
        scalar p0_;
        scalar p_1_;
        scalar p_2_;

        //- Historical flow values [m³/s]
        //  q0_ is current, q_1_ is t-dt, etc.
        scalar q0_;
        scalar q_1_;
        scalar q_2_;
        scalar q_3_;

        //- Track last update time to prevent multiple updates per timestep
        mutable scalar lastUpdateTime_;

        //- Cached patch area [m²] (for implicit coupling)
        scalar patchArea_;


public:

    //- Runtime type information
    TypeName("modularWKPressure");


    // Constructors

        //- Construct from patch and internal field and dictionary
        modularWKPressureFvPatchScalarField
        (
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping
        modularWKPressureFvPatchScalarField
        (
            const modularWKPressureFvPatchScalarField&,
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&,
            const fvPatchFieldMapper&
        );

        //- Construct as copy
        modularWKPressureFvPatchScalarField
        (
            const modularWKPressureFvPatchScalarField&,
            const DimensionedField<scalar, volMesh>&
        );

    //- Destructor
    virtual ~modularWKPressureFvPatchScalarField() = default;


    // Member Functions

        //- Update the coefficients
        virtual void updateCoeffs();

        //- Return gradient at the boundary (for implicit coupling)
        virtual tmp<Field<scalar>> snGrad() const;

        //- Return the matrix diagonal coefficients
        virtual tmp<Field<scalar>> valueInternalCoeffs
        (
            const tmp<scalarField>&
        ) const;

        //- Return the matrix source coefficients
        virtual tmp<Field<scalar>> valueBoundaryCoeffs
        (
            const tmp<scalarField>&
        ) const;

        //- Write
        virtual void write(Ostream&) const;

    // Helper functions

        //- Calculate Windkessel impedance for implicit coupling [s/m]
        scalar calculateImpedance() const;
};

} // End namespace Foam
#endif
