/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2024 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Class
    Foam::stabilizedWindkesselVelocityFvPatchVectorField

Description
    A backflow-stabilized velocity boundary condition for windkessel outlets.

    Supports three stabilization methods:

    1. Simple damping (stabilizationType = "simple"):
        V_stabilized = (1 - beta) * V_backflow + V_tangential
        - Fast, intuitive
        - beta = 0.9 recommended

    2. Flux-based FVM stabilization (stabilizationType = "fluxBased"):
        Uses face flux (phi) for true finite volume detection
        - Native FVM approach, uses surfaceScalarField phi
        - Applies momentum flux correction during backflow
        - beta = 0.5-1.0 recommended
        - Most FVM-consistent method

    3. Traction-based (stabilizationType = "traction", from Moghadam et al. 2011):
        Adds convective traction: -beta*rho*(v·n)⁻*v
        - Physics-based approach adapted for FVM
        - Proven in patient-specific cardiovascular cases
        - beta = 0.2-0.5 recommended

    The fluxBased method is recommended for FVM solvers as it directly
    uses the computed face fluxes, providing the most consistent treatment.

\*---------------------------------------------------------------------------*/

#ifndef stabilizedWindkesselVelocityFvPatchVectorField_H
#define stabilizedWindkesselVelocityFvPatchVectorField_H

#include "fvPatchFields.H"
#include "zeroGradientFvPatchFields.H"

namespace Foam
{

class stabilizedWindkesselVelocityFvPatchVectorField
:
    public zeroGradientFvPatchVectorField
{
    // Private Data

        //- Stabilization coefficient
        //  Simple method: 0.9 recommended
        //  VMS method: 0.2-0.5 recommended
        scalar beta_;

        //- Flag to enable/disable stabilization
        bool enableStabilization_;

        //- Stabilization type: "simple", "fluxBased", or "traction"
        //  simple: Direct velocity damping
        //  fluxBased: True FVM method using face flux (phi)
        //  traction: Traction-based method (adapted from Esmaily-Moghadam 2011)
        word stabilizationType_;

        //- Damping factor for stabilization methods (0.0-1.0)
        //  Controls how much of the stabilization term is applied
        scalar dampingFactor_;

public:

    //- Runtime type information
    TypeName("stabilizedWindkesselVelocity");


    // Constructors

        //- Construct from patch and internal field and dictionary
        stabilizedWindkesselVelocityFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping
        stabilizedWindkesselVelocityFvPatchVectorField
        (
            const stabilizedWindkesselVelocityFvPatchVectorField&,
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const fvPatchFieldMapper&
        );

        //- Construct as copy
        stabilizedWindkesselVelocityFvPatchVectorField
        (
            const stabilizedWindkesselVelocityFvPatchVectorField&,
            const DimensionedField<vector, volMesh>&
        );

    //- Destructor
    virtual ~stabilizedWindkesselVelocityFvPatchVectorField() = default;


    // Member Functions

        //- Update the coefficients
        virtual void updateCoeffs();
        
        //- Evaluate the patch field
        virtual void evaluate
        (
            const Pstream::commsTypes commsType = Pstream::commsTypes::blocking
        );

        //- Return the matrix diagonal coefficients corresponding to the
        //  evaluation of the value of this patchField with given weights
        virtual tmp<Field<vector>> valueInternalCoeffs
        (
            const tmp<scalarField>&
        ) const;

        //- Return the matrix source coefficients corresponding to the
        //  evaluation of the value of this patchField with given weights
        virtual tmp<Field<vector>> valueBoundaryCoeffs
        (
            const tmp<scalarField>&
        ) const;

        //- Write
        virtual void write(Ostream&) const;
};

} // End namespace Foam

#endif