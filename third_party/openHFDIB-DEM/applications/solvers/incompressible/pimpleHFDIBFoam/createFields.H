Info<< "Reading field p\n" << endl;
volScalarField p
(
    IOobject
    (
        "p",
        Time::timeName(runTime.value()),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        Time::timeName(runTime.value()),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

InfoH<< "Reading field Ui\n" << endl;
volVectorField Ui
(
    IOobject
    (
        "Ui",
        Time::timeName(runTime.value()),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    U
);

InfoH<< "Reading field lambda\n" << endl;
volScalarField lambda
(
    IOobject
    (
        "lambda",
        Time::timeName(runTime.value()),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("zero",dimless,0)
);

InfoH<< "Reading field f\n" << endl;
volVectorField f
(
    IOobject
    (
        "f",
        Time::timeName(runTime.value()),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

InfoH<< "Reading field refineF\n" << endl;
volScalarField refineF
(
    IOobject
    (
        "refineF",
        Time::timeName(runTime.value()),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("zero",dimless,0)
);

InfoH<< "Reading field surface\n" << endl;
volScalarField surface
(
    IOobject
    (
        "surface",
        Time::timeName(runTime.value()),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("zero",dimless,0)
);

#include "createPhi.H"


label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p, pimple.dict(), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());

#include "readGravitationalAcceleration.H"

autoPtr<viscosityModel> viscosity(viscosityModel::New(mesh));

autoPtr<incompressible::momentumTransportModel> turbulence
(
    incompressible::momentumTransportModel::New(U, phi, viscosity)
);

#include "createMRF.H"
#include "createFvModels.H"
#include "createFvConstraints.H"
