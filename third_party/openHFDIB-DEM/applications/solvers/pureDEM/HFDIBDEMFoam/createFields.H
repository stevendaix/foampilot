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

Info<< "Reading field lambda\n" << endl;
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
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Creating field refineF\n" << endl;
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


#include "createPhi.H"

#include "readGravitationalAcceleration.H"

IOdictionary transportProperties
(
    IOobject
    (
        "transportProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    )
);

dimensionedScalar rho = dimensionedScalar(
  transportProperties.lookup("rho")
);

IOdictionary HFDIBDEMDict
(
    IOobject
    (
        "HFDIBDEMDict",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    )
);
