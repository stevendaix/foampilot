IOdictionary myDict
(

    IOobject
    (
        "myDictFile",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ
    )

);

IOdictionary diamDistrDict
(

    IOobject
    (
        "diamDistrDict",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ
    )

);

IOdictionary velXCDFDict
(

    IOobject
    (
        "velXCDFDict",
         runTime.constant(),
        mesh,
        IOobject::MUST_READ
    )

);

IOdictionary velRadCDFDict
(

    IOobject
    (
        "velRadCDFDict",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ
    )

);

IOdictionary velAzCDFDict
(

    IOobject
    (
        "velAzCDFDict",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ
    )

);

IOdictionary dCDFDict
(

    IOobject
    (
        "dCDFDict",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ
    )

);
