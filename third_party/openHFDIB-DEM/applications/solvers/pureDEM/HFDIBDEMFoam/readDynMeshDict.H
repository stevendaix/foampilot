// OpenFOAM 13 static-mesh port.
// Dynamic mesh motion/topology changes are rejected until the Foundation-13
// fvMeshMover/fvMeshTopoChanger port is implemented.
label maxRefinementLevel = 0;

IOdictionary dynamicMeshDict
(
    IOobject
    (
        "dynamicMeshDict",
        runTime.constant(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::NO_WRITE
    )
);

if (dynamicMeshDict.found("dynamicFvMesh"))
{
    const word dynamicFvMeshTypeName
    (
        dynamicMeshDict.lookup("dynamicFvMesh")
    );

    if (dynamicFvMeshTypeName != "staticFvMesh")
    {
        FatalErrorInFunction
            << "HFDIBDEMFoam OF13 static port does not support dynamic mesh "
            << "type " << dynamicFvMeshTypeName << nl
            << "Use staticFvMesh or complete the fvMeshMover port."
            << exit(FatalError);
    }
}

Info<< "Using OpenFOAM 13 static fvMesh" << endl;
