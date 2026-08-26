// OpenFOAM 13 static-mesh port: initial refinement is disabled.
// A dynamic refinement implementation must use the OF13
// fvMeshTopoChanger API and explicitly rebuild HFDIB bodies after changes.
if (maxRefinementLevel > 0)
{
    FatalErrorInFunction
        << "Static OF13 HFDIBDEMFoam cannot perform dynamic refinement"
        << exit(FatalError);
}
