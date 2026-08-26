// OpenFOAM 13 static-mesh port: initial refinement is disabled.
// Dynamic refinement requires a dedicated fvMeshTopoChanger implementation.
if (maxRefinementLevel > 0)
{
    FatalErrorInFunction
        << "Static OF13 pimpleHFDIBFoam cannot perform dynamic refinement"
        << exit(FatalError);
}
