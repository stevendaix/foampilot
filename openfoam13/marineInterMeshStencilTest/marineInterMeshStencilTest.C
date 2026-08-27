#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "MarineInterMeshStencilState.H"

using namespace Foam;

int main(int argc, char* argv[])
{
    argList::noParallel();
    argList args(argc, argv);
    Time runTime(Time::controlDictName, args);
    fvMesh mesh
    (
        IOobject
        (
            "region0",
            runTime.timePath().name(),
            runTime,
            IOobject::MUST_READ
        )
    );

    MarineInterMeshStencilState state(mesh, 134064);
    if
    (
        state.size() != mesh.nCells()
     || state.donorRegion() != word("background")
     || state.acceptorRegion() != word("hull")
    )
    {
        FatalErrorInFunction
            << "Unexpected inter-mesh contract metadata or size" << nl
            << exit(FatalError);
    }

    Info<< "inter-mesh stencil reader passed: acceptors=" << state.size()
        << ", donorRegion=" << state.donorRegion()
        << ", acceptorRegion=" << state.acceptorRegion() << nl;
    return 0;
}
