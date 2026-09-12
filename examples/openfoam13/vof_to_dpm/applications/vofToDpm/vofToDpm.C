/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           |
     \\/     M anipulation  |
Application
    vofToDpm

Description
    Convert connected VOF liquid fragments into DPM-oriented parcel outputs.
    The selected cells are alpha >= threshold, while physical volume is
    integrated as alpha*cellVolume without renormalisation. This first native
    C++ implementation is serial and writes cloudPositions plus a fragment
    dictionary for manualInjection or a later solver-side cloud adapter.
*---------------------------------------------------------------------------*/

#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"
#include "DynamicList.H"
#include "OFstream.H"
#include "Pstream.H"
#include "mathematicalConstants.H"

using namespace Foam;

struct vofFragment
{
    labelList cells;
    scalar volume;
    vector centroid;
    vector velocity;
};

int main(int argc, char *argv[])
{
    argList::addOption("alpha", "name", "VOF field name (default alpha.liquid)");
    argList::addOption("U", "name", "optional velocity field name (default none)");
    argList::addOption("output", "file", "positions output (default constant/cloudPositions)");
    argList::addOption("threshold", "scalar", "eligible alpha threshold (default 0.5)");
    argList::addOption("minCells", "label", "minimum cells per fragment (default 1)");
    argList::addOption("minVolume", "scalar", "minimum fragment volume (default 0)");
    argList::addOption("rhoLiquid", "scalar", "liquid density (default 1000)");

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    if (Pstream::parRun())
    {
        FatalErrorInFunction
            << "vofToDpm currently requires a serial case. Parallel fragment "
            << "reconciliation must be implemented before enabling MPI output."
            << exit(FatalError);
    }

    word alphaName("alpha.liquid");
    args.optionReadIfPresent("alpha", alphaName);
    word UName("none");
    args.optionReadIfPresent("U", UName);
    fileName positionsPath(runTime.constant()/"cloudPositions");
    args.optionReadIfPresent("output", positionsPath);
    scalar threshold = 0.5;
    args.optionReadIfPresent("threshold", threshold);
    label minCells = 1;
    args.optionReadIfPresent("minCells", minCells);
    scalar minVolume = 0;
    args.optionReadIfPresent("minVolume", minVolume);
    scalar rhoLiquid = 1000;
    args.optionReadIfPresent("rhoLiquid", rhoLiquid);

    if (threshold < 0 || threshold > 1 || minCells < 1 || minVolume < 0 || rhoLiquid <= 0)
    {
        FatalErrorInFunction
            << "Require 0 <= threshold <= 1, minCells >= 1, minVolume >= 0 "
            << "and rhoLiquid > 0" << exit(FatalError);
    }

    volScalarField alpha
    (
        IOobject
        (
            alphaName, runTime.name(), mesh,
            IOobject::MUST_READ, IOobject::NO_WRITE
        ),
        mesh
    );

    autoPtr<volVectorField> UPtr(nullptr);
    if (UName != "none")
    {
        UPtr.reset
        (
            new volVectorField
            (
                IOobject
                (
                    UName, runTime.name(), mesh,
                    IOobject::MUST_READ, IOobject::NO_WRITE
                ),
                mesh
            )
        );
    }

    const scalarField& alphaField = alpha.internalField();
    const scalarField& cellVolumes = mesh.V();
    const vectorField& cellCentres = mesh.C();
    const labelListList& cellNeighbours = mesh.cellCells();
    vectorField velocity(mesh.nCells(), vector::zero);
    if (UPtr.valid())
    {
        velocity = UPtr().internalField();
    }

    boolList visited(mesh.nCells(), false);
    DynamicList<vofFragment> fragments;
    scalar selectedVolume = 0;
    scalar convertedVolume = 0;

    forAll(alphaField, seed)
    {
        if (visited[seed] || alphaField[seed] < threshold)
        {
            continue;
        }

        DynamicList<label> stack(32);
        DynamicList<label> component(32);
        stack.append(seed);
        visited[seed] = true;
        scalar fragmentVolume = 0;
        vector fragmentCentroid = vector::zero;
        vector fragmentVelocity = vector::zero;

        while (stack.size())
        {
            const label celli = stack.remove();
            component.append(celli);
            const scalar liquidWeight = alphaField[celli]*cellVolumes[celli];
            fragmentVolume += liquidWeight;
            fragmentCentroid += liquidWeight*cellCentres[celli];
            fragmentVelocity += liquidWeight*velocity[celli];

            const labelList& neighbours = cellNeighbours[celli];
            forAll(neighbours, nbrI)
            {
                const label nbr = neighbours[nbrI];
                if (!visited[nbr] && alphaField[nbr] >= threshold)
                {
                    visited[nbr] = true;
                    stack.append(nbr);
                }
            }
        }

        selectedVolume += fragmentVolume;
        if (component.size() < minCells || fragmentVolume < minVolume || fragmentVolume <= SMALL)
        {
            continue;
        }

        fragmentCentroid /= fragmentVolume;
        fragmentVelocity /= fragmentVolume;
        vofFragment fragment;
        fragment.cells.transfer(component);
        fragment.volume = fragmentVolume;
        fragment.centroid = fragmentCentroid;
        fragment.velocity = fragmentVelocity;
        fragments.append(fragment);
        convertedVolume += fragmentVolume;
    }

    if (fragments.empty())
    {
        FatalErrorInFunction
            << "No VOF fragment survived the threshold and filters. Selected volume = "
            << selectedVolume << exit(FatalError);
    }

    mkDir(positionsPath.path());
    OFstream positionsFile(positionsPath);
    positionsFile
        << "FoamFile\n{\n"
        << "    format ascii;\n    class vectorField;\n"
        << "    location \"constant\";\n    object " << positionsPath.name() << ";\n"
        << "}\n\n(\n";
    forAll(fragments, fragmentI)
    {
        positionsFile << fragments[fragmentI].centroid << nl;
    }
    positionsFile << ")\n";

    const fileName propertiesPath(positionsPath.path()/"vofToDpmFragments");
    OFstream propertiesFile(propertiesPath);
    propertiesFile
        << "FoamFile\n{\n"
        << "    format ascii;\n    class dictionary;\n"
        << "    location \"constant\";\n    object vofToDpmFragments;\n"
        << "}\n\nfragments\n(\n";
    forAll(fragments, fragmentI)
    {
        const vofFragment& fragment = fragments[fragmentI];
        const scalar diameter = Foam::cbrt
        (6.0*fragment.volume/constant::mathematical::pi);
        propertiesFile
            << "    { index " << fragmentI
            << "; cells " << fragment.cells.size()
            << "; volume " << fragment.volume
            << "; mass " << rhoLiquid*fragment.volume
            << "; diameter " << diameter
            << "; centroid " << fragment.centroid
            << "; velocity " << fragment.velocity << "; }\n";
    }
    propertiesFile << ");\n";

    const fileName reportPath(positionsPath.path()/"vofToDpmReport");
    OFstream reportFile(reportPath);
    reportFile
        << "FoamFile\n{\n"
        << "    format ascii;\n    class dictionary;\n"
        << "    location \"constant\";\n    object vofToDpmReport;\n"
        << "}\n\n"
        << "alphaName " << alphaName << ";\n"
        << "velocityName " << UName << ";\n"
        << "threshold " << threshold << ";\n"
        << "minCells " << minCells << ";\n"
        << "minVolume " << minVolume << ";\n"
        << "rhoLiquid " << rhoLiquid << ";\n"
        << "fragmentCount " << fragments.size() << ";\n"
        << "selectedVolume " << selectedVolume << ";\n"
        << "convertedVolume " << convertedVolume << ";\n"
        << "discardedVolume " << selectedVolume - convertedVolume << ";\n";

    Info<< "Fragments: " << fragments.size() << nl
        << "Selected volume: " << selectedVolume << nl
        << "Converted volume: " << convertedVolume << nl
        << "Discarded volume: " << selectedVolume - convertedVolume << nl
        << "Wrote positions: " << positionsPath << nl
        << "Wrote properties: " << propertiesPath << nl
        << "Wrote report: " << reportPath << nl << endl;

    return 0;
}

// ************************************************************************* //
