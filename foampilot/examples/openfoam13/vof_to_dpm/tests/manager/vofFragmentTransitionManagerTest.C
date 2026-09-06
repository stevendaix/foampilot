#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"
#include "vofFragmentTransitionManager.H"
#include "Pstream.H"
#include "mathematicalConstants.H"
#include "OSspecific.H"
#include <fstream>
#include <ctime>

using namespace Foam;

namespace
{

void trace(const word& event)
{
    mkDir("postProcessing");
    const fileName path
    (
        "postProcessing/managerTrace.rank"
      + name(Pstream::myProcNo())
      + ".log"
    );
    std::ofstream os(path.c_str(), std::ios_base::app);
    os << "time=" << std::time(nullptr)
       << " rank=" << Pstream::myProcNo()
       << " event=" << event << '\n';
}

enum scenarioType
{
    nominal,
    missing,
    wrongOwner,
    duplicate,
    wrongMass
};

label failures = 0;

void check(const bool condition, const word& name)
{
    if (Pstream::master())
    {
        Info<< (condition ? "PASS: " : "FAIL: ") << name << nl;
    }
    if (!condition)
    {
        ++failures;
    }
}

vofFragmentBatch makeBatch(const label timeIndex)
{
    vofFragmentBatch batch;
    batch.timeIndex = timeIndex;
    batch.fragments.setSize(4);

    forAll(batch.fragments, fragmentI)
    {
        vofGlobalFragment& fragment = batch.fragments[fragmentI];
        fragment.id = 1000 + fragmentI;
        fragment.ownerProc = fragmentI % Pstream::nProcs();
        fragment.volume = scalar(fragmentI + 1);
        fragment.mass = fragment.volume;
        fragment.globalCells.append(10 + fragmentI);
        fragment.localCells.append(10 + fragmentI);
    }

    return batch;
}

List<vofParcelConfirmation> makeConfirmations
(
    const vofFragmentBatch& batch,
    const scenarioType scenario
)
{
    List<vofParcelConfirmation> result;
    const label myProc = Pstream::myProcNo();

    forAll(batch.fragments, fragmentI)
    {
        const vofGlobalFragment& fragment =
            batch.fragments[fragmentI];

        if (fragment.ownerProc != myProc)
        {
            continue;
        }

        if
        (
            scenario == missing
         && fragment.id == batch.fragments[0].id
        )
        {
            continue;
        }

        vofParcelConfirmation c;
        c.fragmentId = fragment.id;
        c.ownerProc = fragment.ownerProc;
        c.parcelsAdded = 1;
        c.massAdded = fragment.mass;
        c.expectedMass = fragment.mass;
        c.success = true;

        if
        (
            scenario == wrongOwner
         && fragment.id == batch.fragments[0].id
        )
        {
            c.ownerProc = Pstream::nProcs() > 1
                ? (fragment.ownerProc + 1) % Pstream::nProcs()
                : -1;
        }

        if
        (
            scenario == wrongMass
         && fragment.id == batch.fragments[0].id
        )
        {
            c.massAdded *= 2;
        }

        result.append(c);

        if
        (
            scenario == duplicate
         && fragment.id == batch.fragments[0].id
        )
        {
            result.append(c);
        }
    }

    return result;
}

bool runScenario
(
    vofFragmentTransitionManager& manager,
    const scenarioType scenario,
    const bool expected
)
{
    trace("scenario.beforeBatch");
    const vofFragmentBatch batch = makeBatch(100 + scenario);
    trace("scenario.beforeConfirmations");
    const List<vofParcelConfirmation> local =
        makeConfirmations(batch, scenario);
    List<vofParcelConfirmation> localResults;

    trace("scenario.beforeReconcile");
    const bool actual = manager.reconcileConfirmationsMPI
    (
        batch,
        local,
        localResults
    );

    trace("scenario.afterReconcile");
    Pout<< "[test rank " << Pstream::myProcNo()
        << "] scenario=" << label(scenario)
        << " actual=" << actual
        << " expected=" << expected
        << " localResults=" << localResults.size()
        << nl << flush;
    const bool localShapeIsValid =
        actual == expected
     && (
            actual
         || localResults.size() <= batch.fragments.size()
        );

    Pout<< "[test rank " << Pstream::myProcNo()
        << "] scenarioCheck=" << label(scenario)
        << " valid=" << localShapeIsValid << nl << flush;
    check(localShapeIsValid, scenario == nominal
        ? "nominal_confirmation"
        : scenario == missing
        ? "missing_confirmation_rejected"
        : scenario == wrongOwner
        ? "wrong_owner_rejected"
        : scenario == duplicate
        ? "duplicate_rejected"
        : "wrong_mass_rejected");

    return localShapeIsValid;
}

}

int main(int argc, char** argv)
{
    #include "setRootCase.H"
    trace("main.afterArgList");

    trace("main.beforeTime");
    #include "createTime.H"

    Pout<< "[test rank " << Pstream::myProcNo() << "] before mesh" << nl << flush;

    trace("main.beforeMesh");
    fvMesh mesh
    (
        IOobject
        (
            fvMesh::defaultRegion,
            runTime.name(),
            runTime,
            IOobject::MUST_READ
        )
    );

    Pout<< "[test rank " << Pstream::myProcNo() << "] after mesh" << nl << flush;

    trace("main.afterMesh");
    volScalarField alpha
    (
        IOobject("alpha.test", runTime.name(), mesh,
            IOobject::NO_READ, IOobject::NO_WRITE),
        mesh,
        dimensionedScalar("alpha", dimless, 0.5)
    );

    volVectorField U
    (
        IOobject("U.test", runTime.name(), mesh,
            IOobject::NO_READ, IOobject::NO_WRITE),
        mesh,
        dimensionedVector("U", dimVelocity, Zero)
    );

    volScalarField rho
    (
        IOobject("rho.test", runTime.name(), mesh,
            IOobject::NO_READ, IOobject::NO_WRITE),
        mesh,
        dimensionedScalar("rho", dimDensity, 1)
    );

    Pout<< "[test rank " << Pstream::myProcNo() << "] before manager" << nl << flush;

    trace("main.beforeManager");
    vofFragmentTransitionManager manager
    (
        mesh,
        alpha,
        U,
        rho,
        0.5,
        1,
        0,
        1
    );

    Pout<< "[test rank " << Pstream::myProcNo() << "] before scenarios" << nl << flush;

    trace("main.beforeScenarios");
    runScenario(manager, nominal, true);
    runScenario(manager, missing, false);
    runScenario(manager, wrongOwner, false);
    runScenario(manager, duplicate, false);
    runScenario(manager, wrongMass, false);

    labelList failuresByProc(Pstream::nProcs(), 0);
    failuresByProc[Pstream::myProcNo()] = failures;
    Pstream::gatherList(failuresByProc);

    label totalFailures = 0;
    if (Pstream::master())
    {
        forAll(failuresByProc, procI)
        {
            totalFailures += failuresByProc[procI];
        }
        Info<< "vofFragmentTransitionManagerTest: "
            << (totalFailures ? "FAILED" : "PASSED")
            << " failures=" << totalFailures << nl;
    }

    labelList result(Pstream::nProcs(), 0);
    if (Pstream::master())
    {
        forAll(result, procI)
        {
            result[procI] = totalFailures;
        }
    }
    Pstream::scatterList(result);

    return result[Pstream::myProcNo()] ? 1 : 0;
}

// ************************************************************************* //
