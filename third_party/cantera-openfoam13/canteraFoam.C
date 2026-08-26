/*---------------------------------------------------------------------------*\
  canteraFoam: OpenFOAM 13 / Cantera thermochemistry bridge
\*---------------------------------------------------------------------------*/
#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"
#include "IOdictionary.H"
#include "OFstream.H"
#include "addToRunTimeSelectionTable.H"
#include <cantera/base/Solution.h>
#include <cantera/thermo/ThermoPhase.h>
#include <cantera/transport/Transport.h>

using namespace Foam;

int main(int argc, char *argv[])
{
    argList::addNote("Evaluate Cantera thermochemistry for OpenFOAM cell states");
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    IOdictionary canteraProperties
    (
        IOobject
        (
            "canteraProperties", runTime.constant(), mesh,
            IOobject::MUST_READ, IOobject::NO_WRITE
        )
    );
    const word mechanism(canteraProperties.lookup("mechanism"));
    const word phase(canteraProperties.lookupOrDefault<word>("phase", "gri30"));
    const word composition(canteraProperties.lookupOrDefault<word>
    (
        "composition", "H2:2,O2:1,N2:3.76"
    ));
    const scalar thermodynamicPressure
    (
        canteraProperties.lookupOrDefault<scalar>("pressure", 101325.0)
    );

    volScalarField T
    (
        IOobject("T", runTime.timeName(runTime.value(), 6), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh
    );
    volScalarField p
    (
        IOobject("p", runTime.timeName(runTime.value(), 6), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh
    );

    auto solution = Cantera::newSolution(mechanism, phase);
    auto thermo = solution->thermo();
    auto transport = solution->transport();
    OFstream output(runTime.path()/"canteraThermo.csv");
    output << "cell,T_eq,p_eq,rho,cp_mass,thermal_conductivity\n";

    forAll(T, celli)
    {
        thermo->setState_TPX(T[celli], thermodynamicPressure, composition);
        thermo->equilibrate("HP");
        output << celli << ',' << thermo->temperature() << ','
               << thermo->pressure() << ',' << thermo->density() << ','
               << thermo->cp_mass() << ',' << transport->thermalConductivity() << '\n';
    }
    Info<< "Wrote " << T.size() << " Cantera cell states to "
        << runTime.path()/"canteraThermo.csv" << nl;
    return 0;
}
