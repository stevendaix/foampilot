#include "collidingCloud.H"
#include "thermoCloud.H"
#include "makeParcelInjectionModels.H"
#include "makeThermoParcelInjectionModels.H"
#include "vofFragmentInjection.H"

makeInjectionModelType(vofFragmentInjection, collidingCloud);
makeInjectionModelType(vofFragmentInjection, thermoCloud);
