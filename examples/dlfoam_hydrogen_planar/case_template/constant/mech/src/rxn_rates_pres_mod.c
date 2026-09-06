#include <math.h>
#include "header.h"
#include "rates.h"

void get_rxn_pres_mod (const double T, const double pres, const double * __restrict__ C, double * __restrict__ pres_mod) {
  // third body variable declaration
  double thd;

  // pressure dependence variable declarations
  double k0;
  double kinf;
  double Pr;

  // troe variable declarations
  double logFcent;
  double A;
  double B;

  double logT = log(T);
  double m = pres / (8.31446210e+03 * T);

  // reaction 0;
  pres_mod[0] = m + 1.5 * C[0] + 11.0 * C[4];

  // reaction 4;
  pres_mod[1] = m - 0.27 * C[0] + 2.65 * C[4];

  // reaction 6;
  pres_mod[2] = m + 1.5 * C[0] + 11.0 * C[4];

  // reaction 7;
  thd = m + 0.5 * C[14] + 2.7 * C[0] + 0.19999999999999996 * C[2] + 6.65 * C[4];
  k0 = exp(4.9266569663351575e+01 - 2.3 * logT - (2.4531450567319320e+04 / T));
  kinf = exp(2.8324168296488494e+01 + 0.9 * logT - (2.4531450567319320e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(5.70000000e-01 * exp(-T / 1.00000000e-30) + 4.30000000e-01 * exp(-T / 1.00000000e+30), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[3] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 19;
  thd = m + 0.30000000000000004 * C[0] + 9.0 * C[4];
  k0 = exp(3.0485765696181563e+01 - 1.23 * logT);
  kinf = exp(2.2260133056545676e+01 + 0.44 * logT);
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(3.30000000e-01 * exp(-T / 1.00000000e-30) + 6.70000000e-01 * exp(-T / 1.00000000e+30) + exp(-1.00000000e+30 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[4] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 20;
  pres_mod[5] = m;

  // reaction 41;
  thd = m + 0.7 * C[14] + 0.3999999999999999 * C[2] + 11.0 * C[4];
  k0 = exp(2.6714730384054395e+01 - (2.8482227371028606e+04 / T));
  kinf = exp(2.7893385380396040e+01 - (3.1486448173237804e+04 / T));
  Pr = k0 * thd / kinf;
  pres_mod[6] =  Pr / (1.0 + Pr);

} // end get_rxn_pres_mod

