#include "mass_mole.h"

/** Function converting species mole fractions to mass fractions.
 *
 * \param[in]  X  array of species mole fractions
 * \param[out] Y  array of species mass fractions
 */
void mole2mass (const double * X, double * Y) {

  // mole fraction of final species
  double X_N;
  X_N = 1.0 - (X[0] + X[1] + X[2] + X[3] + X[4] + X[5] + X[6] + X[7] + X[8]
                + X[9] + X[10] + X[11] + X[12] + X[13]);
  // average molecular weight
  double mw_avg = 0.0;
  mw_avg += X[0] * 2.0160000000000000e+00;
  mw_avg += X[1] * 1.0080000000000000e+00;
  mw_avg += X[2] * 3.1998000000000001e+01;
  mw_avg += X[3] * 1.5999000000000001e+01;
  mw_avg += X[4] * 1.8015000000000001e+01;
  mw_avg += X[5] * 1.7007000000000001e+01;
  mw_avg += X[6] * 3.4014000000000003e+01;
  mw_avg += X[7] * 3.3006000000000000e+01;
  mw_avg += X[8] * 3.0006000000000000e+01;
  mw_avg += X[9] * 4.4012999999999998e+01;
  mw_avg += X[10] * 1.4007000000000000e+01;
  mw_avg += X[11] * 1.5015000000000001e+01;
  mw_avg += X[12] * 2.9021999999999998e+01;
  mw_avg += X[13] * 1.6023000000000000e+01;
  mw_avg += X_N * 2.8013999999999999e+01;

  // calculate mass fractions
  Y[0] = X[0] * 2.0160000000000000e+00 / mw_avg;
  Y[1] = X[1] * 1.0080000000000000e+00 / mw_avg;
  Y[2] = X[2] * 3.1998000000000001e+01 / mw_avg;
  Y[3] = X[3] * 1.5999000000000001e+01 / mw_avg;
  Y[4] = X[4] * 1.8015000000000001e+01 / mw_avg;
  Y[5] = X[5] * 1.7007000000000001e+01 / mw_avg;
  Y[6] = X[6] * 3.4014000000000003e+01 / mw_avg;
  Y[7] = X[7] * 3.3006000000000000e+01 / mw_avg;
  Y[8] = X[8] * 3.0006000000000000e+01 / mw_avg;
  Y[9] = X[9] * 4.4012999999999998e+01 / mw_avg;
  Y[10] = X[10] * 1.4007000000000000e+01 / mw_avg;
  Y[11] = X[11] * 1.5015000000000001e+01 / mw_avg;
  Y[12] = X[12] * 2.9021999999999998e+01 / mw_avg;
  Y[13] = X[13] * 1.6023000000000000e+01 / mw_avg;

} // end mole2mass

/** Function converting species mass fractions to mole fractions.
 *
 * \param[in]  Y  array of species mass fractions
 * \param[out] X  array of species mole fractions
 */
void mass2mole (const double * Y, double * X) {

  // mass fraction of final species
  double Y_N;
  Y_N = 1.0 - (Y[0] + Y[1] + Y[2] + Y[3] + Y[4] + Y[5] + Y[6] + Y[7] + Y[8]
                + Y[9] + Y[10] + Y[11] + Y[12] + Y[13]);
  // average molecular weight
  double mw_avg = 0.0;
  mw_avg += Y[0] / 2.0160000000000000e+00;
  mw_avg += Y[1] / 1.0080000000000000e+00;
  mw_avg += Y[2] / 3.1998000000000001e+01;
  mw_avg += Y[3] / 1.5999000000000001e+01;
  mw_avg += Y[4] / 1.8015000000000001e+01;
  mw_avg += Y[5] / 1.7007000000000001e+01;
  mw_avg += Y[6] / 3.4014000000000003e+01;
  mw_avg += Y[7] / 3.3006000000000000e+01;
  mw_avg += Y[8] / 3.0006000000000000e+01;
  mw_avg += Y[9] / 4.4012999999999998e+01;
  mw_avg += Y[10] / 1.4007000000000000e+01;
  mw_avg += Y[11] / 1.5015000000000001e+01;
  mw_avg += Y[12] / 2.9021999999999998e+01;
  mw_avg += Y[13] / 1.6023000000000000e+01;
  mw_avg += Y_N / 2.8013999999999999e+01;
  mw_avg = 1.0 / mw_avg;

  // calculate mole fractions
  X[0] = Y[0] * mw_avg / 2.0160000000000000e+00;
  X[1] = Y[1] * mw_avg / 1.0080000000000000e+00;
  X[2] = Y[2] * mw_avg / 3.1998000000000001e+01;
  X[3] = Y[3] * mw_avg / 1.5999000000000001e+01;
  X[4] = Y[4] * mw_avg / 1.8015000000000001e+01;
  X[5] = Y[5] * mw_avg / 1.7007000000000001e+01;
  X[6] = Y[6] * mw_avg / 3.4014000000000003e+01;
  X[7] = Y[7] * mw_avg / 3.3006000000000000e+01;
  X[8] = Y[8] * mw_avg / 3.0006000000000000e+01;
  X[9] = Y[9] * mw_avg / 4.4012999999999998e+01;
  X[10] = Y[10] * mw_avg / 1.4007000000000000e+01;
  X[11] = Y[11] * mw_avg / 1.5015000000000001e+01;
  X[12] = Y[12] * mw_avg / 2.9021999999999998e+01;
  X[13] = Y[13] * mw_avg / 1.6023000000000000e+01;

} // end mass2mole

/** Function calculating density from mole fractions.
 *
 * \param[in]  temp  temperature
 * \param[in]  pres  pressure
 * \param[in]  X     array of species mole fractions
 * \return     rho  mixture mass density
 */
double getDensity (const double temp, const double pres, const double * X) {

  // mole fraction of final species
  double X_N;
  X_N = 1.0 - (X[0] + X[1] + X[2] + X[3] + X[4] + X[5] + X[6] + X[7] + X[8]
                + X[9] + X[10] + X[11] + X[12] + X[13]);
  // average molecular weight
  double mw_avg = 0.0;
  mw_avg += X[0] * 2.0160000000000000e+00;
  mw_avg += X[1] * 1.0080000000000000e+00;
  mw_avg += X[2] * 3.1998000000000001e+01;
  mw_avg += X[3] * 1.5999000000000001e+01;
  mw_avg += X[4] * 1.8015000000000001e+01;
  mw_avg += X[5] * 1.7007000000000001e+01;
  mw_avg += X[6] * 3.4014000000000003e+01;
  mw_avg += X[7] * 3.3006000000000000e+01;
  mw_avg += X[8] * 3.0006000000000000e+01;
  mw_avg += X[9] * 4.4012999999999998e+01;
  mw_avg += X[10] * 1.4007000000000000e+01;
  mw_avg += X[11] * 1.5015000000000001e+01;
  mw_avg += X[12] * 2.9021999999999998e+01;
  mw_avg += X[13] * 1.6023000000000000e+01;
  mw_avg += X_N * 2.8013999999999999e+01;

  return pres * mw_avg / (8.31446210e+03 * temp);
} // end getDensity

