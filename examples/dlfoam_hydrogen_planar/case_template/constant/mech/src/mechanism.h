#ifndef MECHANISM_h
#define MECHANISM_h

#include <string.h>
//last_spec 0
/* Species Indexes
0  H2
1  H
2  O2
3  O
4  H2O
5  OH
6  H2O2
7  HO2
8  NO
9  N2O
10  N
11  NH
12  NNH
13  NH2
14  N2
*/

//Number of species
#define NSP 15
//Number of variables. NN = NSP + 1 (temperature)
#define NN 16
//Number of forward reactions
#define FWD_RATES 47
//Number of reversible reactions
#define REV_RATES 47
//Number of reactions with pressure modified rates
#define PRES_MOD_RATES 7

//Must be implemented by user on a per mechanism basis in mechanism.c
void set_same_initial_conditions(int, double**, double**);

#if defined (RATES_TEST) || defined (PROFILER)
    void write_jacobian_and_rates_output(int NUM);
#endif
//apply masking of ICs for cache optimized mechanisms
void apply_mask(double*);
void apply_reverse_mask(double*);
#endif

