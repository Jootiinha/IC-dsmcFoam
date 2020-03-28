#ifndef DEFINE_DEVICE_CONSTANTS_CUH_INCLUDED
#define DEFINE_DEVICE_CONSTANTS_CUH_INCLUDED 1

__constant__ double d_gs   =  0.5;             // Gyromagnetic ratio
__constant__ double d_MF   = -1.0;             // Magnetic quantum number
__constant__ double d_muB  = 9.27400915e-24;   // Bohr magneton
__constant__ double d_mass = 1.443160648e-25;  // 87Rb mass
__constant__ double d_pi   = 3.14159265;       // Pi
__constant__ double d_a    = 5.3e-9;           // Constant cross-section formula
__constant__ double d_kB   = 1.3806503e-23;    // Boltzmann's Constant
__constant__ double d_hbar = 1.05457148e-34;   // hbar

#endif  // DEFINE_DEVICE_CONSTANTS_CUH_INCLUDED