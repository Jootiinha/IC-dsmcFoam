#ifndef DECLARE_DEVICE_CONSTANTS_CUH_INCLUDED
#define DECLARE_DEVICE_CONSTANTS_CUH_INCLUDED 1

extern __constant__ double d_gs;    // Gyromagnetic ratio
extern __constant__ double d_MF;    // Magnetic quantum number
extern __constant__ double d_muB;   // Bohr magneton
extern __constant__ double d_mass;  // 87Rb mass
extern __constant__ double d_pi;    // Pi
extern __constant__ double d_a;     // Constant cross-section formula
extern __constant__ double d_kB;    // Boltzmann's Constant
extern __constant__ double d_hbar;  // hbar

#endif  // DECLARE_DEVICE_CONSTANTS_CUH_INCLUDED