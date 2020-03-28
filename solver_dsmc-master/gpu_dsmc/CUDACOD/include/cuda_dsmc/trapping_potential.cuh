#ifndef TRAPPING_POTENTIAL_CUH_INCLUDED
#define TRAPPING_POTENTIAL_CUH_INCLUDED 1

// #ifdef CUDA
#include <cuda_runtime.h>
// #endif
#include "vector_math.cuh"

typedef struct trap_geo{
    double Bz;
    double B0;
} trap_geo;

__host__ __device__ double3 B(double3 r,
                              trap_geo params);

__host__ __device__ double3 dB_dx(double3 pos,
                                  trap_geo params);

__host__ __device__ double3 dB_dy(double3 pos,
                                  trap_geo params);

__host__ __device__ double3 dB_dz(double3 pos,
                                  trap_geo params);

__device__ double d_dV_dx(double3 pos,
                          trap_geo params);

__device__ double d_dV_dy(double3 pos,
                          trap_geo params);

__device__ double d_dV_dz(double3 pos,
                          trap_geo params);

#endif  // TRAPPING_POTENTIAL_CUH_INCLUDED