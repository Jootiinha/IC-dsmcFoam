#ifndef TEST_HELPERS_CUH_INCLUDED
#define TEST_HELPERS_CUH_INCLUDED 1

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "helper_cuda.h"

#include "vector_math.cuh"

__host__ void uniform_prng_launcher(int num_elements,
                                    curandState *state,
                                    double *h_r);

__global__ void g_uniform_prng(int num_elements,
                               curandState *state,
                               double *r);

__host__ void gaussian_prng(int num_elements,
                           curandState *state,
                           double *h_r);

__global__ void g_gaussian_prng(int num_elements,
                                curandState *state,
                                double *r);

#endif // TEST_HELPERS_CUH_INCLUDED
