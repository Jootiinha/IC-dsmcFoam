#include <math.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>

#include <g3log/g3log.hpp>
#include <iostream>
#include <iomanip>

#include "vector_math.cuh"

__host__ void cu_initialise_rng_states(int n_states,
                                       curandState *state);

__global__ void g_initialise_rng_states(int n_states,
                                        curandState *state);

__device__ double3 d_gaussian_point(double mean,
                                    double std,
                                    curandState *state);
