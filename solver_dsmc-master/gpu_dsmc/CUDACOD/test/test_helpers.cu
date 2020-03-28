#include "test_helpers.cuh"

__host__ void uniform_prng_launcher(int num_elements,
                                    curandState *state,
                                    double *h_r) {
    double *d_r;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_r),
                               num_elements*sizeof(double)));

    g_uniform_prng<<<1,1>>>(num_elements,
                            state,
                            d_r);

    checkCudaErrors(cudaMemcpy(h_r,
                               d_r,
                               num_elements*sizeof(double),
                               cudaMemcpyDeviceToHost));

    return;
}

__global__ void g_uniform_prng(int num_elements,
                               curandState *state,
                               double *r) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elements;
         i += blockDim.x * gridDim.x) {
        r[i] = curand_uniform(state);
    }

    return;
}

__host__ void gaussian_prng(int num_elements,
                           curandState *state,
                           double *h_r) {
    double *d_r;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_r),
                               num_elements*sizeof(double)));

    g_gaussian_prng<<<num_elements,1>>>(num_elements,
                                        state,
                                        d_r);

    checkCudaErrors(cudaMemcpy(h_r,
                               d_r,
                               num_elements*sizeof(double),
                               cudaMemcpyDeviceToHost));

    return;
}

__global__ void g_gaussian_prng(int num_elements,
                                curandState *state,
                                double *r) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elements;
         i += blockDim.x * gridDim.x) {
        r[i] = curand_normal(state);
    }

    return;
}
