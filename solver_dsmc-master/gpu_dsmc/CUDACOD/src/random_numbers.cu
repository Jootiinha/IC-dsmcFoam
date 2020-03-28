#include <assert.h>
#include <math.h>
#include <stdint.h>

#include "random_numbers.cuh"

/** \fn __host__ void cu_initialise_rng_states(int n_states,
                                               curandState *state) 
 *  \brief Initialises and launches a kernel for generating n_state
 * curandState elements in the `state` array.
 *  \param n_states Number of rng states required.
 *  \param *state Pointer to a curandState array of length `n_states`.
 *  \exception not yet.
 *  \return void
*/

__host__ void cu_initialise_rng_states(int n_states,
                                       curandState *state) {
    LOGF(DEBUG, "\nCalculating optimal launch configuration for the state "
                "intialisation kernel.\n");
    int block_size = 0;
    int min_grid_size = 0;
    int grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                       &block_size,
                                       (const void *) g_initialise_rng_states,
                                       0,
                                       n_states);
    grid_size = (n_states + block_size - 1) / block_size;

    LOGF(DEBUG, "\nLaunch config set as <<<%i,%i>>>\n",
                grid_size, block_size);
    g_initialise_rng_states<<<grid_size,
                              block_size>>>
                           (n_states,
                            state);
    return;
}

/** \fn __global__ void g_initialise_rng_states(int n_states,
 *                                              curandState *state) 
 *  \brief Initialises the state array with `n_state` `curandSate` states for
 *  the rng
 *  \param n_states Number of rng seeds required.
 *  \param *state Pointer to a `curandState` array of length `n_seeds`.
 *  \exception not yet.
 *  \return void
*/

__global__ void g_initialise_rng_states(int n_states,
                                        curandState *state) {
    for (int id = blockIdx.x * blockDim.x + threadIdx.x;
         id < n_states;
         id += blockDim.x * gridDim.x) {
        /* Each thread gets same seed, a different sequence number, 
           no offset */
        curand_init(1234, id, 0, &state[id]);
    }
    return;
}

/** \fn __device__ double3 d_gaussian_point(double mean,
 *                                          double std,
 *                                          curandState *state) 
 *  \brief Generates a `double3` where each component is normally distributed
 *  with mean and std as the mean and standard deviation respectively.
 *  \param mean Gaussian mean of the components of the output double3.
 *  \param std Standard deviation of the components of the output double3.
 *  \param *state State for the rng.
 *  \exception not yet.
 *  \return A gaussian distributed `double3` point in cartesian space.
*/

__device__ double3 d_gaussian_point(double mean,
                                    double std,
                                    curandState *state) {
    double3 p = make_double3(0., 0., 0.);
    p.x = curand_normal(state);
    p.y = curand_normal(state);
    p.z = curand_normal(state);

    return mean + std * p;
}
