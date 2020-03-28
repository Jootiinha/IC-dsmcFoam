#ifndef DISTRIBUTION_GENERATION_HPP_INCLUDED
#define DISTRIBUTION_GENERATION_HPP_INCLUDED 1

#include <cuda_runtime.h>

#include "trapping_potential.cuh"
#include "random_numbers.hpp"
#include "vector_math.cuh"

#ifdef CUDA
__host__ void generate_thermal_velocities(int num_atoms,
                                          double temp,
                                          curandState *state,
                                          double3 *vel);
#endif

void generate_thermal_velocities(int num_atoms,
                                 double temp,
                                 pcg32_random_t *state,
                                 double3 *vel);

double3 thermal_vel(double temp,
                    pcg32_random_t *state);

#ifdef CUDA
__host__ void generate_thermal_positions(int num_atoms,
                                         double temp,
                                         trap_geo params,
                                         curandState *state,
                                         double3 *pos);
#endif

void generate_thermal_positions(int num_atoms,
                                double temp,
                                trap_geo params,
                                pcg32_random_t *state,
                                double3 *pos);

double3 thermal_pos(double temp,
                    trap_geo params,
                    pcg32_random_t *state);

#endif  // DISTRIBUTION_GENERATION_HPP_INCLUDED
