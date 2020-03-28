#ifndef DISTRIBUTION_EVOLUTION_HPP_INCLUDED
#define DISTRIBUTION_EVOLUTION_HPP_INCLUDED 1

#if defined(MKL)
#include <mkl.h>
#endif
#include <g3log/g3log.hpp>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "trapping_potential.hpp"
#include "vector_math.cuh"

void velocity_verlet_update(int num_atoms,
                            double dt,
                            trap_geo params,
                            cublasHandle_t cublas_handle,
                            double3 *pos,
                            double3 *vel,
                            double3 *acc);

void sympletic_euler_update(int num_atoms,
                            double dt,
                            trap_geo params,
                            cublasHandle_t cublas_handle,
                            double3 *pos,
                            double3 *vel,
                            double3 *acc);

void update_positions(int num_atoms,
                      double dt,
                      cublasHandle_t cublas_handle,
                      double3 *vel,
                      double3 *pos);

double3 update_atom_position(double dt,
                             double3 pos,
                             double3 vel);

void update_velocities(int num_atoms,
                       double dt,
                       cublasHandle_t cublas_handle,
                       double3 *acc,
                       double3 *vel);

double3 update_atom_velocity(double dt,
                             double3 vel,
                             double3 acc);

void update_accelerations(int num_atoms,
                          trap_geo params,
                          double3 *pos,
                          double3 *acc);

double3 update_atom_acceleration(trap_geo params,
                                 double3 pos);

#endif  // DISTRIBUTION_EVOLUTION_HPP_INCLUDED
