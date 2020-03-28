#ifndef DISTRIBUTION_EVOLUTION_CUH_INCLUDED
#define DISTRIBUTION_EVOLUTION_CUH_INCLUDED 1

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "helper_cuda.h"
#include "distribution_evolution.hpp"
#include "trapping_potential.cuh"

__host__ void cu_update_positions(int num_atoms,
                                  double dt,
                                  cublasHandle_t cublas_handle,
                                  double3 *vel,
                                  double3 *pos);

__global__ void g_update_atom_position(int num_atoms,
                                       double dt,
                                       double3 *vel,
                                       double3 *pos);

__device__ double3 d_update_atom_position(double dt,
                                          double3 pos,
                                          double3 vel);

__host__ void cu_update_velocities(int num_atoms,
                                   double dt,
                                   cublasHandle_t cublas_handle,
                                   double3 *acc,
                                   double3 *vel);

__global__ void g_update_atom_velocity(int num_atoms,
                                       double dt,
                                       double3 *acc,
                                       double3 *vel);

__device__ double3 d_update_atom_velocity(double dt,
                                          double3 vel,
                                          double3 acc);

__host__ void cu_update_accelerations(int num_atoms,
                                      trap_geo params,
                                      double3 *pos,
                                      double3 *acc);

__global__ void g_update_atom_acceleration(int num_atoms,
                                           trap_geo params,
                                           double3 *pos,
                                           double3 *acc);

__device__ double3 d_update_atom_acceleration(double3 pos,
                                              trap_geo params);

#endif    // DISTRIBUTION_EVOLUTION_CUH_INCLUDED