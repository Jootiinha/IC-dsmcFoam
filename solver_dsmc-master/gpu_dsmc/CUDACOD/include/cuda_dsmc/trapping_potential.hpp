#ifndef TRAPPING_POTENTIAL_HPP_INCLUDED
#define TRAPPING_POTENTIAL_HPP_INCLUDED 1

#include "trapping_potential.cuh"

__host__ double kinetic_energy(double3 vel);

__host__ double potential_energy(double3 pos,
                                 trap_geo params);

__host__ double dV_dx(double3 pos,
                      trap_geo params);

__host__ double dV_dy(double3 pos,
                      trap_geo params);

__host__ double dV_dz(double3 pos,
                      trap_geo params);

#endif  // TRAPPING_POTENTIAL_HPP_INCLUDED
