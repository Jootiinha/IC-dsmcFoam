#ifndef TEST_HELPERS_HPP_INCLUDED
#define TEST_HELPERS_HPP_INCLUDED 1

#include <cuda_runtime.h>

#include "trapping_potential.hpp"

double mean(double *array,
            int num_elements);

double mean(double3 *array,
            int num_elements);

double mean_x(double3 *array,
              int num_elements);

double mean_y(double3 *array,
              int num_elements);

double mean_z(double3 *array,
              int num_elements);

double mean_norm(double3 *array,
                 int num_elements);

double mean_modified_radius(double3 *pos,
                            int num_elements);

double std_dev(double *array,
               int num_elements);

double std_dev(double3 *array,
               int num_elements);

double std_dev_x(double3 *array,
                 int num_elements);

double std_dev_y(double3 *array,
                 int num_elements);

double std_dev_z(double3 *array,
                 int num_elements);

double std_norm(double3 *vel,
                int num_elements);

double std_modified_radius(double3 *pos,
                           int num_elements);

double z_score(double value,
               double mean,
               double std);

double mean_kinetic_energy(int num_atoms,
                           double3 *vel);

#endif // TEST_HELPERS_CUH_INCLUDED
