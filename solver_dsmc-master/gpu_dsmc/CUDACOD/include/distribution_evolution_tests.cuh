#ifndef DISTRIBUTION_EVOLUTION_TESTS_CUH_INCLUDED
#define DISTRIBUTION_EVOLUTION_TESTS_CUH_INCLUDED 1

#include <cuda_runtime.h>

#include <float.h>
#include <algorithm>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "distribution_evolution.hpp"
#include "distribution_generation.hpp"
#include "random_numbers.hpp"
#include "trapping_potential.hpp"
#include "helper_cuda.h"
#include "test_helpers.hpp"
#include "test_helpers.cuh"

#include "define_host_constants.hpp"
#include "declare_device_constants.cuh"

#endif  // DISTRIBUTION_EVOLUTION_TESTS_CUH_INCLUDED
