
#ifndef RANDOM_NUMBER_GENERATION_TESTS_CUH_INCLUDED
#define RANDOM_NUMBER_GENERATION_TESTS_CUH_INCLUDED 1

#include <cuda_runtime.h>
#include <curand.h>

#include <float.h>
#include <algorithm>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
extern "C"
{
#include "unif01.h"
#include "bbattery.h" 
}

#include "random_numbers.hpp"
#include "helper_cuda.h"
#include "test_helpers.hpp"
#include "test_helpers.cuh"

#include "define_host_constants.hpp"

#endif  // RANDOM_NUMBER_GENERATION_TESTS_CUH_INCLUDED
