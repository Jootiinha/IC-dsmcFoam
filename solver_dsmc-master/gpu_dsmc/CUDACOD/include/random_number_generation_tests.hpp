
#ifndef RANDOM_NUMBER_GENERATION_TESTS_HPP_INCLUDED
#define RANDOM_NUMBER_GENERATION_TESTS_HPP_INCLUDED 1

#include <cuda_runtime.h>

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
#include "test_helpers.hpp"

#include "define_host_constants.hpp"

#endif  // RANDOM_NUMBER_GENERATION_TESTS_HPP_INCLUDED
