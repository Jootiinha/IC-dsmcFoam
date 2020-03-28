#ifndef DISTRIBUTION_EVOLUTION_TESTS_HPP_INCLUDED
#define DISTRIBUTION_EVOLUTION_TESTS_HPP_INCLUDED 1

#include <cuda_runtime.h>

#include <float.h>
#include <algorithm>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "random_numbers.hpp"
#include "distribution_evolution.hpp"
#include "distribution_generation.hpp"
#include "trapping_potential.hpp"
#include "test_helpers.hpp"

#include "define_host_constants.hpp"
#include "declare_device_constants.cuh"

#endif  // DISTRIBUTION_EVOLUTION_TESTS_HPP_INCLUDED
