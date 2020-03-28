#include "random_number_generation_tests.cuh"

SCENARIO("[DEVICE] Uniform random number generation", "[d-urng]") {
    GIVEN("An appropriate seed") {
        curandState *state;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&state),
                                   sizeof(curandState)));
        initialise_rng_states(1,
                              state);

        WHEN("The random number generator is called") {
            double r;
            uniform_prng_launcher(1,
                                  state,
                                  &r);

            THEN("The result should be between 0 and 1") {
                REQUIRE(r >= 0.);
                REQUIRE(r <= 1.);
            }
        }

        // WHEN("We assign the local seed to the global seed") {
        //     g_rng = rng;
        //     unif01_Gen *gen;
        //     char* rng_name = "g_uniform_prng";
        //     gen = unif01_CreateExternGen01(rng_name,
        //                                    g_uniform_prng);

        //     THEN("We expect to pass small crush") {
        //         bbattery_SmallCrush(gen);
        //         bool complete = true;
        //         REQUIRE(complete);
        //     }

        //     unif01_DeleteExternGen01(gen);
        // }

        cudaFree(state);
    }
}

SCENARIO("[DEVICE] Normally distributed random number generation", "[d-nrng]") {
    GIVEN("An array of appropriate seeds") {
        int num_test = 10000;

        curandState *state;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&state),
                                   num_test*sizeof(curandState)));
        initialise_rng_states(num_test,
                              state);

        WHEN("We generate 10,000 numbers using a mean of 0 and a std of 1") {
            double *d_test_values;
            checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_test_values),
                                       num_test*sizeof(double)));
            gaussian_prng(num_test,
                          state,
                          d_test_values);

            double *test_values;
            test_values = reinterpret_cast<double*>(calloc(num_test,
                                                    sizeof(double)));
            checkCudaErrors(cudaMemcpy(test_values,
                                       d_test_values,
                                       num_test*sizeof(double),
                                       cudaMemcpyDeviceToHost));

            THEN("The result should pass the back-of-the-envelope test") {
                double val_mean = mean(test_values,
                                       num_test);
                double val_std  = std_dev(test_values,
                                          num_test);
                double val_max = *std::max_element(test_values,
                                                   test_values+num_test);
                double val_min = *std::min_element(test_values,
                                                   test_values+num_test);

                double Z_max = z_score(val_max,
                                       val_mean,
                                       val_std);
                double Z_min = z_score(val_min,
                                       val_mean,
                                       val_std);
                REQUIRE(Z_max <= 4.);
                REQUIRE(Z_min >=-4.);
            }

            cudaFree(d_test_values);
            free(test_values);
            // Also need to implement a more rigorous test
        }

        cudaFree(state);
    }
}
