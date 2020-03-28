#include "random_number_generation_tests.hpp"

double g_uniform_prng(void);

pcg32_random_t g_rng;

SCENARIO("[HOST] Uniform random number generation", "[h-urng]") {
    GIVEN("An appropriate seed") {
        pcg32_random_t rng;
        pcg32_srandom_r(&rng, 42u, 54u);

        WHEN("The random number generator is called") {
            double r = uniform_prng(&rng);

            THEN("The result should be between 0 and 1") {
                REQUIRE(r >= 0.);
                REQUIRE(r <= 1.);
            }
        }

        WHEN("We assign the local seed to the global seed") {
            g_rng = rng;
            unif01_Gen *gen;
            char* rng_name = "g_uniform_prng";
            gen = unif01_CreateExternGen01(rng_name,
                                           g_uniform_prng);

            THEN("We expect to pass small crush") {
                bbattery_SmallCrush(gen);
                bool complete = true;
                REQUIRE(complete);
            }

            unif01_DeleteExternGen01(gen);
        }
    }
}

SCENARIO("[HOST] Normally distributed random number generation", "[h-nrng]") {
    GIVEN("An appropriate seed") {
        pcg32_random_t rng;
        pcg32_srandom_r(&rng, 42u, 54u);
        int num_test = 5000;
        // Initialise rng
        pcg32_random_t *state;
        state = reinterpret_cast<pcg32_random_t*>(calloc(num_test,
                                                         sizeof(pcg32_random_t)));

        initialise_rng_states(num_test,
                              state,
                              false);

        WHEN("We generate 5,000 numbers using a mean of 0 and a std of 1") {
            double *test_values;
            test_values = reinterpret_cast<double*>(calloc(num_test,
                                                           sizeof(double)));
            for (int i = 0; i < num_test/2; ++i) {
                // test_values[i] = gaussian_ziggurat(&state[i]);
                double2 r = box_muller(&state[i]);
                test_values[2*i] = r.x;
                test_values[2*i+1] = r.y;
            }

            THEN("The result should pass the back-of-the-envelope test") {
                double val_mean = mean(test_values,
                                       5000);
                double val_std  = std_dev(test_values,
                                          5000);
                double val_max = *std::max_element(test_values,
                                                   test_values+5000);
                double val_min = *std::min_element(test_values,
                                                   test_values+5000);

                double Z_max = z_score(val_max,
                                       val_mean,
                                       val_std);
                double Z_min = z_score(val_min,
                                       val_mean,
                                       val_std);
                REQUIRE(val_mean >= 0. - val_std/sqrt(num_test));
                REQUIRE(val_mean <= 0. + val_std/sqrt(num_test));
                REQUIRE(val_std >= 1. - val_std/sqrt(num_test));
                REQUIRE(val_std <= 1. + val_std/sqrt(num_test));
                REQUIRE(Z_max <= 4.);
                REQUIRE(Z_min >=-4.);
            }
            free(test_values);
            // Also need to implement a more rigorous test
        }
    }
}

double g_uniform_prng(void) {
    return uniform_prng(&g_rng);
}
