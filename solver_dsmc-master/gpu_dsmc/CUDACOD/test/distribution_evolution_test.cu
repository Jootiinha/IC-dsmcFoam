#include "distribution_evolution_tests.cuh"

double tol = 1.e-6;

SCENARIO("[DEVICE] Acceleration Update", "[d-acc]") {
    GIVEN("A thermal distribution of 5000 positions, help in a quadrupole trap with a Bz = 2.0") {
        int num_test = 5000;

        // Initialise trapping parameters
        trap_geo trap_parameters;
        trap_parameters.Bz = 2.0;
        trap_parameters.B0 = 0.;

        // Initialise rng
        curandState *state;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&state),
                                   num_test*sizeof(curandState)));
        initialise_rng_states(num_test,
                              state,
                              false);

        // Initialise positions
        double3 *d_pos;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_pos),
                                   num_test*sizeof(double3)));

        // Generate velocity distribution
        generate_thermal_positions(num_test,
                                   20.e-6,
                                   trap_parameters,
                                   state,
                                   d_pos);

        WHEN("The update_atom_accelerations function is called") {
            // Initialise accelerations
            double3 *d_test_acc;
            checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_test_acc),
                                       num_test*sizeof(double3)));

            // Generate accelerations
            update_accelerations(num_test,
                                 trap_parameters,
                                 d_pos,
                                 d_test_acc);;

            double3 *test_acc;
            test_acc = reinterpret_cast<double3*>(calloc(num_test,
                                                 sizeof(double3)));
            checkCudaErrors(cudaMemcpy(test_acc,
                                       d_test_acc,
                                       num_test*sizeof(double3),
                                       cudaMemcpyDeviceToHost));

            double mean_acc_x = mean_x(test_acc,
                                       num_test);
            double mean_acc_y = mean_y(test_acc,
                                       num_test);
            double mean_acc_z = mean_z(test_acc,
                                       num_test);

            double std_acc_x = std_dev_x(test_acc,
                                         num_test);
            double std_acc_y = std_dev_y(test_acc,
                                         num_test);
            double std_acc_z = std_dev_z(test_acc,
                                         num_test);

            THEN("The mean in each direction should be 0.") {
                REQUIRE(mean_acc_x <= 0. + std_acc_x / sqrt(num_test));
                REQUIRE(mean_acc_x >= 0. - std_acc_x / sqrt(num_test));
                REQUIRE(mean_acc_y <= 0. + std_acc_y / sqrt(num_test));
                REQUIRE(mean_acc_y >= 0. - std_acc_y / sqrt(num_test));
                REQUIRE(mean_acc_z <= 0. + std_acc_z / sqrt(num_test));
                REQUIRE(mean_acc_z >= 0. - std_acc_z / sqrt(num_test));
            }

            double expected_std_x_y = sqrt(trap_parameters.Bz*trap_parameters.Bz * gs*gs * muB*muB / 
                                           (48. * mass*mass));
            double expected_std_z = sqrt(trap_parameters.Bz*trap_parameters.Bz * gs*gs * muB*muB / 
                                           (12. * mass*mass));
            THEN("The standard deviation in each direction should be given by blah") {
                REQUIRE(std_acc_x <= expected_std_x_y + std_acc_x / sqrt(num_test));
                REQUIRE(std_acc_x >= expected_std_x_y - std_acc_x / sqrt(num_test));
                REQUIRE(std_acc_y <= expected_std_x_y + std_acc_y / sqrt(num_test));
                REQUIRE(std_acc_y >= expected_std_x_y - std_acc_y / sqrt(num_test));
                REQUIRE(std_acc_z <= expected_std_z + std_acc_z / sqrt(num_test));
                REQUIRE(std_acc_z >= expected_std_z - std_acc_z / sqrt(num_test));
            }

            cudaFree(d_test_acc);
            free(test_acc);
        }

        cudaFree(d_pos);
    }
}

SCENARIO("[DEVICE] Velocity Update", "[d-vel]") {
    GIVEN("A thermal distribution of 5000 positions, help in a quadrupole trap with a Bz = 2.0") {
        double init_T = 20.e-6;
        int num_test = 5000;

        // Initialise trapping parameters
        trap_geo trap_parameters;
        trap_parameters.Bz = 2.0;
        trap_parameters.B0 = 0.;

        // Initialise rng
        curandState *state;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&state),
                                   num_test*sizeof(curandState)));

        initialise_rng_states(num_test,
                              state,
                              false);

        // Initialise positions
        double3 *d_pos;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_pos),
                                   num_test*sizeof(double3)));

        // Generate velocity distribution
        generate_thermal_positions(num_test,
                                   20.e-6,
                                   trap_parameters,
                                   state,
                                   d_pos);

        // Initialise accelerations
        double3 *d_acc;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_acc),
                                   num_test*sizeof(double3)));

            // Generate accelerations
            update_accelerations(num_test,
                                 trap_parameters,
                                 d_pos,
                                 d_acc);

        WHEN("The update_velocities function is called with dt=1.e-6") {
            double dt = 1.e-6;
            // Initialise velocities
            double3 *d_test_vel;
            checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_test_vel),
                                       num_test*sizeof(double3)));

            // Generate velocity distribution
            generate_thermal_velocities(num_test,
                                        init_T,
                                        state,
                                        d_test_vel);

            double3 *test_vel;
            test_vel = reinterpret_cast<double3*>(calloc(num_test,
                                                  sizeof(double3)));
            checkCudaErrors(cudaMemcpy(test_vel,
                                       d_test_vel,
                                       num_test*sizeof(double3),
                                       cudaMemcpyDeviceToHost));

            double initial_kinetic_energy = mean_kinetic_energy(num_test,
                                                                test_vel);

            cublasHandle_t cublas_handle;
            checkCudaErrors(cublasCreate(&cublas_handle));
            update_velocities(num_test,
                              dt,
                              cublas_handle,
                              d_acc,
                              d_test_vel);
            cublasDestroy(cublas_handle);

            checkCudaErrors(cudaMemcpy(test_vel,
                                       d_test_vel,
                                       num_test*sizeof(double3),
                                       cudaMemcpyDeviceToHost));

            double final_kinetic_energy = mean_kinetic_energy(num_test,
                                                              test_vel);

            THEN("The change in kinetic energy should be 0") {
                REQUIRE(final_kinetic_energy - initial_kinetic_energy > -tol);
                REQUIRE(final_kinetic_energy - initial_kinetic_energy < tol);
            }

            cudaFree(d_test_vel);
            
        }

        cudaFree(d_pos);
        cudaFree(d_acc);
    }
}
