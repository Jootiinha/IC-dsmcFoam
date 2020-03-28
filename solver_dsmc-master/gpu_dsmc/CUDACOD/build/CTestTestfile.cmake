# CMake generated Testfile for 
# Source directory: /home/joaocrm/Documentos/Pessoais/CUDA/cuda_dsmc
# Build directory: /home/joaocrm/Documentos/Pessoais/CUDA/cuda_dsmc/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
ADD_TEST(cuda_dmsc_runs "cuda_dsmc")
ADD_TEST(host_random_number_generation "host_random_number_generation_test" "-s" "-o" "h_rng_test.out")
ADD_TEST(host_distribution_generation "host_distribution_generation_test" "-s" "-o" "h_dist_gen_test.out")
ADD_TEST(host_distribution_evolution "host_distribution_evolution_test" "-s" "-o" "h_dist_ev_test.out")
