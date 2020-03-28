#Project cmake
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "Debug")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "Release")

# Add project cmake modules to path.
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

cmake_minimum_required (VERSION 2.8)
project (cuda_dsmc)

# User defined g3log library destintation
IF(G3LOG_LIBRARY_PATH)
    link_directories("${G3LOG_LIBRARY_PATH}")
ENDIF()

# User defined g3log header destintation
IF(G3LOG_INCLUDE_PATH)
    include_directories("${G3LOG_INCLUDE_PATH}")
ENDIF()

# User defined pcg library destintation
IF(PCG_LIBRARY_PATH)
    link_directories("${PCG_LIBRARY_PATH}")
ENDIF()

# User defined g3log header destintation
IF(PCG_INCLUDE_PATH)
    include_directories("${PCG_INCLUDE_PATH}")
ENDIF()

# User defined pcg library destintation
IF(TESTU01_LIBRARY_PATH)
    link_directories("${TESTU01_LIBRARY_PATH}")
ENDIF()

# User defined g3log header destintation
IF(TESTU01_INCLUDE_PATH)
    include_directories("${TESTU01_INCLUDE_PATH}")
ENDIF()


# We will optionally build with CUDA.
option(CUDA "Compile with GPU acceleration" OFF)

IF(NOT WIN32)
    find_package(CUDA QUIET REQUIRED)
    SET(CUDA_NVCC_FLAGS "-std=c++11")
    SET(CUDA_SEPARABLE_COMPILATION ON)
    SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
ENDIF()

# We will optionally build with MKL.
option(MKL "Compile with MKL acceleration" OFF)

IF(MKL)
    SET(CMAKE_C_FLAGS " -m64 -I${MKLROOT}/include ${CMAKE_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS " -m64 -I${MKLROOT}/include ${CMAKE_CXX_FLAGS}")
ENDIF()

SET(CMAKE_C_FLAGS "-std=c99 ${CMAKE_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "-std=gnu++11 ${CMAKE_CXX_FLAGS}")

include_directories("${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/include/cuda_dsmc")
IF(CUDA)
    include_directories("${CUDA_INCLUDE_DIRS}" "/usr/local/include")
ENDIF()

IF(UNIX)
    link_directories("/usr/local/lib/")
    IF(APPLE)
        link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib")
    ELSE()
        link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
    ENDIF()
ENDIF()

IF(CUDA)
    SET(SOURCES src/main.cpp src/distribution_generation.cpp src/distribution_generation.cu src/distribution_evolution.cpp src/distribution_evolution.cu src/trapping_potential.cpp src/trapping_potential.cu src/random_numbers.cpp src/random_numbers.cu src/vector_math.cu src/entropy.c)
    add_definitions(-DCUDA)
    cuda_add_executable(cuda_dsmc ${SOURCES})
    target_link_libraries(cuda_dsmc pcg_random cudart cublas g3logger)
    # target_compile_definitions(cuda_dsmc PUBLIC "CUDA")
ELSEIF(MKL)
    add_definitions(-DMKL)
    SET(SOURCES src/main.cpp src/main.cpp src/distribution_generation.cpp src/distribution_evolution.cpp src/trapping_potential.cu src/trapping_potential.cpp src/random_numbers.cpp src/vector_math.cu src/entropy.c)
    cuda_add_executable(cuda_dsmc ${SOURCES})
    target_link_libraries(cuda_dsmc gomp mkl_intel_lp64 mkl_core mkl_gnu_thread dl pthread m pcg_random g3logger)
ELSE()
    SET(SOURCES src/main.cpp src/main.cpp src/distribution_generation.cpp src/distribution_evolution.cpp src/trapping_potential.cu src/trapping_potential.cpp src/random_numbers.cpp src/vector_math.cu src/entropy.c)
    cuda_add_executable(cuda_dsmc ${SOURCES})
    target_link_libraries(cuda_dsmc pcg_random g3logger)
ENDIF()

# We probably don't want this to run on every build.
option(COVERALLS "Generate coveralls data" OFF)

IF(COVERALLS)
    include(cmake/Coveralls.cmake)
    coveralls_turn_on_coverage()
    SET(COVERAGE_SRCS "${SOURCES}")

    # Create the coveralls target.
    coveralls_setup(
        "${COVERAGE_SRCS}" # The source files.
        ON)                # If we should upload.
ENDIF()

# make run
add_custom_target(run
    COMMAND cuda_dsmc
    DEPENDS cuda_dsmc
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)

# ============================================================================
# cmake clean command
# ============================================================================
IF(NOT MSVC)
   MESSAGE("\n
      *******************************************************************
      Please do 'make clean-cmake' before next cmake generation. 
      It is a good idea to purge your build directory of CMake 
      generated cache files
      *******************************************************************
       ")
   add_custom_target(clean-cmake
      COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/CleanAll.cmake
   )
ENDIF()

# does the application run
enable_testing()

IF(CUDA)
    # DEVICE - Random number generation tests
    SET(TEST_SOURCES test/random_number_generation_test.cu src/random_numbers.cpp src/trapping_potential.cpp src/trapping_potential.cu src/random_numbers.cu src/vector_math.cu src/entropy.c test/test_helpers.cpp test/test_helpers.cu)
    cuda_add_executable(device_random_number_generation_test ${TEST_SOURCES})
    target_link_libraries(device_random_number_generation_test pcg_random cudart g3logger)
    # target_compile_definitions(device_random_number_generation_test PUBLIC "CUDA")
    # DEVICE - Distribution generation tests
    SET(TEST_SOURCES test/distribution_generation_test.cu src/distribution_generation.cpp src/distribution_generation.cu src/trapping_potential.cpp src/trapping_potential.cu src/random_numbers.cpp src/random_numbers.cu src/vector_math.cu src/entropy.c test/test_helpers.cpp test/test_helpers.cu)
    cuda_add_executable(device_distribution_generation_test ${TEST_SOURCES})
    target_link_libraries(device_distribution_generation_test pcg_random cudart g3logger)
    # target_compile_definitions(device_distribution_generation_test PUBLIC "CUDA")
    # DEVICE - Distribution evolution tests
    SET(TEST_SOURCES test/distribution_evolution_test.cu src/distribution_evolution.cpp src/distribution_evolution.cu src/distribution_generation.cpp src/distribution_generation.cu src/trapping_potential.cpp src/trapping_potential.cu src/random_numbers.cpp src/random_numbers.cu src/vector_math.cu src/entropy.c test/test_helpers.cpp test/test_helpers.cu)
    cuda_add_executable(device_distribution_evolution_test ${TEST_SOURCES})
    target_link_libraries(device_distribution_evolution_test pcg_random cudart cublas g3logger)
    # target_compile_definitions(device_distribution_evolution_test PUBLIC "CUDA")
ENDIF()

# HOST - Random number generation tests
SET(TEST_SOURCES test/random_number_generation_test.cpp src/trapping_potential.cpp src/trapping_potential.cu src/random_numbers.cpp src/random_numbers.cu src/vector_math.cu src/entropy.c test/test_helpers.cpp)
cuda_add_executable(host_random_number_generation_test ${TEST_SOURCES})
target_link_libraries(host_random_number_generation_test pcg_random cudart g3logger testu01 mylib probdist)
# HOST - Distribution generation tests
SET(TEST_SOURCES test/distribution_generation_test.cpp src/distribution_generation.cpp src/distribution_generation.cu src/trapping_potential.cpp src/trapping_potential.cu src/random_numbers.cpp src/random_numbers.cu src/vector_math.cu src/entropy.c test/test_helpers.cpp)
cuda_add_executable(host_distribution_generation_test ${TEST_SOURCES})
target_link_libraries(host_distribution_generation_test pcg_random cudart g3logger)
# HOST - Distribution evolution tests
SET(TEST_SOURCES test/distribution_evolution_test.cpp src/distribution_evolution.cpp src/distribution_evolution.cu src/distribution_generation.cpp src/distribution_generation.cu src/trapping_potential.cpp src/trapping_potential.cu src/random_numbers.cpp src/random_numbers.cu src/vector_math.cu src/entropy.c test/test_helpers.cpp)
cuda_add_executable(host_distribution_evolution_test ${TEST_SOURCES})
target_link_libraries(host_distribution_evolution_test pcg_random cudart cublas g3logger)

add_test(cuda_dmsc_runs cuda_dsmc)
add_test(host_random_number_generation host_random_number_generation_test -s -o h_rng_test.out)
add_test(host_distribution_generation host_distribution_generation_test -s -o h_dist_gen_test.out)
add_test(host_distribution_evolution host_distribution_evolution_test -s -o h_dist_ev_test.out)
IF(CUDA)
    add_test(device_random_number_generation device_random_number_generation_test -s -o d_rng_test.out)
    add_test(device_distribution_generation device_distribution_generation_test -s -o d_dist_gen_test.out)
    add_test(device_distribution_evolution device_distribution_evolution_test -s -o d_dist_ev_test.out)
ENDIF()

# does it have the correct output
# add_test (hello_world_test hello)
# set_tests_properties( hello_world_test
#   PROPERTIES PASS_REGULAR_EXPRESSION "Hello World!")
# 