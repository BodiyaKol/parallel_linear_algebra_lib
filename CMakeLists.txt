cmake_minimum_required(VERSION 3.20)

project(parallel_linear_algebra_lib 
    VERSION 0.1 
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

option(USE_OPENMP "Enable OpenMP parallelism" OFF)

# Architecture flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(ARCH_FLAGS -march=native -mtune=native -mavx2 -mfma)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(ARCH_FLAGS -mcpu=native)
endif()

set(OPT_FLAGS -O3 -ffast-math -funroll-loops -fomit-frame-pointer)

add_compile_options(${OPT_FLAGS} ${ARCH_FLAGS})

# Dependencies
find_package(Eigen3 REQUIRED)
if (USE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()


# Library
add_library(pla STATIC
        src/api/main_api.cpp
)

target_include_directories(pla PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(pla PUBLIC cxx_std_23)
target_link_libraries(pla PUBLIC Eigen3::Eigen)

# target_compile_options(pla PRIVATE
#     -O3
#     -ffast-math
#     -funroll-loops
#     -fomit-frame-pointer
# )

# Architecture specific optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")

    target_compile_options(pla PRIVATE
        -march=native
        -mtune=native
        -mavx2
        -mfma
    )

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")

    # Apple Silicon / ARM CPUs
    target_compile_options(pla PRIVATE
        -mcpu=native
    )

endif()


if(USE_OPENMP)
    target_link_libraries(pla PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(pla PRIVATE USE_OPENMP)
endif() 


# --------------------------------------------------
# Testing
# --------------------------------------------------

enable_testing()

find_package(GTest QUIET)

if(NOT GTest_FOUND)
    include(FetchContent)

    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/release-1.14.0.zip
    )

    FetchContent_MakeAvailable(googletest)
endif()


# square matrices multiplication test
add_executable(test_multiplication test/matrix-matrix/multiplication/test_multiplication_gtest.cpp)

target_link_libraries(test_multiplication PRIVATE pla GTest::gtest_main)

# rectangular matrices multiplication test
add_executable(test_multiplic_rectangular test/matrix-matrix/multiplication/test_multiplic_rectangular.cpp)

target_link_libraries(test_multiplic_rectangular PRIVATE pla GTest::gtest_main)

# type tests
add_executable(test_types test/types/test_types.cpp)
target_link_libraries(test_types PRIVATE pla GTest::gtest_main)
# target_compile_options(test_multiplication PRIVATE -O3 -march=native -mavx2 -mfma -ffast-math)

# LU tests
add_executable(test_lu test/matrix-matrix/lu-factorization/test_lu.cpp)
target_link_libraries(test_lu PRIVATE pla GTest::gtest_main)

# target_compile_options(test_multiplic_rectangular PRIVATE -O3 -march=native -mavx2 -mfma -ffast-math)

include(GoogleTest)

gtest_discover_tests(test_multiplication)
gtest_discover_tests(test_multiplic_rectangular)
gtest_discover_tests(test_lu)
gtest_discover_tests(test_types)
