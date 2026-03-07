cmake_minimum_required(VERSION 3.20)

project(parallel_linear_algebra_lib LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

# --------------------------------------------------
# MAXIMUM OPTIMIZATION FLAGS
# --------------------------------------------------

set(MAX_OPT_FLAGS
    -O3
    -march=native
    -mtune=native
    -mavx2
    -mfma
    -ffast-math
    -funroll-loops
    -fomit-frame-pointer
)

# Enable LTO
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# Apply globally
add_compile_options(${MAX_OPT_FLAGS})

# --------------------------------------------------
# Eigen
# --------------------------------------------------

set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
include_directories(${EIGEN3_INCLUDE_DIR})

# --------------------------------------------------
# PLA library
# --------------------------------------------------

add_library(pla STATIC
    src/api/main_api.cpp
    src/types/matrix.cpp
    src/types/vector.cpp
)

target_include_directories(pla
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(pla PRIVATE ${MAX_OPT_FLAGS})

# --------------------------------------------------
# OpenMP (optional)
# --------------------------------------------------

option(USE_OPENMP "Enable OpenMP parallelism" OFF)

if(USE_OPENMP)
    find_package(OpenMP QUIET)

    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(pla PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(pla PRIVATE USE_OPENMP)
        message(STATUS "OpenMP enabled")
    else()
        message(WARNING "OpenMP not found")
    endif()
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

add_executable(test_multiplication
    test/matrix-matrix/multiplication/test_multiplication_gtest.cpp)

target_compile_options(test_multiplication PRIVATE ${MAX_OPT_FLAGS})

target_link_libraries(test_multiplication
    PRIVATE
    pla
    GTest::gtest_main
)


add_executable(test_multiplic_rectangular
    test/matrix-matrix/multiplication/test_multiplic_rectangular.cpp)

target_compile_options(test_multiplic_rectangular PRIVATE ${MAX_OPT_FLAGS})

target_link_libraries(test_multiplic_rectangular
    PRIVATE
    pla
    GTest::gtest_main
)

include(GoogleTest)

gtest_discover_tests(test_multiplication)
gtest_discover_tests(test_multiplic_rectangular)
