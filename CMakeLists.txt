cmake_minimum_required(VERSION 3.20)
project(parallel_linear_algebra_lib LANGUAGES CXX)

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER clang)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
include_directories(${EIGEN3_INCLUDE_DIR})

add_library(pla STATIC
    src/api/main_api.cpp
    src/types/matrix.cpp
    src/types/vector.cpp
)

target_include_directories(pla PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_options(pla PRIVATE
-march=native
-O0
-fno-tree-vectorize
-fno-slp-vectorize
)


set_source_files_properties(
    src/types/matrix.cpp
    PROPERTIES COMPILE_OPTIONS "-O0;-mavx2;-mfma;-fno-tree-vectorize;-fno-slp-vectorize;-march=native"
)
# -fopenmp

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

# 3. Turn off optimization for the pla library

enable_testing()

find_package(GTest QUIET)
if (NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/release-1.14.0.zip
    )
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(test_multiplication
    test/matrix-matrix/multiplication/test_multiplication_gtest.cpp
)

# Turn on O3
target_link_libraries(test_multiplication
    PRIVATE pla GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_multiplication)
