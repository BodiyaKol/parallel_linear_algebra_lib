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

# Library
add_library(pla INTERFACE
        include/pla/types/shape.h
        include/pla/core/vector_impl.hpp
        include/pla/decompos/lu_impl.hpp)

target_include_directories(pla INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(pla INTERFACE cxx_std_23)
target_link_libraries(pla INTERFACE Eigen3::Eigen)


if(USE_OPENMP)
    if(APPLE)
        execute_process(
                COMMAND brew --prefix libomp
                OUTPUT_VARIABLE LIBOMP_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        target_compile_options(pla INTERFACE
                -Xpreprocessor
                -fopenmp
        )
        target_include_directories(pla INTERFACE ${LIBOMP_PREFIX}/include)
        target_link_libraries(pla INTERFACE ${LIBOMP_PREFIX}/lib/libomp.dylib)
        target_compile_definitions(pla INTERFACE USE_OPENMP _OPENMP)
    else()
        find_package(OpenMP REQUIRED)
        target_link_libraries(pla INTERFACE OpenMP::OpenMP_CXX)
        target_compile_definitions(pla INTERFACE USE_OPENMP)
    endif()
endif()

# target_compile_options(pla PRIVATE
#     -O3
#     -ffast-math
#     -funroll-loops
#     -fomit-frame-pointer
# )

# Architecture specific optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")

    target_compile_options(pla INTERFACE
        -march=native
        -mtune=native
        -mavx2
        -mfma
    )

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")

    # Apple Silicon / ARM CPUs
    target_compile_options(pla INTERFACE
        -mcpu=native
    )

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

# FINDS ALL TESTS RECURSIVLY

function(register_tests TEST_DIR)
    file(GLOB_RECURSE TEST_SOURCES "${TEST_DIR}/test_*.cpp")

    foreach(TEST_SOURCE ${TEST_SOURCES})
        # test/opt/multiplication/test_matrix_opr.cpp → test_matrix_opr
        get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

        add_executable(${TEST_NAME} ${TEST_SOURCE})
        target_link_libraries(${TEST_NAME} PRIVATE pla GTest::gtest_main)
        gtest_discover_tests(${TEST_NAME})

        message(STATUS "Registered test: ${TEST_NAME} <- ${TEST_SOURCE}")
    endforeach()
endfunction()

register_tests(${CMAKE_CURRENT_SOURCE_DIR}/test)
