cmake_minimum_required(VERSION 3.20)

project(parallel_linear_algebra_lib
        VERSION 0.1
        LANGUAGES CXX
)


option(USE_OPENMP      "Enable OpenMP parallelism"         ON)
option(PLA_NATIVE_ARCH "Enable native CPU optimizations"   ON)
option(PLA_BUILD_TESTS "Build tests"                       ON)


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

find_package(Eigen3 REQUIRED)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(pla INTERFACE)
add_library(pla::pla ALIAS pla)

target_include_directories(pla INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(pla INTERFACE cxx_std_23)

target_link_libraries(pla INTERFACE Eigen3::Eigen)



if(PLA_NATIVE_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_compile_options(pla INTERFACE
                $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
                    -march=native
                    -mtune=native
                    -mavx2
                    -mfma
                >
        )
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        target_compile_options(pla INTERFACE
                $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-mcpu=native>
        )
    endif()
endif()


if(USE_OPENMP)
    if(APPLE)
        find_program(BREW_EXECUTABLE brew)
        if(NOT BREW_EXECUTABLE)
            message(FATAL_ERROR
                "Homebrew not found. Install it or set USE_OPENMP=OFF.")
        endif()

        execute_process(
                COMMAND ${BREW_EXECUTABLE} --prefix libomp
                OUTPUT_VARIABLE LIBOMP_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE BREW_RESULT
        )

        if(NOT BREW_RESULT EQUAL 0 OR NOT LIBOMP_PREFIX)
            message(FATAL_ERROR
                "libomp not found via Homebrew. Run: brew install libomp\n"
                "Or configure with -DUSE_OPENMP=OFF")
        endif()

        find_library(LIBOMP_LIB
                NAMES omp
                HINTS "${LIBOMP_PREFIX}/lib"
                REQUIRED
        )
        find_path(LIBOMP_INCLUDE
                NAMES omp.h
                HINTS "${LIBOMP_PREFIX}/include"
                REQUIRED
        )

        target_compile_options(pla INTERFACE -Xpreprocessor -fopenmp)
        target_include_directories(pla INTERFACE "${LIBOMP_INCLUDE}")
        target_link_libraries(pla INTERFACE "${LIBOMP_LIB}")
        target_compile_definitions(pla INTERFACE USE_OPENMP _OPENMP)

    else()
        find_package(OpenMP REQUIRED)
        target_link_libraries(pla INTERFACE OpenMP::OpenMP_CXX)
        target_compile_definitions(pla INTERFACE USE_OPENMP)
    endif()
endif()



install(TARGETS pla
        EXPORT plaTargets
)

install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT plaTargets
        FILE      plaTargets.cmake
        NAMESPACE pla::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pla
)


configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/plaConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/plaConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pla
)

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/plaConfigVersion.cmake
        VERSION       ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/plaConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/plaConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pla
)

# --------------------------------------------------
# Testing — only when this is the top-level project
# or PLA_BUILD_TESTS is explicitly ON
# --------------------------------------------------
if(PROJECT_IS_TOP_LEVEL AND PLA_BUILD_TESTS)
    enable_testing()

    find_package(GTest QUIET)

    if(NOT GTest_FOUND)
        include(FetchContent)

        FetchContent_Declare(
                googletest
                URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        )

        FetchContent_MakeAvailable(googletest)
    endif()

    function(register_tests TEST_DIR)
        file(GLOB_RECURSE TEST_SOURCES "${TEST_DIR}/test_*.cpp")

        foreach(TEST_SOURCE ${TEST_SOURCES})
            get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

            add_executable(${TEST_NAME} ${TEST_SOURCE})
            target_link_libraries(${TEST_NAME} PRIVATE pla GTest::gtest_main)

            gtest_discover_tests(${TEST_NAME})

            message(STATUS "Registered test: ${TEST_NAME}")
        endforeach()
    endfunction()

    register_tests(${CMAKE_CURRENT_SOURCE_DIR}/test)
endif()