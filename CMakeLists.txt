cmake_minimum_required(VERSION 3.20)
project(parallel_linear_algebra_lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(pla STATIC
    src/api/main_api.cpp
    src/types/matrix.cpp
    src/types/vector.cpp
)

target_include_directories(pla
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_testing()

add_executable(test_types
    test/types/test_types.cpp
)

target_link_libraries(test_types
    PRIVATE
        pla
)

add_test(NAME pla_test_types COMMAND test_types)
