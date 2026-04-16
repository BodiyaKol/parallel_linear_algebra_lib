cmake_minimum_required(VERSION 3.16)
project(basic_usage LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(vector_usage vector_usage.cpp)
add_executable(matrix_usage matrix_usage.cpp)

find_package(OpenMP REQUIRED)
find_package(pla REQUIRED)

target_link_libraries(vector_usage PRIVATE pla::pla)
target_link_libraries(matrix_usage PRIVATE pla::pla)
