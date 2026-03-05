cmake_minimum_required(VERSION 3.20)
project(parallel_linear_algebra_lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Оптимізації ----
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O0 -DNDEBUG")

# ---- OpenMP ----
if(APPLE)
    set(OPENMP_FLAGS "-Xpreprocessor -fopenmp")
    set(OPENMP_LIBS "-L/opt/homebrew/opt/libomp/lib -lomp")
    include_directories(/opt/homebrew/opt/libomp/include)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENMP_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OPENMP_LIBS}")
else()
    # Linux / Windows
    find_package(OpenMP REQUIRED)
    if(OpenMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

#find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)

# ---- Бібліотека ----
add_library(pla STATIC
        src/api/main_api.cpp
        src/types/matrix.cpp
        src/types/vector.cpp
        src/matrix-matrix/lu-factorization/lu.cpp
)

target_include_directories(pla
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(pla
        PUBLIC
        Eigen3::Eigen
)

# ---- Тести ----
enable_testing()

# тест типів
add_executable(test_types
        test/types/test_types.cpp
)
target_link_libraries(test_types PRIVATE pla)
add_test(NAME pla_test_types COMMAND test_types)

# тест LU
add_executable(test_lu
        test/matrix-matrix/lu-factorization/test_lu.cpp
)
target_link_libraries(test_lu PRIVATE pla)
add_test(NAME pla_test_lu COMMAND test_lu)