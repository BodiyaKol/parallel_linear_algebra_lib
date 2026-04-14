# Parallel Linear Algebra Library (PLA)

## Installation & Usage Guide

---

# 1. Build and Install the Library

## Clone repository

```bash
git clone git@github.com:BodiyaKol/parallel_linear_algebra_lib.git
cd parallel_linear_algebra_lib
```

---

## Configure

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=ON
```

---

## Build

```bash
cmake --build build -j
```

---

## Install

```bash
cmake --install build
```

---

## Verify installation

```bash
ls /usr/local/lib/cmake/pla
```

Expected files:

```
plaConfig.cmake
plaConfigVersion.cmake
plaTargets.cmake
```

---

# 2. Using PLA in another project

## Project structure

```
my_app/
├── CMakeLists.txt
└── main.cpp
```

---

## main.cpp

```cpp
#include <pla.h>

int main() {
    // your code here
}
```

---

## CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

find_package(pla REQUIRED)

add_executable(app main.cpp)

target_link_libraries(app PRIVATE pla::pla)
```

---

## Build application

```bash
cmake -S . -B build
cmake --build build
```

---

## If CMake cannot find PLA

If PLA is installed in a non-standard location:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/pla/install
```

Then build:

```bash
cmake --build build
./build/app
```