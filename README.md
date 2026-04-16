# Parallel Linear Algebra Library (PLA)

A header-only C++ library for linear algebra with platform-specific SIMD optimizations.

Supports:
- **x86-64** — AVX2 + FMA via `__m256d` / `__m256`
- **ARM64 / Apple Silicon** — NEON via `float64x2_t` / `float32x4_t`
- **Generic fallback** — portable scalar implementation for any other platform
- **OpenMP** parallelism (optional)

---

## Requirements

| Tool | Version |
|------|---------|
| CMake | ≥ 3.20 |
| C++ compiler | C++23 (GCC 13+, Clang 16+, Apple Clang 15+) |
| Eigen3 | any recent version |
| libomp *(macOS only)* | via Homebrew |

---

## 1. Build and Install

### Clone

```bash
git clone git@github.com:BodiyaKol/parallel_linear_algebra_lib.git
cd parallel_linear_algebra_lib
```

### Configure

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

Available options:

| Option | Default | Description |
|--------|---------|-------------|
| `USE_OPENMP` | `ON` | Enable OpenMP parallelism |
| `PLA_NATIVE_ARCH` | `ON` | Enable `-march=native` and SIMD flags |
| `PLA_BUILD_TESTS` | `ON` | Build tests (only when top-level project) |

Example — disable native arch (e.g. for a distributable build):

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DPLA_NATIVE_ARCH=OFF \
    -DUSE_OPENMP=ON
```

macOS requires Homebrew `libomp` when `USE_OPENMP=ON`:

```bash
brew install libomp
```

### Build

```bash
cmake --build build -j
```

### Install

```bash
# to /usr/local (default)
sudo cmake --install build

# or to a custom prefix
cmake --install build --prefix ~/.local
```

### Verify

```bash
ls /usr/local/lib/cmake/pla
```

Expected output:

```
plaConfig.cmake
plaConfigVersion.cmake
plaTargets.cmake
```

---

## 2. Run Tests

Tests are only built when PLA is the top-level project and `PLA_BUILD_TESTS=ON`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPLA_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

---

## 3. Using PLA in Another Project

### Project structure

```
my_app/
├── CMakeLists.txt
└── main.cpp
```

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_app LANGUAGES CXX)

find_package(pla REQUIRED)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE pla::pla)
```

> `find_package(pla)` automatically pulls in Eigen3 and OpenMP — no need to find them manually.

### main.cpp

```cpp
#include <pla>   // adjust to your actual header path

int main() {
    // your code here
}
```

### Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/app
```

### If CMake cannot find PLA

If PLA was installed to a non-standard prefix, pass it via `CMAKE_PREFIX_PATH`:

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=~/.local
```

---

## 4. Platform Notes

### x86-64 (Linux / Windows)

With `PLA_NATIVE_ARCH=ON` the library passes `-march=native -mavx2 -mfma` to consumers.
Requires a CPU with AVX2 support (Intel Haswell 2013+ or AMD Ryzen 2017+).

### Apple Silicon (arm64)

With `PLA_NATIVE_ARCH=ON` the library passes `-mcpu=native`.
OpenMP requires `brew install libomp` — Apple Clang does not bundle it.

### Other platforms

Falls back to a portable scalar implementation automatically. No configuration needed.