# PLA — Parallel Linear Algebra Library

Header-only C++20 library for dense linear algebra with performance comparable to Eigen.  
Single include: `#include "pla/pla.h"`

---

## Table of Contents

- [Types & Concepts](#types--concepts)
- [Vector\<Scalar\>](#vectorscalar)
- [Matrix\<Scalar\>](#matrixscalar)
- [Decompositions](#decompositions)
- [Algorithms](#algorithms)
- [Execution Policy](#execution-policy)
- [Error Handling](#error-handling)
- [Quick Examples](#quick-examples)

---

## Types & Concepts

```cpp
using Index = std::size_t;

template<typename Scalar>
concept Numeric = std::is_arithmetic_v<Scalar>;  // int, float, double, …

enum class StorageOrder { RowMajor, ColMajor };
```

All templates are constrained with `requires Numeric<Scalar>`.  
Default scalar type is `double` for both `Vector` and `Matrix`.

### Utility types

| Type | Fields | Purpose |
|------|--------|---------|
| `VectorShape` | `n` | Describes vector size |
| `MatrixShape` | `rows`, `cols` | Describes matrix shape |
| `VectorView<S>` / `ConstVectorView<S>` | `ptr`, `size`, `stride` | Non-owning view into contiguous data |
| `MatrixView<S>` / `ConstMatrixView<S>` | `ptr`, `rows`, `cols`, `ld`, `order` | Non-owning view into matrix data |
| `Status` | `code`, `message` | Return status without exceptions |
| `StatusCode` | enum | `Ok`, `InvalidArgument`, `SizeMismatch`, `SingularMatrix`, `NonConvergent`, `NotImplemented`, `BackendError` |

---

## Vector\<Scalar\>

```cpp
#include "pla/core/vector.h"  // or pla/pla.h
```

### Construction

```cpp
Vector<double> v;                      // empty
Vector<double> v(5);                   // 5 zeros
Vector<double> v(5, 1.0);             // 5 ones
Vector<double> v = {1.0, 2.0, 3.0};  // initializer list
```

### Element access

| Expression | Description |
|-----------|-------------|
| `v[i]` | Unchecked access |
| `v.at(i)` | Bounds-checked access (throws on OOB) |
| `v.data()` | Raw pointer to underlying storage |

### Properties

| Method | Returns | Description |
|--------|---------|-------------|
| `v.dimension()` | `Index` | Number of elements |
| `v.empty()` | `bool` | True if dimension is 0 |

### Arithmetic operators

```cpp
v + u,  v - u,  -v          // element-wise, returns new Vector
v * scalar,  v / scalar     // scaling
scalar * v                  // free function
v += u,  v -= u,  v *= s,  v /= s   // in-place
```

### Math methods

| Method | Returns | Description |
|--------|---------|-------------|
| `v.dot(u)` | `Scalar` | Dot product |
| `v.norm()` | `Scalar` | L2 norm |
| `v.norm_squared()` | `Scalar` | Squared L2 norm |
| `v.normalize()` | `void` | Normalizes in-place |
| `v.normalized()` | `Vector` | Returns normalized copy |
| `v.is_unit(tol)` | `bool` | Checks if unit vector (default tol `1e-9`) |

### Utilities

```cpp
v.resize(n);   // resize (does not preserve values)
v.clear();     // reset to empty
v.swap(u);     // O(1) swap
swap(v, u);    // free function

std::cout << v;  // operator<< supported
```

---

## Matrix\<Scalar\>

```cpp
#include "pla/core/matrix.h"  // or pla/pla.h
```

Memory is 64-byte aligned for SIMD efficiency.

### Construction

```cpp
Matrix<double> A;                                   // empty
Matrix<double> A(3, 4);                             // 3×4 zeros, RowMajor
Matrix<double> A(3, 4, 1.0);                        // 3×4 filled with 1.0
Matrix<double> A(3, 4, StorageOrder::ColMajor);     // column-major zeros
auto I = Matrix<double>::identity(n);               // n×n identity
```

### Element access

| Expression | Description |
|-----------|-------------|
| `A(r, c)` | Unchecked access |
| `A.at(r, c)` | Bounds-checked access (throws on OOB) |
| `A.data()` | Raw pointer to storage |

### Properties

| Method | Returns | Description |
|--------|---------|-------------|
| `A.rows()` | `Index` | Row count |
| `A.cols()` | `Index` | Column count |
| `A.size()` | `Index` | Total elements (`rows * cols`) |
| `A.order()` | `StorageOrder` | `RowMajor` or `ColMajor` |
| `A.is_square()` | `bool` | True if rows == cols |

### Arithmetic operators

```cpp
A + B,  A - B,  -A           // matrix arithmetic
A * scalar,  A / scalar      // scaling
scalar * A                   // free function
A * v                        // matrix-vector product → Vector
A * B                        // matrix-matrix product
A += B,  A -= B,  A *= s,  A /= s   // in-place
A == B,  A != B              // equality
```

### Math methods

| Method | Returns | Description |
|--------|---------|-------------|
| `A.transpose()` | `Matrix` | Returns transposed copy |
| `A.transpose_inplace()` | `void` | In-place transpose |
| `A.norm()` | `Scalar` | Frobenius norm |
| `A.row(r)` | `Vector` | Copy of row `r` |
| `A.col(c)` | `Vector` | Copy of column `c` |

### Utilities

```cpp
A.fill(value);    // fill all elements
A.set_identity(); // make into identity matrix (must be square)
A.clear();        // reset to 0×0
A.swap(B);        // O(1) swap
swap(A, B);       // free function

std::cout << A;   // operator<< supported
```

---

## Decompositions

### LU

```cpp
#include "pla/decompos/lu.h"
```

```cpp
struct LUResult<Scalar> {
    Matrix<Scalar> L;          // lower triangular
    Matrix<Scalar> U;          // upper triangular
    Matrix<Scalar> LU_packed;  // combined packed form
    std::vector<Index> perm;   // row permutation
};

LUResult<double> lu_blocked(const Matrix<double>& A, Index block_size = 32);
LUResult<double> lu_naive(const Matrix<double>& A);
```

`lu_blocked` is preferred for large matrices (uses cache-friendly blocking).

---

### QR

```cpp
#include "pla/decompos/qr.h"
```

```cpp
struct QRResult<Scalar> {
    Matrix<Scalar> Q;  // orthogonal
    Matrix<Scalar> R;  // upper triangular
};

QRResult<double> qr(const Matrix<double>& A);               // recommended
QRResult<double> qr_householder(const Matrix<double>& A);   // explicit Householder
QRResult<double> qr_givens(const Matrix<double>& A);        // explicit Givens
```

---

### Hessenberg Reduction

```cpp
#include "pla/decompos/hessenberg.h"
```

```cpp
struct HessenbergOptions<Scalar> {
    bool accumulate_q = false;   // also compute Q
    Scalar tolerance = 1e-12;
};

struct HessenbergResult<Scalar> {
    Matrix<Scalar> H;   // upper Hessenberg form
    Matrix<Scalar> Q;   // orthogonal (only if accumulate_q = true)
    bool has_q;
};

HessenbergResult<double> hessenberg_reduce(
    const Matrix<double>& A,
    const HessenbergOptions<double>& opts = {},
    HessenbergWorkspace<double>* workspace = nullptr  // optional, avoids reallocation
);

bool is_hessenberg(const Matrix<double>& H, double tolerance = 1e-10);
```

---

### Real Schur Decomposition

```cpp
#include "pla/decompos/schur.h"
```

```cpp
struct RealSchurOptions<Scalar> {
    Scalar tolerance = 1e-10;
    Index max_iterations = 1000;
    bool accumulate_u = false;
};

struct RealSchurResult<Scalar> {
    Matrix<Scalar> T;         // quasi-upper triangular (Schur form)
    Matrix<Scalar> U;         // orthogonal (only if accumulate_u = true)
    Index iterations;
    bool converged;
    bool has_u;
};

RealSchurResult<double> real_schur(const Matrix<double>& A,
                                   const RealSchurOptions<double>& opts = {});
```

---

### Determinant

```cpp
#include "pla/decompos/determinant.h"

double det = determinant(A);  // uses LU internally
```

---

## Algorithms

### Linear System Solve

```cpp
#include "pla/algorithms/solve.h"
```

Solves `Ax = b` or `AX = B` via LU decomposition.

```cpp
// Solve for single right-hand side
Vector<double> x = solve(A, b);

// Reuse existing LU factorization
LUResult<double> lu = lu_blocked(A);
Vector<double> x = solve(lu, b);

// Solve for multiple right-hand sides (AX = B)
Matrix<double> X = solve(A, B);
```

---

### Matrix Inverse

```cpp
#include "pla/algorithms/inverse.h"

Matrix<double> Ainv = inverse(A);  // throws if A is singular
```

---

### Eigenvalues & Eigenvectors

```cpp
#include "pla/algorithms/eigen.h"
```

```cpp
struct EigenOptions<Scalar> {
    Scalar tolerance = 1e-10;
    Index max_iterations = 1000;
};

// Eigenvalues only
struct EigenvaluesResult<Scalar> {
    std::vector<std::complex<Scalar>> values;
    Matrix<Scalar> schur_form;
    Index iterations;
    bool converged;
};

// Eigenvalues + eigenvectors
struct EigenResult<Scalar> {
    std::vector<std::complex<Scalar>> values;
    Matrix<Scalar> real_eigenvectors;
    std::vector<bool> has_real_eigenvector;  // false for complex-conjugate pairs
    Matrix<Scalar> schur_form;
    Index iterations;
    bool converged;
};

EigenvaluesResult<double> eigenvalues_general(const Matrix<double>& A,
                                              const EigenOptions<double>& opts = {});

EigenResult<double> eigen_general(const Matrix<double>& A,
                                  const EigenOptions<double>& opts = {});
```

> **Note:** For real matrices with complex conjugate eigenvalue pairs, the corresponding entry in `has_real_eigenvector` is `false`.

---

## Execution Policy

```cpp
#include "pla/execution_policy.h"

enum class Backend { Serial, Simd, Tbb };

struct ExecutionPolicy {
    Backend backend = Backend::Serial;
    Index thread_count = 0;
    Index block_size = 0;
};
```

Pass an `ExecutionPolicy` to algorithms that support parallel execution.

---

## Error Handling

PLA throws on invalid operations. All exceptions derive from `pla::PLAException` → `std::runtime_error`.

| Exception | When thrown |
|-----------|------------|
| `PLAException` | Base class |
| `SizeMismatchException` | Vector/matrix size mismatch |
| `ShapeMismatchException` | Matrix shape mismatch in operation |
| `NonSquareMatrixException` | Operation requires square matrix |
| `SingularMatrixException` | Matrix is singular (solve/inverse) |
| `OutOfRangeException` | `at()` called with invalid index |

```cpp
try {
    auto x = solve(A, b);
} catch (const pla::SingularMatrixException& e) {
    std::cerr << e.what();
} catch (const pla::PLAException& e) {
    std::cerr << e.what();
}
```

Alternatively, check `Status` / `StatusCode` values where returned (internal use).

---

## Quick Examples

### Basic matrix operations

```cpp
#include "pla/pla.h"

pla::Matrix<double> A(3, 3);
A.set_identity();

pla::Matrix<double> B(3, 3, 2.0);   // all 2.0
auto C = A + B;
auto D = A * B;

std::cout << D;
```

### Solve a linear system

```cpp
pla::Matrix<double> A = { /* 3×3 */ };
pla::Vector<double> b = {1.0, 2.0, 3.0};

auto x = pla::solve(A, b);

// Or reuse factorization for multiple RHS:
auto lu = pla::lu_blocked(A);
auto x1 = pla::solve(lu, b1);
auto x2 = pla::solve(lu, b2);
```

### QR decomposition

```cpp
auto [Q, R] = pla::qr(A);

// Verify: A ≈ Q * R
auto Arecon = Q * R;
```

### Eigenvalues of a real matrix

```cpp
pla::EigenOptions<double> opts;
opts.tolerance = 1e-12;

auto result = pla::eigen_general(A, opts);
if (!result.converged)
    std::cerr << "Did not converge in " << result.iterations << " iterations\n";

for (size_t i = 0; i < result.values.size(); ++i) {
    std::cout << "λ" << i << " = " << result.values[i];
    if (result.has_real_eigenvector[i])
        std::cout << "  (real eigenvector available)";
    std::cout << "\n";
}
```

### Vector operations

```cpp
pla::Vector<double> u = {1.0, 0.0, 0.0};
pla::Vector<double> v = {0.0, 1.0, 0.0};

double d = u.dot(v);          // 0.0
auto w   = u + v;
w.normalize();
std::cout << w.norm();        // 1.0
```