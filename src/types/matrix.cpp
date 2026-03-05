#include "types/matrix.h"
#include "types/vector.h"
#include "types/exceptions.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <Eigen/Dense>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace pla {


Matrix Matrix::operator+(const Matrix& other) const {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw ShapeMismatchException(rows(), cols(), other.rows(), other.cols());
    }

    Matrix result(rows(), cols(), 0.0, order_);
    for(Index i = 0; i < rows(); i++)
        for(Index j = 0; j < cols(); j++)
            result(i,j) = (*this)(i,j) + other(i,j);
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw ShapeMismatchException(rows(), cols(), other.rows(), other.cols());
    }

    Matrix result(rows(), cols(), 0.0, order_);
    for(Index i = 0; i < rows(); i++)
        for(Index j = 0; j < cols(); j++)
            result(i,j) = (*this)(i,j) - other(i,j);
    return result;
}

Matrix Matrix::operator-() const {
    Matrix result(rows(), cols(), 0.0, order_);
    for(Index i = 0; i < rows(); i++)
        for(Index j = 0; j < cols(); j++)
            result(i,j) = -(*this)(i,j);
    return result;
}

Matrix Matrix::operator*(Scalar scalar) const {
    Matrix result(rows(), cols(), 0.0, order_);
    for(Index i = 0; i < rows(); i++)
        for(Index j = 0; j < cols(); j++)
            result(i,j) = (*this)(i,j) * scalar;
    return result;
}

Matrix Matrix::operator/(Scalar scalar) const {
    if (std::abs(scalar) < 1e-15)
        throw InvalidScalarException("Division by zero");

    Matrix result(rows(), cols(), 0.0, order_);
    for(Index i = 0; i < rows(); i++)
        for(Index j = 0; j < cols(); j++)
            result(i,j) = (*this)(i,j) / scalar;
    return result;
}

Vector Matrix::operator*(const Vector& vec) const {
    if(cols() != vec.dimension())
        throw ShapeMismatchException(rows_, cols_, vec.dimension(), 1);

    Vector result(rows());
    for(Index i = 0; i < rows(); i++){
        Scalar sum = 0.0;
        for(Index j = 0; j < cols(); j++)
            sum += (*this)(i,j) * vec[j];
        result[i] = sum;
    }
    return result;
}


Matrix Matrix::operator*(const Matrix& B) const {
    const Matrix& A = *this;
    if (A.cols() != B.rows())
        throw ShapeMismatchException(A.rows_, A.cols_, B.rows_, B.cols_);

    Matrix C(A.rows_, B.cols_, 0.0, A.order_);

    // const double* __restrict__ a = A.elements_.get();
    // const double* __restrict__ b = B.elements_.get();
    // double* __restrict__ c = C.elements_.get();

    // hint to compiler that it's aligned to 64
    const double* __restrict__ a = static_cast<const double*>(__builtin_assume_aligned(A.elements_.get(), 64));
    const double* __restrict__ b = static_cast<const double*>(__builtin_assume_aligned(B.elements_.get(), 64));
    double*       __restrict__ c = static_cast<double*>      (__builtin_assume_aligned(C.elements_.get(), 64));

    const Index M = A.rows();
    const Index N = B.cols();
    const Index K = A.cols(); // same as B.rows()

#ifdef __AVX2__
    // sizeof(L1 cache) == (48KB)
    constexpr Index TILE_M = 16;
    constexpr Index TILE_K = 32;
    constexpr Index TILE_N = 64;

    // 16 * 32 = 512
    // 32 * 64 = 2048
    // 16 * 64 = 1024
    // 512 + 2048 + 1024 = 3584 elements
    // 3584 * 8 = 28672 / 1024 = 28 KB are used per block < 48 KB

#ifdef USE_OPENMP
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (Index ii = 0; ii < M; ii += TILE_M) {
    for (Index jj = 0; jj < N; jj += TILE_N) {
    for (Index kk = 0; kk < K; kk += TILE_K) {
#else
    for (Index ii = 0; ii < M; ii += TILE_M) {
    for (Index kk = 0; kk < K; kk += TILE_K) {
    for (Index jj = 0; jj < N; jj += TILE_N) {
#endif

        const Index i_end = std::min(ii + TILE_M, M);
        const Index k_end = std::min(kk + TILE_K, K);
        const Index j_end = std::min(jj + TILE_N, N);

        constexpr Index RB = 4; // lines we calculate parallely

        // per cycle calculate multiple lines
        Index i = ii;
        for (; i + (RB - 1) < i_end; i += RB) {

            for (Index k = kk; k < k_end; ++k) {
                // mm256 - size of register; copies one element to 4 parts of register; pd - packed double
                // 4 scalars from A — one per each RB rows
                const __m256d va0 = _mm256_set1_pd(a[(i+0) * K + k]);
                const __m256d va1 = _mm256_set1_pd(a[(i+1) * K + k]);
                const __m256d va2 = _mm256_set1_pd(a[(i+2) * K + k]);
                const __m256d va3 = _mm256_set1_pd(a[(i+3) * K + k]);

                const double* __restrict__ b_row = b + k * N;

                double* __restrict__ c0_row = (i+0) * N + c;
                double* __restrict__ c1_row = (i+1) * N + c;
                double* __restrict__ c2_row = (i+2) * N + c;
                double* __restrict__ c3_row = (i+3) * N + c;

                Index j = jj;

                // Main cycle: 16 doubles = 4 vectors × 4 doubles
                //  4(rows) × 4(vectors) = 16 FMA per iteration
                for (; j + 15 < j_end; j += 16) {
                    // double* addr - address to load to cache
                    // rw: 0 - read, 1 - write
                    // locality: 0-3. 1 optimal by trials and fails
                    __builtin_prefetch(b_row  + j + 64, 0, 1);
                    __builtin_prefetch(c0_row + j + 64, 1, 1);
                    __builtin_prefetch(c1_row + j + 64, 1, 1);

                    // Load 16 doubles from b_row
                    const __m256d b0 = _mm256_load_pd(b_row + j);
                    const __m256d b1 = _mm256_load_pd(b_row + j + 4);
                    const __m256d b2 = _mm256_load_pd(b_row + j + 8);
                    const __m256d b3 = _mm256_load_pd(b_row + j + 12);

                    // 1st c_row, 4 doubles, FMA
                    _mm256_store_pd(c0_row+j,    _mm256_fmadd_pd(va0, b0, _mm256_load_pd(c0_row+j)));
                    _mm256_store_pd(c0_row+j+4,  _mm256_fmadd_pd(va0, b1, _mm256_load_pd(c0_row+j+4)));
                    _mm256_store_pd(c0_row+j+8,  _mm256_fmadd_pd(va0, b2, _mm256_load_pd(c0_row+j+8)));
                    _mm256_store_pd(c0_row+j+12, _mm256_fmadd_pd(va0, b3, _mm256_load_pd(c0_row+j+12)));

                    // 2nd c_row, 4 doubles, FMA
                    _mm256_store_pd(c1_row+j,    _mm256_fmadd_pd(va1, b0, _mm256_load_pd(c1_row+j)));
                    _mm256_store_pd(c1_row+j+4,  _mm256_fmadd_pd(va1, b1, _mm256_load_pd(c1_row+j+4)));
                    _mm256_store_pd(c1_row+j+8,  _mm256_fmadd_pd(va1, b2, _mm256_load_pd(c1_row+j+8)));
                    _mm256_store_pd(c1_row+j+12, _mm256_fmadd_pd(va1, b3, _mm256_load_pd(c1_row+j+12)));

                    // 3rd row
                    _mm256_store_pd(c2_row+j,    _mm256_fmadd_pd(va2, b0, _mm256_load_pd(c2_row+j)));
                    _mm256_store_pd(c2_row+j+4,  _mm256_fmadd_pd(va2, b1, _mm256_load_pd(c2_row+j+4)));
                    _mm256_store_pd(c2_row+j+8,  _mm256_fmadd_pd(va2, b2, _mm256_load_pd(c2_row+j+8)));
                    _mm256_store_pd(c2_row+j+12, _mm256_fmadd_pd(va2, b3, _mm256_load_pd(c2_row+j+12)));

                    // 4th row
                    _mm256_store_pd(c3_row+j,    _mm256_fmadd_pd(va3, b0, _mm256_load_pd(c3_row+j)));
                    _mm256_store_pd(c3_row+j+4,  _mm256_fmadd_pd(va3, b1, _mm256_load_pd(c3_row+j+4)));
                    _mm256_store_pd(c3_row+j+8,  _mm256_fmadd_pd(va3, b2, _mm256_load_pd(c3_row+j+8)));
                    _mm256_store_pd(c3_row+j+12, _mm256_fmadd_pd(va3, b3, _mm256_load_pd(c3_row+j+12)));
                }

                // Remainder of > 4 doubles
                for (; j + 3 < j_end; j += 4) {
                    const __m256d vb = _mm256_load_pd(b_row + j);
                    _mm256_store_pd(c0_row+j, _mm256_fmadd_pd(va0, vb, _mm256_load_pd(c0_row+j)));
                    _mm256_store_pd(c1_row+j, _mm256_fmadd_pd(va1, vb, _mm256_load_pd(c1_row+j)));
                    _mm256_store_pd(c2_row+j, _mm256_fmadd_pd(va2, vb, _mm256_load_pd(c2_row+j)));
                    _mm256_store_pd(c3_row+j, _mm256_fmadd_pd(va3, vb, _mm256_load_pd(c3_row+j)));
                }

                // the smallest remainder
                const double alpha0 = a[(i+0) * K + k];
                const double alpha1 = a[(i+1) * K + k];
                const double alpha2 = a[(i+2) * K + k];
                const double alpha3 = a[(i+3) * K + k];
                for (; j < j_end; ++j) {
                    c0_row[j] += alpha0 * b_row[j];
                    c1_row[j] += alpha1 * b_row[j];
                    c2_row[j] += alpha2 * b_row[j];
                    c3_row[j] += alpha3 * b_row[j];
                }
            }
        }

        // ── Залишкові рядки (якщо TILE_M не кратне RB=4) ────────────────────
        for (; i < i_end; ++i) {
            for (Index k = kk; k < k_end; ++k) {
                const __m256d valpha   = _mm256_set1_pd(a[i * K + k]);
                const double* b_row    = b + k * N;
                double*       c_row    = c + i * N;
                Index j = jj;
                for (; j + 15 < j_end; j += 16) {
                    __builtin_prefetch(b_row + j + 64, 0, 1);
                    __builtin_prefetch(c_row + j + 64, 1, 1);
                    _mm256_store_pd(c_row+j,    _mm256_fmadd_pd(valpha, _mm256_load_pd(b_row+j),    _mm256_load_pd(c_row+j)));
                    _mm256_store_pd(c_row+j+4,  _mm256_fmadd_pd(valpha, _mm256_load_pd(b_row+j+4),  _mm256_load_pd(c_row+j+4)));
                    _mm256_store_pd(c_row+j+8,  _mm256_fmadd_pd(valpha, _mm256_load_pd(b_row+j+8),  _mm256_load_pd(c_row+j+8)));
                    _mm256_store_pd(c_row+j+12, _mm256_fmadd_pd(valpha, _mm256_load_pd(b_row+j+12), _mm256_load_pd(c_row+j+12)));
                }
                for (; j + 3 < j_end; j += 4) {
                    _mm256_store_pd(c_row+j, _mm256_fmadd_pd(valpha, _mm256_load_pd(b_row+j), _mm256_load_pd(c_row+j)));
                }
                const double alpha = a[i * K + k];
                for (; j < j_end; ++j) c_row[j] += alpha * b_row[j];
            }
        }
    }}}
#else
    for (Index i = 0; i < M; ++i) {
        for (Index k = 0; k < K; ++k) {
            Scalar alpha = A(i, k);
            for (Index j = 0; j < N; ++j) {
                C(i, j) += alpha * B(k, j);
            }
        }
    }
#endif

    return C;
}

//TODO:
Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw ShapeMismatchException(rows(), cols(), other.rows(), other.cols());
    }
    for(Index i = 0; i < rows_ * cols_; ++i)
        elements_.get()[i] += other.elements_.get()[i];
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw ShapeMismatchException(rows(), cols(), other.rows(), other.cols());
    }
    for(Index i = 0; i < rows_ * cols_; ++i)
        elements_.get()[i] -= other.elements_.get()[i];
    return *this;
}

Matrix& Matrix::operator*=(Scalar scalar) {
    return *this;
}

Matrix& Matrix::operator/=(Scalar scalar) {
    if (std::abs(scalar) < 1e-15) {
        throw InvalidScalarException("Division by zero");
    }
    return *this;
}

Matrix Matrix::transpose() const {
    return Matrix{};
}

void Matrix::transpose_inplace() {
    if (!is_square()) {
        throw NonSquareMatrixException(rows(), cols());
    }

}

Vector Matrix::row(Index r) const {
    if (r >= rows()) {
        throw IndexOutOfRangeException(r, rows());
    }
    return Vector{};
}

Vector Matrix::col(Index c) const {
    if (c >= cols()) {
        throw IndexOutOfRangeException(c, cols());
    }
    
    return Vector{};
}

void Matrix::fill(Scalar value) {
    if (value == 0.0) std::memset(elements_.get(), 0, rows_ * cols_ * sizeof(double));
    else              std::fill(  elements_.get(), elements_.get() + rows_ * cols_, value);
}

void Matrix::set_identity() {
    if (!is_square()) {
        throw NonSquareMatrixException(rows(), cols());
    }
    
}

Matrix Matrix::identity(Index n, StorageOrder order) {
    Matrix result(n, n, 0.0, order);
    for (Index i = 0; i < n; i++) {
        result(i, i) = 1.0;
    }
    return result;
}

Scalar Matrix::norm() const {
    return 0.0;
}

Matrix operator*(Scalar scalar, const Matrix& mat) {
    return mat * scalar;
}

Scalar& Matrix::at(Index r, Index c) {
    if (r >= rows() || c >= cols()) {
        throw IndexOutOfRangeException(r * cols() + c, rows() * cols());
    }
    return (*this)(r, c);
}

const Scalar& Matrix::at(Index r, Index c) const {
    if (r >= rows() || c >= cols()) {
        throw IndexOutOfRangeException(r * cols() + c, rows() * cols());
    }
    return (*this)(r, c);
}

bool Matrix::operator==(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_ || order_ != other.order_) return false;
    return std::equal(elements_.get(), elements_.get() + rows_ * cols_, other.elements_.get());
}

bool Matrix::operator!=(const Matrix& other) const {
    return !(*this == other);
}

void swap(Matrix& a, Matrix& b) noexcept {
    a.swap(b);
}

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    os << "[";
    for (Index i = 0; i < m.rows(); ++i) {
        if (i > 0) os << ",\n ";
        os << "[";
        for (Index j = 0; j < m.cols(); ++j) {
            if (j > 0) os << ", ";
            os << m(i, j);
        }
        os << "]";
    }
    os << "]";
    return os;
}

} // namespace pla
