#pragma once

#include "vector.h"
#include "exceptions.h"

#include <algorithm>
#include <cmath>
#include <ostream>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace pla {

template<typename Scalar>
    requires Numeric<Scalar>
Matrix<Scalar> Matrix<Scalar>::operator+(const Matrix& B) const {
    const Matrix<Scalar>& A = *this;
    if (A.rows() != B.rows() || A.cols() != B.cols())
        throw ShapeMismatchException(A.rows(), A.cols(), B.rows(), B.cols());

    Matrix<Scalar> result(A.rows_, A.cols_, Scalar{0}, A.order_);
    const Index sz = A.size();
    const Scalar* lhs = A.data();
    const Scalar* rhs = B.data();
    Scalar* dst = result.data();

#ifdef __AVX2__
    if constexpr (std::is_same_v<Scalar, double>) {
        Index i = 0;
        for (; i + 3 < sz; i += 4) {
            __m256d a = _mm256_load_pd(lhs + i);
            __m256d b = _mm256_load_pd(rhs + i);
            _mm256_store_pd(dst + i, _mm256_add_pd(a, b));
        }
        for (; i < sz; ++i) dst[i] = lhs[i] + rhs[i];
    } else if constexpr (std::is_same_v<Scalar, float>) {
        Index i = 0;
        for (; i + 7 < sz; i += 8) {
            __m256 a = _mm256_load_ps(lhs + i);
            __m256 b = _mm256_load_ps(rhs + i);
            _mm256_store_ps(dst + i, _mm256_add_ps(a, b));
        }
        for (; i < sz; ++i) dst[i] = lhs[i] + rhs[i];
    } else {
        for (Index i = 0; i < sz; ++i) dst[i] = lhs[i] + rhs[i];
    }
#else
    for (Index i = 0; i < sz; ++i) dst[i] = lhs[i] + rhs[i];
#endif

    return result;
}


template<typename Scalar>
    requires Numeric<Scalar>
Matrix<Scalar> Matrix<Scalar>::operator-(const Matrix& B) const {
    Matrix<Scalar>& A = *this;
    if (A.rows() != B.rows() || A.cols() != B.cols())
        throw ShapeMismatchException(A.rows(), A.cols(), B.rows(), B.cols());

    Matrix result(A.rows_, A.cols_, Scalar{0}, A.order_);

    const Index sz = A.size();
    const Scalar* lhs = A.data();
    const Scalar* rhs = B.data();
    Scalar* dst = result.data();

#ifdef __AVX2__
    if constexpr (std::is_same_v<Scalar, double>) {
        Index i = 0;
        for (; i + 3 < sz; i += 4) {
            __m256d a = _mm256_load_pd(lhs + i);
            __m256d b = _mm256_load_pd(rhs + i);
            _mm256_store_pd(dst + i, _mm256_sub_pd(a, b));
        }
        for (; i < sz; ++i) dst[i] = lhs[i] - rhs[i];
    } else if constexpr (std::is_same_v<Scalar, float>) {
        Index i = 0;
        for (; i + 7 < sz; i += 8) {
            __m256 a = _mm256_load_ps(lhs + i);
            __m256 b = _mm256_load_ps(rhs + i);
            _mm256_store_ps(dst + i, _mm256_sub_ps(a, b));
        }
        for (; i < sz; ++i) dst[i] = lhs[i] - rhs[i];
    } else {
        for (Index i = 0; i < sz; ++i) dst[i] = lhs[i] - rhs[i];
    }
#else
    for (Index i = 0; i < sz; ++i) dst[i] = lhs[i] - rhs[i];
#endif

    return result;
}


template<typename Scalar>
    requires Numeric<Scalar>
Matrix<Scalar> Matrix<Scalar>::operator-() const {
    Matrix result(rows_, cols_, Scalar{0}, order_);
    Scalar* dst = result.data();
    const Scalar* src = data();
    const Index sz = size();

#ifdef __AVX2__
    if constexpr (std::is_same_v<Scalar, double>) {
        Index i = 0;
        __m256d vneg = _mm256_set1_pd(-1.0);
        for (; i + 3 < sz; i += 4) {
            __m256d v = _mm256_load_pd(src + i);
            _mm256_store_pd(dst + i, _mm256_mul_pd(v, vneg));
        }
        for (; i < sz; ++i) dst[i] = -src[i];
    } else if constexpr (std::is_same_v<Scalar, float>) {
        Index i = 0;
        __m256 vneg = _mm256_set1_ps(-1.0f);
        for (; i + 7 < sz; i += 8) {
            __m256 v = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, _mm256_mul_ps(v, vneg));
        }
        for (; i < sz; ++i) dst[i] = -src[i];
    } else {
        for (Index i = 0; i < sz; ++i) dst[i] = -src[i];
    }
#else
    for (Index i = 0; i < sz; ++i) dst[i] = -src[i];
#endif

    return result;
}


template<typename Scalar>
    requires Numeric<Scalar>
Matrix<Scalar> Matrix<Scalar>::operator*(Scalar scalar) const {
    Matrix result(rows(), cols(), Scalar{0}, order_);
    result *= scalar;
    return result;
}


template<typename Scalar>
    requires Numeric<Scalar>
Matrix<Scalar> Matrix<Scalar>::operator/(Scalar scalar) const {
    if (std::abs(scalar) < 1e-15)
        throw InvalidScalarException("Division by zero");

    Matrix result(rows(), cols(), 0.0, order_);
    result /= scalar;

    return result;
}


template<typename Scalar>
    requires Numeric<Scalar>
Vector<Scalar> Matrix<Scalar>::operator*(const Vector<Scalar>& vec) const {
    if(cols() != vec.dimension())
        throw ShapeMismatchException(rows_, cols_, vec.dimension(), 1);

    Vector<Scalar> result(rows());
    for(Index i = 0; i < rows(); i++){
        Scalar sum = 0.0;
        for(Index j = 0; j < cols(); j++)
            sum += (*this)(i,j) * vec[j];
        result[i] = sum;
    }
    return result;
}


template<typename Scalar>
    requires Numeric<Scalar>
Matrix<Scalar> Matrix<Scalar>::operator*(const Matrix& B) const {
    const Matrix<Scalar>& A = *this;
    if (A.cols() != B.rows())
        throw ShapeMismatchException(A.rows_, A.cols_, B.rows_, B.cols_);

    Matrix<Scalar> C{A.rows_, B.cols_, Scalar{0}, A.order_};

    // hint to compiler that it's aligned to 64
    const auto* __restrict__ a = static_cast<const Scalar*>(__builtin_assume_aligned(A.data(), 64));
    const auto* __restrict__ b = static_cast<const Scalar*>(__builtin_assume_aligned(B.data(), 64));
    auto*       __restrict__ c = static_cast<      Scalar*>(__builtin_assume_aligned(C.data(), 64));

    const Index M = A.rows();
    const Index N = B.cols();
    const Index K = A.cols(); // same as B.rows()

#ifdef __AVX2__
    if constexpr (std::is_same_v<Scalar, double>) {
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

                    double* __restrict__ c0_row = c + (i+0) * N;
                    double* __restrict__ c1_row = c + (i+1) * N;
                    double* __restrict__ c2_row = c + (i+2) * N;
                    double* __restrict__ c3_row = c + (i+3) * N;

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
                        const __m256d b0 = _mm256_loadu_pd(b_row + j);
                        const __m256d b1 = _mm256_loadu_pd(b_row + j + 4);
                        const __m256d b2 = _mm256_loadu_pd(b_row + j + 8);
                        const __m256d b3 = _mm256_loadu_pd(b_row + j + 12);

                        // 1st c_row, 4 doubles, FMA
                        _mm256_storeu_pd(c0_row+j,    _mm256_fmadd_pd(va0, b0, _mm256_loadu_pd(c0_row+j)));
                        _mm256_storeu_pd(c0_row+j+4,  _mm256_fmadd_pd(va0, b1, _mm256_loadu_pd(c0_row+j+4)));
                        _mm256_storeu_pd(c0_row+j+8,  _mm256_fmadd_pd(va0, b2, _mm256_loadu_pd(c0_row+j+8)));
                        _mm256_storeu_pd(c0_row+j+12, _mm256_fmadd_pd(va0, b3, _mm256_loadu_pd(c0_row+j+12)));

                        // 2nd c_row, 4 doubles, FMA
                        _mm256_storeu_pd(c1_row+j,    _mm256_fmadd_pd(va1, b0, _mm256_loadu_pd(c1_row+j)));
                        _mm256_storeu_pd(c1_row+j+4,  _mm256_fmadd_pd(va1, b1, _mm256_loadu_pd(c1_row+j+4)));
                        _mm256_storeu_pd(c1_row+j+8,  _mm256_fmadd_pd(va1, b2, _mm256_loadu_pd(c1_row+j+8)));
                        _mm256_storeu_pd(c1_row+j+12, _mm256_fmadd_pd(va1, b3, _mm256_loadu_pd(c1_row+j+12)));

                        // 3rd row
                        _mm256_storeu_pd(c2_row+j,    _mm256_fmadd_pd(va2, b0, _mm256_loadu_pd(c2_row+j)));
                        _mm256_storeu_pd(c2_row+j+4,  _mm256_fmadd_pd(va2, b1, _mm256_loadu_pd(c2_row+j+4)));
                        _mm256_storeu_pd(c2_row+j+8,  _mm256_fmadd_pd(va2, b2, _mm256_loadu_pd(c2_row+j+8)));
                        _mm256_storeu_pd(c2_row+j+12, _mm256_fmadd_pd(va2, b3, _mm256_loadu_pd(c2_row+j+12)));

                        // 4th row
                        _mm256_storeu_pd(c3_row+j,    _mm256_fmadd_pd(va3, b0, _mm256_loadu_pd(c3_row+j)));
                        _mm256_storeu_pd(c3_row+j+4,  _mm256_fmadd_pd(va3, b1, _mm256_loadu_pd(c3_row+j+4)));
                        _mm256_storeu_pd(c3_row+j+8,  _mm256_fmadd_pd(va3, b2, _mm256_loadu_pd(c3_row+j+8)));
                        _mm256_storeu_pd(c3_row+j+12, _mm256_fmadd_pd(va3, b3, _mm256_loadu_pd(c3_row+j+12)));
                    }

                    // Remainder of > 4 doubles
                    for (; j + 3 < j_end; j += 4) {
                        const __m256d vb = _mm256_loadu_pd(b_row + j);
                        _mm256_storeu_pd(c0_row+j, _mm256_fmadd_pd(va0, vb, _mm256_loadu_pd(c0_row+j)));
                        _mm256_storeu_pd(c1_row+j, _mm256_fmadd_pd(va1, vb, _mm256_loadu_pd(c1_row+j)));
                        _mm256_storeu_pd(c2_row+j, _mm256_fmadd_pd(va2, vb, _mm256_loadu_pd(c2_row+j)));
                        _mm256_storeu_pd(c3_row+j, _mm256_fmadd_pd(va3, vb, _mm256_loadu_pd(c3_row+j)));
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

            for (; i < i_end; ++i) {
                for (Index k = kk; k < k_end; ++k) {
                    const __m256d valpha   = _mm256_set1_pd(a[i * K + k]);
                    const double* b_row    = b + k * N;
                    double*       c_row    = c + i * N;
                    Index j = jj;
                    for (; j + 15 < j_end; j += 16) {
                        __builtin_prefetch(b_row + j + 64, 0, 1);
                        __builtin_prefetch(c_row + j + 64, 1, 1);
                        _mm256_storeu_pd(c_row+j,    _mm256_fmadd_pd(valpha, _mm256_loadu_pd(b_row+j),    _mm256_loadu_pd(c_row+j)));
                        _mm256_storeu_pd(c_row+j+4,  _mm256_fmadd_pd(valpha, _mm256_loadu_pd(b_row+j+4),  _mm256_loadu_pd(c_row+j+4)));
                        _mm256_storeu_pd(c_row+j+8,  _mm256_fmadd_pd(valpha, _mm256_loadu_pd(b_row+j+8),  _mm256_loadu_pd(c_row+j+8)));
                        _mm256_storeu_pd(c_row+j+12, _mm256_fmadd_pd(valpha, _mm256_loadu_pd(b_row+j+12), _mm256_loadu_pd(c_row+j+12)));
                    }
                    for (; j + 3 < j_end; j += 4) {
                        _mm256_storeu_pd(c_row+j, _mm256_fmadd_pd(valpha, _mm256_loadu_pd(b_row+j), _mm256_loadu_pd(c_row+j)));
                    }
                    const double alpha = a[i * K + k];
                    for (; j < j_end; ++j) c_row[j] += alpha * b_row[j];
                }
            }
        }}}
    } else {
        // float та інші — scalar fallback
        for (Index i = 0; i < M; ++i)
            for (Index k = 0; k < K; ++k) {
                Scalar alpha = A(i, k);
                for (Index j = 0; j < N; ++j)
                    C(i, j) += alpha * B(k, j);
            }
    }
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


template<typename Scalar>
    requires Numeric<Scalar>
Matrix<Scalar>& Matrix<Scalar>::operator+=(const Matrix& B) {
    Matrix<Scalar>& A = *this;
    if (A.rows() != B.rows() || A.cols() != B.cols())
        throw ShapeMismatchException(A.rows(), A.cols(), B.rows(), B.cols());

    Scalar* dst = A.data();
    const Scalar* src = B.data();
    const Index sz = A.size();

#ifdef __AVX2__
    if constexpr (std::is_same_v<Scalar, double>) {
        Index i = 0;
        for (; i + 3 < sz; i += 4) {
            __m256d a = _mm256_load_pd(dst + i);
            __m256d b = _mm256_load_pd(src + i);
            _mm256_store_pd(dst + i, _mm256_add_pd(a, b));
        }
        for (; i < sz; ++i) dst[i] += src[i];
    } else if constexpr (std::is_same_v<Scalar, float>) {
        Index i = 0;
        for (; i + 7 < sz; i += 8) {
            __m256 a = _mm256_load_ps(dst + i);
            __m256 b = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, _mm256_add_ps(a, b));
        }
        for (; i < sz; ++i) dst[i] += src[i];
    } else {
        for (Index i = 0; i < sz; ++i) dst[i] += src[i];
    }
#else
    for (Index i = 0; i < sz; ++i) dst[i] += src[i];
#endif

    return *this;
}


template<typename Scalar>
    requires Numeric<Scalar>
Matrix<Scalar>& Matrix<Scalar>::operator-=(const Matrix& B) {
    Matrix<Scalar>& A = *this;
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw ShapeMismatchException(A.rows(), A.cols(), B.rows(), B.cols());
    }
    Scalar* dst = A.data();
    const Scalar* src = B.data();
    Index size = A.size();

#ifdef __AVX2__
    if constexpr (std::is_same_v<Scalar, double>) {
        Index i = 0;
        for (; i + 3 < size; i += 4) {
            __m256d a = _mm256_load_pd(dst + i);
            __m256d b = _mm256_load_pd(src + i);
            _mm256_store_pd(dst + i, _mm256_sub_pd(a, b));
        }
        for (; i < size; ++i) dst[i] -= src[i];
    } else if constexpr (std::is_same_v<Scalar, float>) {
        Index i = 0;
        for (; i + 7 < size; i += 8) {
            __m256 a = _mm256_load_ps(dst + i);
            __m256 b = _mm256_load_ps(src + i);
            _mm256_store_ps(dst + i, _mm256_sub_ps(a, b));
        }
        for (; i < size; ++i) dst[i] -= src[i];
    } else {
        for(Index i = 0; i < size; ++i) dst[i] -= src[i];
    }
#else
    for (Index i = 0; i < size; ++i) dst[i] -= src[i];
#endif

    return *this;
}


template<typename Scalar>
    requires Numeric<Scalar>
Matrix<Scalar>& Matrix<Scalar>::operator*=(Scalar scalar) {
    Scalar* ptr = this->data();
    const Index size = this->size();

#ifdef __AVX2__
    if constexpr (std::is_same_v<Scalar, double>) {
        Index i = 0;
        __m256d vscalar = _mm256_set1_pd(scalar);
        for (; i + 3 < size; i += 4) {
            __m256d v = _mm256_load_pd(ptr + i);
            v = _mm256_mul_pd(v, vscalar);
            _mm256_store_pd(ptr + i, v);
        }
    } else if constexpr (std::is_same_v<Scalar, float>) {
        Index i = 0;
        __m256 vscalar = _mm256_set1_ps(scalar);
        for (; i + 7 < size; i += 8) {
            __m256 v = _mm256_load_ps(ptr + i);
            v = _mm256_mul_ps(v, vscalar);
            _mm256_store_ps(ptr + i, v);
        }
        for (; i < size; ++i) ptr[i] *= scalar;
    } else {
        for (Index i = 0; i < size; ++i) ptr[i] *= scalar;
    }


#else
    for (Index i = 0; i < size; i++) ptr[i] *= scalar;
#endif

    return *this;
}


template<typename Scalar>
    requires Numeric<Scalar>
Matrix<Scalar>& Matrix<Scalar>::operator/=(Scalar scalar) {
    if (std::abs(scalar) < 1e-15) {
        throw InvalidScalarException("Division by zero");
    }

    Scalar* ptr = this->data();
    const Index sz = this->size();
    
#ifdef __AVX2__
    if constexpr (std::is_same_v<Scalar, double>) {
        Index i = 0;
        __m256d vscalar = _mm256_set1_pd(scalar);
        for (; i + 3 < sz; i += 4) {
            __m256d v = _mm256_load_pd(ptr + i);
            v = _mm256_div_pd(v, vscalar);
            _mm256_store_pd(ptr + i, v);
        }
        for (; i < sz; ++i) ptr[i] /= scalar;
    } else if constexpr (std::is_same_v<Scalar, float>) {
        Index i = 0;
        __m256 vscalar = _mm256_set1_ps(scalar);
        for (; i + 7 < sz; i += 8) {
            __m256 v = _mm256_load_ps(ptr + i);
            v = _mm256_div_ps(v, vscalar);
            _mm256_store_ps(ptr + i, v);
        }
        for (; i < sz; ++i) ptr[i] /= scalar;
    } else {
        for (Index i = 0; i < sz; ++i) ptr[i] /= scalar;
    }
#else
    for (Index i = 0; i < sz; ++i) ptr[i] /= scalar;
#endif

    return *this;
}


template<typename Scalar>
    requires Numeric<Scalar>
Matrix<Scalar> Matrix<Scalar>::transpose() const {
    return Matrix{};
}


template<typename Scalar>
    requires Numeric<Scalar>
void Matrix<Scalar>::transpose_inplace() {
    if (!is_square()) {
        throw NonSquareMatrixException(rows(), cols());
    }
}


template<typename Scalar>
    requires Numeric<Scalar>
Vector<Scalar> Matrix<Scalar>::row(Index r) const {
    if (r >= rows()) {
        throw IndexOutOfRangeException(r, rows());
    }
    return Vector<Scalar>{};
}


template<typename Scalar>
    requires Numeric<Scalar>
Vector<Scalar> Matrix<Scalar>::col(Index c) const {
    if (c >= cols()) {
        throw IndexOutOfRangeException(c, cols());
    }
    
    return Vector<Scalar>{};
}

template<typename Scalar>
    requires Numeric<Scalar>
void Matrix<Scalar>::fill(Scalar value) {
    if (value == Scalar{0}) std::memset(data(), 0, size() * sizeof(Scalar));
    else              std::fill(  data(), data() + size(), value);
}


template<typename Scalar>
    requires Numeric<Scalar>
void Matrix<Scalar>::set_identity() {
    if (!is_square()) {
        throw NonSquareMatrixException(rows(), cols());
    }
    
}


template<typename Scalar>
    requires Numeric<Scalar>
Matrix<Scalar> Matrix<Scalar>::identity(Index n, StorageOrder order) {
    Matrix result(n, n, 0.0, order);
    for (Index i = 0; i < n; i++) {
        result(i, i) = 1.0;
    }
    return result;
}


template<typename Scalar>
    requires Numeric<Scalar>
Scalar Matrix<Scalar>::norm() const {
    return 0.0;
}


template<typename Scalar>
Matrix<Scalar> operator*(Scalar scalar, const Matrix<Scalar>& mat) {
    return mat * scalar;
}


template<typename Scalar>
    requires Numeric<Scalar>
Scalar& Matrix<Scalar>::at(Index r, Index c) {
    if (r >= rows() || c >= cols()) {
        throw IndexOutOfRangeException(r * cols() + c, rows() * cols());
    }
    return (*this)(r, c);
}


template<typename Scalar>
    requires Numeric<Scalar>
const Scalar& Matrix<Scalar>::at(Index r, Index c) const {
    if (r >= rows() || c >= cols()) {
        throw IndexOutOfRangeException(r * cols() + c, rows() * cols());
    }
    return (*this)(r, c);
}


template<typename Scalar>
    requires Numeric<Scalar>
bool Matrix<Scalar>::operator==(const Matrix<Scalar>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_ || order_ != other.order_) return false;
    return std::equal(data(), data() + rows_ * cols_, other.data());
}


template<typename Scalar>
    requires Numeric<Scalar>
bool Matrix<Scalar>::operator!=(const Matrix<Scalar>& other) const {
    return !(*this == other);
}


template<typename Scalar>
void swap(Matrix<Scalar>& a, Matrix<Scalar>& b) noexcept {
    a.swap(b);
}


template<typename Scalar>
std::ostream& operator<<(std::ostream& os, const Matrix<Scalar>& m) {
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
