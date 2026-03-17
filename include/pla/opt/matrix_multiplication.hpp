#pragma once

#include <algorithm>
#include <cmath>
#include <ostream>

#include "pla/core/matrix.h"
#include "../core/vector.h"
#include "pla/exceptions.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace pla {

template<typename Scalar>
    requires Numeric<Scalar>
Matrix<Scalar> Matrix<Scalar>::operator*(Scalar scalar) const {
    Matrix result(*this);
    result *= scalar;
    return result;
}


template<typename Scalar>
    requires Numeric<Scalar>
Matrix<Scalar> Matrix<Scalar>::operator/(Scalar scalar) const {
    if (std::abs(scalar) < 1e-15)
        throw InvalidScalarException("Division by zero");

    Matrix result(*this);
    result /= scalar;

    return result;
}


template<typename Scalar>
    requires Numeric<Scalar>
Vector<Scalar> Matrix<Scalar>::operator*(const Vector<Scalar>& vec) const {
    if(cols() != vec.dimension())
        throw ShapeMismatchException(rows_, cols_, vec.dimension(), 1);

    Vector<Scalar> result(rows());
    for(Index i = 0; i < rows(); i++) {
        Scalar sum = Scalar{0};
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
    } else if constexpr (std::is_same_v<Scalar, float>) {
        // sizeof(L1 cache) == (48KB)
        constexpr Index TILE_M = 16;
        constexpr Index TILE_K = 32;
        constexpr Index TILE_N = 128;

        // 16 * 32 = 512
        // 32 * 128 = 4096
        // 16 * 128 = 2048
        // 512 + 4096 + 2048 = 6656 elements
        // 6656 * 4 = 26624 / 1024 = 26 KB are used per block < 48 KB
        // TILE_N is 2x larger than double because float is 2x smaller —
        // __m256 holds 8 floats vs 4 doubles, so main loop processes 32 floats at once

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
                    // mm256 - size of register; copies one element to 8 parts of register; ps - packed single
                    // 4 scalars from A — one per each RB rows
                    const __m256 va0 = _mm256_set1_ps(a[(i+0) * K + k]);
                    const __m256 va1 = _mm256_set1_ps(a[(i+1) * K + k]);
                    const __m256 va2 = _mm256_set1_ps(a[(i+2) * K + k]);
                    const __m256 va3 = _mm256_set1_ps(a[(i+3) * K + k]);

                    const float* __restrict__ b_row = b + k * N;

                    float* __restrict__ c0_row = c + (i+0) * N;
                    float* __restrict__ c1_row = c + (i+1) * N;
                    float* __restrict__ c2_row = c + (i+2) * N;
                    float* __restrict__ c3_row = c + (i+3) * N;

                    Index j = jj;

                    // Main cycle: 32 floats = 4 vectors × 8 floats
                    //  4(rows) × 4(vectors) = 16 FMA per iteration
                    for (; j + 31 < j_end; j += 32) {
                        // float* addr - address to load to cache
                        // rw: 0 - read, 1 - write
                        // locality: 0-3. 1 optimal by trials and fails
                        __builtin_prefetch(b_row  + j + 128, 0, 1);
                        __builtin_prefetch(c0_row + j + 128, 1, 1);
                        __builtin_prefetch(c1_row + j + 128, 1, 1);

                        // Load 32 floats from b_row
                        const __m256 b0 = _mm256_loadu_ps(b_row + j);
                        const __m256 b1 = _mm256_loadu_ps(b_row + j + 8);
                        const __m256 b2 = _mm256_loadu_ps(b_row + j + 16);
                        const __m256 b3 = _mm256_loadu_ps(b_row + j + 24);

                        // 1st c_row, 8 floats, FMA
                        _mm256_storeu_ps(c0_row+j,    _mm256_fmadd_ps(va0, b0, _mm256_loadu_ps(c0_row+j)));
                        _mm256_storeu_ps(c0_row+j+8,  _mm256_fmadd_ps(va0, b1, _mm256_loadu_ps(c0_row+j+8)));
                        _mm256_storeu_ps(c0_row+j+16, _mm256_fmadd_ps(va0, b2, _mm256_loadu_ps(c0_row+j+16)));
                        _mm256_storeu_ps(c0_row+j+24, _mm256_fmadd_ps(va0, b3, _mm256_loadu_ps(c0_row+j+24)));

                        // 2nd c_row, 8 floats, FMA
                        _mm256_storeu_ps(c1_row+j,    _mm256_fmadd_ps(va1, b0, _mm256_loadu_ps(c1_row+j)));
                        _mm256_storeu_ps(c1_row+j+8,  _mm256_fmadd_ps(va1, b1, _mm256_loadu_ps(c1_row+j+8)));
                        _mm256_storeu_ps(c1_row+j+16, _mm256_fmadd_ps(va1, b2, _mm256_loadu_ps(c1_row+j+16)));
                        _mm256_storeu_ps(c1_row+j+24, _mm256_fmadd_ps(va1, b3, _mm256_loadu_ps(c1_row+j+24)));

                        // 3rd row
                        _mm256_storeu_ps(c2_row+j,    _mm256_fmadd_ps(va2, b0, _mm256_loadu_ps(c2_row+j)));
                        _mm256_storeu_ps(c2_row+j+8,  _mm256_fmadd_ps(va2, b1, _mm256_loadu_ps(c2_row+j+8)));
                        _mm256_storeu_ps(c2_row+j+16, _mm256_fmadd_ps(va2, b2, _mm256_loadu_ps(c2_row+j+16)));
                        _mm256_storeu_ps(c2_row+j+24, _mm256_fmadd_ps(va2, b3, _mm256_loadu_ps(c2_row+j+24)));

                        // 4th row
                        _mm256_storeu_ps(c3_row+j,    _mm256_fmadd_ps(va3, b0, _mm256_loadu_ps(c3_row+j)));
                        _mm256_storeu_ps(c3_row+j+8,  _mm256_fmadd_ps(va3, b1, _mm256_loadu_ps(c3_row+j+8)));
                        _mm256_storeu_ps(c3_row+j+16, _mm256_fmadd_ps(va3, b2, _mm256_loadu_ps(c3_row+j+16)));
                        _mm256_storeu_ps(c3_row+j+24, _mm256_fmadd_ps(va3, b3, _mm256_loadu_ps(c3_row+j+24)));
                    }

                    // Remainder of > 8 floats
                    for (; j + 7 < j_end; j += 8) {
                        const __m256 vb = _mm256_loadu_ps(b_row + j);
                        _mm256_storeu_ps(c0_row+j, _mm256_fmadd_ps(va0, vb, _mm256_loadu_ps(c0_row+j)));
                        _mm256_storeu_ps(c1_row+j, _mm256_fmadd_ps(va1, vb, _mm256_loadu_ps(c1_row+j)));
                        _mm256_storeu_ps(c2_row+j, _mm256_fmadd_ps(va2, vb, _mm256_loadu_ps(c2_row+j)));
                        _mm256_storeu_ps(c3_row+j, _mm256_fmadd_ps(va3, vb, _mm256_loadu_ps(c3_row+j)));
                    }

                    // the smallest remainder
                    const float alpha0 = a[(i+0) * K + k];
                    const float alpha1 = a[(i+1) * K + k];
                    const float alpha2 = a[(i+2) * K + k];
                    const float alpha3 = a[(i+3) * K + k];
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
                    const __m256 valpha   = _mm256_set1_ps(a[i * K + k]);
                    const float* b_row    = b + k * N;
                    float*       c_row    = c + i * N;
                    Index j = jj;
                    for (; j + 31 < j_end; j += 32) {
                        __builtin_prefetch(b_row + j + 128, 0, 1);
                        __builtin_prefetch(c_row + j + 128, 1, 1);
                        _mm256_storeu_ps(c_row+j,    _mm256_fmadd_ps(valpha, _mm256_loadu_ps(b_row+j),    _mm256_loadu_ps(c_row+j)));
                        _mm256_storeu_ps(c_row+j+8,  _mm256_fmadd_ps(valpha, _mm256_loadu_ps(b_row+j+8),  _mm256_loadu_ps(c_row+j+8)));
                        _mm256_storeu_ps(c_row+j+16, _mm256_fmadd_ps(valpha, _mm256_loadu_ps(b_row+j+16), _mm256_loadu_ps(c_row+j+16)));
                        _mm256_storeu_ps(c_row+j+24, _mm256_fmadd_ps(valpha, _mm256_loadu_ps(b_row+j+24), _mm256_loadu_ps(c_row+j+24)));
                    }
                    for (; j + 7 < j_end; j += 8) {
                        _mm256_storeu_ps(c_row+j, _mm256_fmadd_ps(valpha, _mm256_loadu_ps(b_row+j), _mm256_loadu_ps(c_row+j)));
                    }
                    const float alpha = a[i * K + k];
                    for (; j < j_end; ++j) c_row[j] += alpha * b_row[j];
                }
            }
        }}}
    } else {
        for (Index i = 0; i < M; ++i)
            for (Index k = 0; k < K; ++k) {
                Scalar alpha = A(i, k);
                for (Index j = 0; j < N; ++j)
                    C(i, j) += alpha * B(k, j);
            }
    }
#elif defined(__ARM_NEON)
    if constexpr (std::is_same_v<Scalar, double>) {
        // sizeof(L1 cache) == (192KB on M1 p-core)
        constexpr Index TILE_M = 8;
        constexpr Index TILE_K = 32;
        constexpr Index TILE_N = 32;

        // 8 * 32 = 256
        // 32 * 32 = 1024
        // 8 * 32 = 256
        // 256 + 1024 + 256 = 1536 elements
        // 1536 * 8 = 12288 / 1024 = 12 KB are used per block < 192 KB

        for (Index ii = 0; ii < M; ii += TILE_M) {
        for (Index kk = 0; kk < K; kk += TILE_K) {
        for (Index jj = 0; jj < N; jj += TILE_N) {

            const Index i_end = std::min(ii + TILE_M, M);
            const Index k_end = std::min(kk + TILE_K, K);
            const Index j_end = std::min(jj + TILE_N, N);

            constexpr Index RB = 4; // lines we calculate parallely

            // per cycle calculate multiple lines
            Index i = ii;
            for (; i + (RB - 1) < i_end; i += RB) {

                for (Index k = kk; k < k_end; ++k) {
                    // float64x2_t - 128-bit register; vdupq_n_f64 copies one element to 2 lanes; pd analog
                    // 4 scalars from A — one per each RB rows
                    const float64x2_t va0 = vdupq_n_f64(a[(i+0) * K + k]);
                    const float64x2_t va1 = vdupq_n_f64(a[(i+1) * K + k]);
                    const float64x2_t va2 = vdupq_n_f64(a[(i+2) * K + k]);
                    const float64x2_t va3 = vdupq_n_f64(a[(i+3) * K + k]);

                    const double* __restrict__ b_row = b + k * N;

                    double* __restrict__ c0_row = c + (i+0) * N;
                    double* __restrict__ c1_row = c + (i+1) * N;
                    double* __restrict__ c2_row = c + (i+2) * N;
                    double* __restrict__ c3_row = c + (i+3) * N;

                    Index j = jj;

                    // Main cycle: 8 doubles = 4 vectors × 2 doubles
                    //  4(rows) × 4(vectors) = 16 FMA per iteration
                    for (; j + 7 < j_end; j += 8) {
                        // double* addr - address to load to cache
                        // rw: 0 - read, 1 - write
                        // locality: 0-3. 1 optimal by trials and fails
                        __builtin_prefetch(b_row  + j + 32, 0, 1);
                        __builtin_prefetch(c0_row + j + 32, 1, 1);
                        __builtin_prefetch(c1_row + j + 32, 1, 1);

                        // Load 8 doubles from b_row (4 vectors × 2 doubles)
                        const float64x2_t b0 = vld1q_f64(b_row + j);
                        const float64x2_t b1 = vld1q_f64(b_row + j + 2);
                        const float64x2_t b2 = vld1q_f64(b_row + j + 4);
                        const float64x2_t b3 = vld1q_f64(b_row + j + 6);

                        // 1st c_row, 2 doubles, FMA
                        // NOTE: vfmaq_f64(c, a, b) == c + a*b  (accumulator is FIRST, unlike _mm256_fmadd_pd)
                        vst1q_f64(c0_row+j,   vfmaq_f64(vld1q_f64(c0_row+j),   va0, b0));
                        vst1q_f64(c0_row+j+2, vfmaq_f64(vld1q_f64(c0_row+j+2), va0, b1));
                        vst1q_f64(c0_row+j+4, vfmaq_f64(vld1q_f64(c0_row+j+4), va0, b2));
                        vst1q_f64(c0_row+j+6, vfmaq_f64(vld1q_f64(c0_row+j+6), va0, b3));

                        // 2nd c_row, 2 doubles, FMA
                        vst1q_f64(c1_row+j,   vfmaq_f64(vld1q_f64(c1_row+j),   va1, b0));
                        vst1q_f64(c1_row+j+2, vfmaq_f64(vld1q_f64(c1_row+j+2), va1, b1));
                        vst1q_f64(c1_row+j+4, vfmaq_f64(vld1q_f64(c1_row+j+4), va1, b2));
                        vst1q_f64(c1_row+j+6, vfmaq_f64(vld1q_f64(c1_row+j+6), va1, b3));

                        // 3rd row
                        vst1q_f64(c2_row+j,   vfmaq_f64(vld1q_f64(c2_row+j),   va2, b0));
                        vst1q_f64(c2_row+j+2, vfmaq_f64(vld1q_f64(c2_row+j+2), va2, b1));
                        vst1q_f64(c2_row+j+4, vfmaq_f64(vld1q_f64(c2_row+j+4), va2, b2));
                        vst1q_f64(c2_row+j+6, vfmaq_f64(vld1q_f64(c2_row+j+6), va2, b3));

                        // 4th row
                        vst1q_f64(c3_row+j,   vfmaq_f64(vld1q_f64(c3_row+j),   va3, b0));
                        vst1q_f64(c3_row+j+2, vfmaq_f64(vld1q_f64(c3_row+j+2), va3, b1));
                        vst1q_f64(c3_row+j+4, vfmaq_f64(vld1q_f64(c3_row+j+4), va3, b2));
                        vst1q_f64(c3_row+j+6, vfmaq_f64(vld1q_f64(c3_row+j+6), va3, b3));
                    }

                    // Remainder of > 2 doubles
                    for (; j + 1 < j_end; j += 2) {
                        const float64x2_t vb = vld1q_f64(b_row + j);
                        vst1q_f64(c0_row+j, vfmaq_f64(vld1q_f64(c0_row+j), va0, vb));
                        vst1q_f64(c1_row+j, vfmaq_f64(vld1q_f64(c1_row+j), va1, vb));
                        vst1q_f64(c2_row+j, vfmaq_f64(vld1q_f64(c2_row+j), va2, vb));
                        vst1q_f64(c3_row+j, vfmaq_f64(vld1q_f64(c3_row+j), va3, vb));
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
                    const float64x2_t valpha = vdupq_n_f64(a[i * K + k]);
                    const double* b_row      = b + k * N;
                    double*       c_row      = c + i * N;
                    Index j = jj;
                    for (; j + 7 < j_end; j += 8) {
                        __builtin_prefetch(b_row + j + 32, 0, 1);
                        __builtin_prefetch(c_row + j + 32, 1, 1);
                        vst1q_f64(c_row+j,   vfmaq_f64(vld1q_f64(c_row+j),   valpha, vld1q_f64(b_row+j)));
                        vst1q_f64(c_row+j+2, vfmaq_f64(vld1q_f64(c_row+j+2), valpha, vld1q_f64(b_row+j+2)));
                        vst1q_f64(c_row+j+4, vfmaq_f64(vld1q_f64(c_row+j+4), valpha, vld1q_f64(b_row+j+4)));
                        vst1q_f64(c_row+j+6, vfmaq_f64(vld1q_f64(c_row+j+6), valpha, vld1q_f64(b_row+j+6)));
                    }
                    for (; j + 1 < j_end; j += 2) {
                        vst1q_f64(c_row+j, vfmaq_f64(vld1q_f64(c_row+j), valpha, vld1q_f64(b_row+j)));
                    }
                    const double alpha = a[i * K + k];
                    for (; j < j_end; ++j) c_row[j] += alpha * b_row[j];
                }
            }
        }}}
    } else if constexpr (std::is_same_v<Scalar, float>) {
        // sizeof(L1 cache) == (192KB on M1 p-core)
        constexpr Index TILE_M = 8;
        constexpr Index TILE_K = 32;
        constexpr Index TILE_N = 64;

        // 8 * 32 = 256
        // 32 * 64 = 2048
        // 8 * 64 = 512
        // 256 + 2048 + 512 = 2816 elements
        // 2816 * 4 = 11264 / 1024 = 11 KB are used per block < 192 KB
        // TILE_N is 2x larger than double because float32x4_t holds 4 floats vs 2 doubles

        for (Index ii = 0; ii < M; ii += TILE_M) {
        for (Index kk = 0; kk < K; kk += TILE_K) {
        for (Index jj = 0; jj < N; jj += TILE_N) {

            const Index i_end = std::min(ii + TILE_M, M);
            const Index k_end = std::min(kk + TILE_K, K);
            const Index j_end = std::min(jj + TILE_N, N);

            constexpr Index RB = 4; // lines we calculate parallely

            // per cycle calculate multiple lines
            Index i = ii;
            for (; i + (RB - 1) < i_end; i += RB) {

                for (Index k = kk; k < k_end; ++k) {
                    // float32x4_t - 128-bit register; vdupq_n_f32 copies one element to 4 lanes; ps analog
                    // 4 scalars from A — one per each RB rows
                    const float32x4_t va0 = vdupq_n_f32(a[(i+0) * K + k]);
                    const float32x4_t va1 = vdupq_n_f32(a[(i+1) * K + k]);
                    const float32x4_t va2 = vdupq_n_f32(a[(i+2) * K + k]);
                    const float32x4_t va3 = vdupq_n_f32(a[(i+3) * K + k]);

                    const float* __restrict__ b_row = b + k * N;

                    float* __restrict__ c0_row = c + (i+0) * N;
                    float* __restrict__ c1_row = c + (i+1) * N;
                    float* __restrict__ c2_row = c + (i+2) * N;
                    float* __restrict__ c3_row = c + (i+3) * N;

                    Index j = jj;

                    // Main cycle: 16 floats = 4 vectors × 4 floats
                    //  4(rows) × 4(vectors) = 16 FMA per iteration
                    for (; j + 15 < j_end; j += 16) {
                        // float* addr - address to load to cache
                        // rw: 0 - read, 1 - write
                        // locality: 0-3. 1 optimal by trials and fails
                        __builtin_prefetch(b_row  + j + 64, 0, 1);
                        __builtin_prefetch(c0_row + j + 64, 1, 1);
                        __builtin_prefetch(c1_row + j + 64, 1, 1);

                        // Load 16 floats from b_row (4 vectors × 4 floats)
                        const float32x4_t b0 = vld1q_f32(b_row + j);
                        const float32x4_t b1 = vld1q_f32(b_row + j + 4);
                        const float32x4_t b2 = vld1q_f32(b_row + j + 8);
                        const float32x4_t b3 = vld1q_f32(b_row + j + 12);

                        // 1st c_row, 4 floats, FMA
                        // NOTE: vfmaq_f32(c, a, b) == c + a*b  (accumulator is FIRST, unlike _mm256_fmadd_ps)
                        vst1q_f32(c0_row+j,    vfmaq_f32(vld1q_f32(c0_row+j),    va0, b0));
                        vst1q_f32(c0_row+j+4,  vfmaq_f32(vld1q_f32(c0_row+j+4),  va0, b1));
                        vst1q_f32(c0_row+j+8,  vfmaq_f32(vld1q_f32(c0_row+j+8),  va0, b2));
                        vst1q_f32(c0_row+j+12, vfmaq_f32(vld1q_f32(c0_row+j+12), va0, b3));

                        // 2nd c_row, 4 floats, FMA
                        vst1q_f32(c1_row+j,    vfmaq_f32(vld1q_f32(c1_row+j),    va1, b0));
                        vst1q_f32(c1_row+j+4,  vfmaq_f32(vld1q_f32(c1_row+j+4),  va1, b1));
                        vst1q_f32(c1_row+j+8,  vfmaq_f32(vld1q_f32(c1_row+j+8),  va1, b2));
                        vst1q_f32(c1_row+j+12, vfmaq_f32(vld1q_f32(c1_row+j+12), va1, b3));

                        // 3rd row
                        vst1q_f32(c2_row+j,    vfmaq_f32(vld1q_f32(c2_row+j),    va2, b0));
                        vst1q_f32(c2_row+j+4,  vfmaq_f32(vld1q_f32(c2_row+j+4),  va2, b1));
                        vst1q_f32(c2_row+j+8,  vfmaq_f32(vld1q_f32(c2_row+j+8),  va2, b2));
                        vst1q_f32(c2_row+j+12, vfmaq_f32(vld1q_f32(c2_row+j+12), va2, b3));

                        // 4th row
                        vst1q_f32(c3_row+j,    vfmaq_f32(vld1q_f32(c3_row+j),    va3, b0));
                        vst1q_f32(c3_row+j+4,  vfmaq_f32(vld1q_f32(c3_row+j+4),  va3, b1));
                        vst1q_f32(c3_row+j+8,  vfmaq_f32(vld1q_f32(c3_row+j+8),  va3, b2));
                        vst1q_f32(c3_row+j+12, vfmaq_f32(vld1q_f32(c3_row+j+12), va3, b3));
                    }

                    // Remainder of > 4 floats
                    for (; j + 3 < j_end; j += 4) {
                        const float32x4_t vb = vld1q_f32(b_row + j);
                        vst1q_f32(c0_row+j, vfmaq_f32(vld1q_f32(c0_row+j), va0, vb));
                        vst1q_f32(c1_row+j, vfmaq_f32(vld1q_f32(c1_row+j), va1, vb));
                        vst1q_f32(c2_row+j, vfmaq_f32(vld1q_f32(c2_row+j), va2, vb));
                        vst1q_f32(c3_row+j, vfmaq_f32(vld1q_f32(c3_row+j), va3, vb));
                    }

                    // the smallest remainder
                    const float alpha0 = a[(i+0) * K + k];
                    const float alpha1 = a[(i+1) * K + k];
                    const float alpha2 = a[(i+2) * K + k];
                    const float alpha3 = a[(i+3) * K + k];
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
                    const float32x4_t valpha = vdupq_n_f32(a[i * K + k]);
                    const float* b_row       = b + k * N;
                    float*       c_row       = c + i * N;
                    Index j = jj;
                    for (; j + 15 < j_end; j += 16) {
                        __builtin_prefetch(b_row + j + 64, 0, 1);
                        __builtin_prefetch(c_row + j + 64, 1, 1);
                        vst1q_f32(c_row+j,    vfmaq_f32(vld1q_f32(c_row+j),    valpha, vld1q_f32(b_row+j)));
                        vst1q_f32(c_row+j+4,  vfmaq_f32(vld1q_f32(c_row+j+4),  valpha, vld1q_f32(b_row+j+4)));
                        vst1q_f32(c_row+j+8,  vfmaq_f32(vld1q_f32(c_row+j+8),  valpha, vld1q_f32(b_row+j+8)));
                        vst1q_f32(c_row+j+12, vfmaq_f32(vld1q_f32(c_row+j+12), valpha, vld1q_f32(b_row+j+12)));
                    }
                    for (; j + 3 < j_end; j += 4) {
                        vst1q_f32(c_row+j, vfmaq_f32(vld1q_f32(c_row+j), valpha, vld1q_f32(b_row+j)));
                    }
                    const float alpha = a[i * K + k];
                    for (; j < j_end; ++j) c_row[j] += alpha * b_row[j];
                }
            }
        }}}
    } else {
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

#elif defined(__ARM_NEON)
    if constexpr (std::is_same_v<Scalar, double>) {
        Index i = 0;
        float64x2_t vscalar = vdupq_n_f64(scalar);
        for (; i + 1 < size; i += 2) {
            float64x2_t v = vld1q_f64(ptr + i);
            v = vmulq_f64(v, vscalar);
            vst1q_f64(ptr + i, v);
        }
        for (; i < size; ++i) ptr[i] *= scalar;
    } else if constexpr (std::is_same_v<Scalar, float>) {
        Index i = 0;
        float32x4_t vscalar = vdupq_n_f32(scalar);
        for (; i + 3 < size; i += 4) {
            float32x4_t v = vld1q_f32(ptr + i);
            v = vmulq_f32(v, vscalar);
            vst1q_f32(ptr + i, v);
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

#elif defined(__ARM_NEON)
    if constexpr (std::is_same_v<Scalar, double>) {
        // ARM has no vdivq_f64 on most cores — multiply by reciprocal (faster)
        const double inv = 1.0 / scalar;
        Index i = 0;
        float64x2_t vinv = vdupq_n_f64(inv);
        for (; i + 1 < sz; i += 2) {
            float64x2_t v = vld1q_f64(ptr + i);
            v = vmulq_f64(v, vinv);
            vst1q_f64(ptr + i, v);
        }
        for (; i < sz; ++i) ptr[i] *= inv;
    } else if constexpr (std::is_same_v<Scalar, float>) {
        // ARM has no vdivq_f32 on most cores — multiply by reciprocal (faster)
        const float inv = 1.0f / scalar;
        Index i = 0;
        float32x4_t vinv = vdupq_n_f32(inv);
        for (; i + 3 < sz; i += 4) {
            float32x4_t v = vld1q_f32(ptr + i);
            v = vmulq_f32(v, vinv);
            vst1q_f32(ptr + i, v);
        }
        for (; i < sz; ++i) ptr[i] *= inv;
    } else {
        for (Index i = 0; i < sz; ++i) ptr[i] /= scalar;
    }
#else
    for (Index i = 0; i < sz; ++i) ptr[i] /= scalar;
#endif

    return *this;
}

} // namespace pla