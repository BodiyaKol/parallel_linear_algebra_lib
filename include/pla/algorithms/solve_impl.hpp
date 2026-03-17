#pragma once

#include <stdexcept>
#include "pla/decompos/lu.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace pla {

// Ly = b
template<typename Scalar>
    requires Numeric<Scalar>
static Vector<Scalar> forward_substitution(const Matrix<Scalar>& L,
                                            const Vector<Scalar>& b) {
    Index n = L.rows();
    Vector<Scalar> y(n);

    for (Index i = 0; i < n; ++i) {
        Scalar sum = Scalar{0};

#ifdef __AVX2__
        if constexpr (std::is_same_v<Scalar, double>) {
            __m256d vsum = _mm256_setzero_pd();
            Index j = 0;
            for (; j + 3 < i; j += 4) {
                __m256d vl = _mm256_loadu_pd(&L(i, j));
                __m256d vy = _mm256_loadu_pd(&y[j]);
                vsum = _mm256_fmadd_pd(vl, vy, vsum);
            }
            __m128d lo = _mm256_castpd256_pd128(vsum);
            __m128d hi = _mm256_extractf128_pd(vsum, 1);
            __m128d s  = _mm_add_pd(lo, hi);
            s = _mm_hadd_pd(s, s);
            sum = _mm_cvtsd_f64(s);
            for (; j < i; ++j)
                sum += L(i, j) * y[j];
        } else {
            for (Index j = 0; j < i; ++j)
                sum += L(i, j) * y[j];
        }
#elif defined(__ARM_NEON)
        if constexpr (std::is_same_v<Scalar, double>) {
            float64x2_t vsum = vdupq_n_f64(0.0);
            Index j = 0;
            for (; j + 1 < i; j += 2) {
                float64x2_t vl = vld1q_f64(&L(i, j));
                float64x2_t vy = vld1q_f64(&y[j]);
                vsum = vfmaq_f64(vsum, vl, vy);
            }
            sum = vgetq_lane_f64(vsum, 0) + vgetq_lane_f64(vsum, 1);
            for (; j < i; ++j)
                sum += L(i, j) * y[j];
        } else {
            for (Index j = 0; j < i; ++j)
                sum += L(i, j) * y[j];
        }
#else
        for (Index j = 0; j < i; ++j)
            sum += L(i, j) * y[j];
#endif

        y[i] = b[i] - sum;
    }

    return y;
}


// Ux = y
template<typename Scalar>
    requires Numeric<Scalar>
static Vector<Scalar> backward_substitution(const Matrix<Scalar>& U,
                                             const Vector<Scalar>& y) {
    Index n = U.rows();
    Vector<Scalar> x(n);

    for (Index i = n - 1; i < n; --i) {
        Scalar sum = Scalar{0};

#ifdef __AVX2__
        if constexpr (std::is_same_v<Scalar, double>) {
            __m256d vsum = _mm256_setzero_pd();
            Index j = i + 1;
            for (; j + 3 < n; j += 4) {
                __m256d vu = _mm256_loadu_pd(&U(i, j));
                __m256d vx = _mm256_loadu_pd(&x[j]);
                vsum = _mm256_fmadd_pd(vu, vx, vsum);
            }
            __m128d lo = _mm256_castpd256_pd128(vsum);
            __m128d hi = _mm256_extractf128_pd(vsum, 1);
            __m128d s  = _mm_add_pd(lo, hi);
            s = _mm_hadd_pd(s, s);
            sum = _mm_cvtsd_f64(s);
            for (; j < n; ++j)
                sum += U(i, j) * x[j];
        } else {
            for (Index j = i + 1; j < n; ++j)
                sum += U(i, j) * x[j];
        }
#elif defined(__ARM_NEON)
        if constexpr (std::is_same_v<Scalar, double>) {
            float64x2_t vsum = vdupq_n_f64(0.0);
            Index j = i + 1;
            for (; j + 1 < n; j += 2) {
                float64x2_t vu = vld1q_f64(&U(i, j));
                float64x2_t vx = vld1q_f64(&x[j]);
                vsum = vfmaq_f64(vsum, vu, vx);
            }
            sum = vgetq_lane_f64(vsum, 0) + vgetq_lane_f64(vsum, 1);
            for (; j < n; ++j)
                sum += U(i, j) * x[j];
        } else {
            for (Index j = i + 1; j < n; ++j)
                sum += U(i, j) * x[j];
        }
#else
        for (Index j = i + 1; j < n; ++j)
            sum += U(i, j) * x[j];
#endif

        if (std::abs(U(i, i)) < 1e-12)
            throw std::runtime_error("solve: singular matrix");

        x[i] = (y[i] - sum) / U(i, i);
    }

    return x;
}


// Ly = b (packed LU)
template<typename Scalar>
    requires Numeric<Scalar>
static Vector<Scalar> forward_substitution_packed(const Matrix<Scalar>& LU,
                                                   const Vector<Scalar>& b) {
    Index n = LU.rows();
    Vector<Scalar> y(n);

    for (Index i = 0; i < n; ++i) {
        Scalar sum = Scalar{0};

#ifdef __AVX2__
        if constexpr (std::is_same_v<Scalar, double>) {
            __m256d vsum = _mm256_setzero_pd();
            Index j = 0;
            for (; j + 3 < i; j += 4) {
                __m256d vl = _mm256_loadu_pd(&LU(i, j));
                __m256d vy = _mm256_loadu_pd(&y[j]);
                vsum = _mm256_fmadd_pd(vl, vy, vsum);
            }
            __m128d lo = _mm256_castpd256_pd128(vsum);
            __m128d hi = _mm256_extractf128_pd(vsum, 1);
            __m128d s  = _mm_add_pd(lo, hi);
            s = _mm_hadd_pd(s, s);
            sum = _mm_cvtsd_f64(s);
            for (; j < i; ++j)
                sum += LU(i, j) * y[j];
        } else {
            for (Index j = 0; j < i; ++j)
                sum += LU(i, j) * y[j];
        }
#elif defined(__ARM_NEON)
        if constexpr (std::is_same_v<Scalar, double>) {
            float64x2_t vsum = vdupq_n_f64(0.0);
            Index j = 0;
            for (; j + 1 < i; j += 2) {
                float64x2_t vl = vld1q_f64(&LU(i, j));
                float64x2_t vy = vld1q_f64(&y[j]);
                vsum = vfmaq_f64(vsum, vl, vy);
            }
            sum = vgetq_lane_f64(vsum, 0) + vgetq_lane_f64(vsum, 1);
            for (; j < i; ++j)
                sum += LU(i, j) * y[j];
        } else {
            for (Index j = 0; j < i; ++j)
                sum += LU(i, j) * y[j];
        }
#else
        for (Index j = 0; j < i; ++j)
            sum += LU(i, j) * y[j];
#endif

        y[i] = b[i] - sum;
    }

    return y;
}


// backward substitution: Ux = y (packed LU)
template<typename Scalar>
    requires Numeric<Scalar>
static Vector<Scalar> backward_substitution_packed(const Matrix<Scalar>& LU,
                                                    const Vector<Scalar>& y) {
    Index n = LU.rows();
    Vector<Scalar> x(n);

    for (Index i = n - 1; i < n; --i) {
        Scalar sum = Scalar{0};

#ifdef __AVX2__
        if constexpr (std::is_same_v<Scalar, double>) {
            __m256d vsum = _mm256_setzero_pd();
            Index j = i + 1;
            for (; j + 3 < n; j += 4) {
                __m256d vu = _mm256_loadu_pd(&LU(i, j));
                __m256d vx = _mm256_loadu_pd(&x[j]);
                vsum = _mm256_fmadd_pd(vu, vx, vsum);
            }
            __m128d lo = _mm256_castpd256_pd128(vsum);
            __m128d hi = _mm256_extractf128_pd(vsum, 1);
            __m128d s  = _mm_add_pd(lo, hi);
            s = _mm_hadd_pd(s, s);
            sum = _mm_cvtsd_f64(s);
            for (; j < n; ++j)
                sum += LU(i, j) * x[j];
        } else {
            for (Index j = i + 1; j < n; ++j)
                sum += LU(i, j) * x[j];
        }
#elif defined(__ARM_NEON)
        if constexpr (std::is_same_v<Scalar, double>) {
            float64x2_t vsum = vdupq_n_f64(0.0);
            Index j = i + 1;
            for (; j + 1 < n; j += 2) {
                float64x2_t vu = vld1q_f64(&LU(i, j));
                float64x2_t vx = vld1q_f64(&x[j]);
                vsum = vfmaq_f64(vsum, vu, vx);
            }
            sum = vgetq_lane_f64(vsum, 0) + vgetq_lane_f64(vsum, 1);
            for (; j < n; ++j)
                sum += LU(i, j) * x[j];
        } else {
            for (Index j = i + 1; j < n; ++j)
                sum += LU(i, j) * x[j];
        }
#else
        for (Index j = i + 1; j < n; ++j)
            sum += LU(i, j) * x[j];
#endif

        if (std::abs(LU(i, i)) < 1e-12)
            throw std::runtime_error("solve: singular matrix");

        x[i] = (y[i] - sum) / LU(i, i);
    }

    return x;
}



template<typename Scalar>
    requires Numeric<Scalar>
Vector<Scalar> solve(const LUResult<Scalar>& lu, const Vector<Scalar>& b) {
    Index n = static_cast<Index>(b.dimension());

    if (lu.LU_packed.rows() > 0) {
        if (lu.LU_packed.rows() != n)
            throw std::invalid_argument("solve: size mismatch");
        Vector<Scalar> y = forward_substitution_packed(lu.LU_packed, b);
        Vector<Scalar> x = backward_substitution_packed(lu.LU_packed, y);
        return x;
    }

    if (lu.L.rows() != n)
        throw std::invalid_argument("solve: size mismatch");
    Vector<Scalar> y = forward_substitution(lu.L, b);
    Vector<Scalar> x = backward_substitution(lu.U, y);
    return x;
}


// solve Ax = b
template<typename Scalar>
    requires Numeric<Scalar>
Vector<Scalar> solve(const Matrix<Scalar>& A, const Vector<Scalar>& b) {
    if (!A.is_square())
        throw std::invalid_argument("solve: matrix must be square");
    if (A.rows() != static_cast<Index>(b.dimension()))
        throw std::invalid_argument("solve: size mismatch");

    LUResult<Scalar> lu = lu_blocked(A);
    return solve(lu, b);
}


// solve AX = B
template<typename Scalar>
    requires Numeric<Scalar>
Matrix<Scalar> solve(const Matrix<Scalar>& A, const Matrix<Scalar>& B) {
    if (!A.is_square())
        throw std::invalid_argument("solve: matrix must be square");
    if (A.rows() != B.rows())
        throw std::invalid_argument("solve: size mismatch");

    Index n    = A.rows();
    Index nrhs = B.cols();

    LUResult<Scalar> lu = lu_blocked(A);

    Matrix<Scalar> X(n, nrhs, Scalar{0});

#ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (Index col = 0; col < nrhs; ++col) {
        Vector<Scalar> b_col(n);
        for (Index i = 0; i < n; ++i)
            b_col[i] = B(i, col);

        Vector<Scalar> x_col = solve(lu, b_col);

        for (Index i = 0; i < n; ++i)
            X(i, col) = x_col[i];
    }

    return X;
}

}