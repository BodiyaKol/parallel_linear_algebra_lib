#ifndef PARALLEL_LINEAR_ALGEBRA_LIB_MATRIX_ADD_H
#define PARALLEL_LINEAR_ALGEBRA_LIB_MATRIX_ADD_H

#include "pla/core/matrix.h"

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


}

#endif //PARALLEL_LINEAR_ALGEBRA_LIB_MATRIX_ADD_H