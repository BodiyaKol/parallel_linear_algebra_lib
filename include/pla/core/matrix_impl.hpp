#ifndef PARALLEL_LINEAR_ALGEBRA_LIB_MATRIX_OTHER_OPR_HPP
#define PARALLEL_LINEAR_ALGEBRA_LIB_MATRIX_OTHER_OPR_HPP

#include <iomanip>
#include "pla/types/index.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace pla {

template<typename Scalar>
    requires Numeric<Scalar>
Matrix<Scalar> Matrix<Scalar>::transpose() const {
    Matrix<Scalar> result(cols_, rows_, Scalar{0}, order_);

    constexpr Index BLOCK = 64 / sizeof(Scalar);

    for (Index i = 0; i < rows_; i += BLOCK)
        for (Index j = 0; j < cols_; j += BLOCK) {
            const Index i_end = std::min(i + BLOCK, rows_);
            const Index j_end = std::min(j + BLOCK, cols_);
            for (Index ii = i; ii < i_end; ++ii)
                for (Index jj = j; jj < j_end; ++jj)
                    result(jj, ii) = (*this)(ii, jj);
        }

    return result;
}


template<typename Scalar>
    requires Numeric<Scalar>
void Matrix<Scalar>::transpose_inplace() {
if (!is_square())
    throw NonSquareMatrixException(rows(), cols());

constexpr Index BLOCK = 64 / sizeof(Scalar);

for (Index i = 0; i < rows_; i += BLOCK)
    for (Index j = i; j < cols_; j += BLOCK) {
        const Index i_end = std::min(i + BLOCK, rows_);
        const Index j_end = std::min(j + BLOCK, cols_);
        for (Index ii = i; ii < i_end; ++ii) {
            const Index j_start = (i == j) ? ii + 1 : j;
            for (Index jj = j_start; jj < j_end; ++jj)
                std::swap((*this)(ii, jj), (*this)(jj, ii));
        }
    }
}


template<typename Scalar>
    requires Numeric<Scalar>
Vector<Scalar> Matrix<Scalar>::row(Index r) const {
    if (r >= rows())
        throw IndexOutOfRangeException(r, rows());

    Vector<Scalar> v(cols_);
    if (order_ == StorageOrder::RowMajor) {
        std::memcpy(v.data(), data() + r * cols_, cols_ * sizeof(Scalar));
    } else {
        for (Index j = 0; j < cols_; ++j)
            v[j] = (*this)(r, j);
    }
    return v;
}


template<typename Scalar>
    requires Numeric<Scalar>
Vector<Scalar> Matrix<Scalar>::col(Index c) const {
    if (c >= cols())
        throw IndexOutOfRangeException(c, cols());

    Vector<Scalar> v(rows_);
    if (order_ == StorageOrder::ColMajor) {
        std::memcpy(v.data(), data() + c * rows_, rows_ * sizeof(Scalar));
    } else {
        for (Index i = 0; i < rows_; ++i)
            v[i] = (*this)(i, c);
    }
    return v;
}

template<typename Scalar>
    requires Numeric<Scalar>
void Matrix<Scalar>::fill(Scalar value) {
    if (value == Scalar{0}) std::memset(data(), Scalar{0}, size() * sizeof(Scalar));
    else                    std::fill  (data(), data() + size(), value);
}


template<typename Scalar>
    requires Numeric<Scalar>
void Matrix<Scalar>::set_identity() {
    if (!is_square())
        throw NonSquareMatrixException(rows(), cols());

    const Index sz = size();
    std::memset(elements_.get(), 0, sz * sizeof(Scalar));

    const Index stride = (order_ == StorageOrder::RowMajor) ? cols_ + 1 : rows_ + 1;
    Scalar* p = elements_.get();
    for (Index i = 0; i < rows_; ++i)
        p[i * stride] = Scalar{1};
}


template<typename Scalar>
    requires Numeric<Scalar>
Matrix<Scalar> Matrix<Scalar>::identity(Index n, StorageOrder order) {
    Matrix<Scalar> result(n, n, Scalar{0}, order);
    const Index stride = (order == StorageOrder::RowMajor) ? n + 1 : n + 1;
    Scalar* p = result.data();
    for (Index i = 0; i < n; ++i)
        p[i * stride] = Scalar{1};
    return result;
}


template<typename Scalar>
Matrix<Scalar> operator*(Scalar scalar, const Matrix<Scalar>& mat) {
    return mat * scalar;
}



template<typename Scalar>
    requires Numeric<Scalar>
Scalar Matrix<Scalar>::norm() const {
    const Scalar* p = data();
    const Index   sz = size();
    Scalar sum{0};

#ifdef __AVX2__
    if constexpr (std::is_same_v<Scalar, double>) {
        __m256d acc = _mm256_setzero_pd();
        Index i = 0;
        for (; i + 3 < sz; i += 4) {
            __m256d v = _mm256_load_pd(p + i);
            acc = _mm256_fmadd_pd(v, v, acc);
        }
        __m128d lo  = _mm256_castpd256_pd128(acc);
        __m128d hi  = _mm256_extractf128_pd(acc, 1);
        __m128d s   = _mm_add_pd(lo, hi);
        s = _mm_hadd_pd(s, s);
        sum = _mm_cvtsd_f64(s);
        for (; i < sz; ++i) sum += p[i] * p[i];
    } else if constexpr (std::is_same_v<Scalar, float>) {
        __m256 acc = _mm256_setzero_ps();
        Index i = 0;
        for (; i + 7 < sz; i += 8) {
            __m256 v = _mm256_load_ps(p + i);
            acc = _mm256_fmadd_ps(v, v, acc);
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s  = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        sum = _mm_cvtss_f32(s);
        for (; i < sz; ++i) sum += p[i] * p[i];
    } else {
        for (Index i = 0; i < sz; ++i) sum += p[i] * p[i];
    }
#else
    for (Index i = 0; i < sz; ++i) sum += p[i] * p[i];
#endif

    return std::sqrt(sum);
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


template<typename Scalar>
    requires Numeric<Scalar>
Matrix<Scalar> operator*(Scalar scalar, const Matrix<Scalar>& mat) {
    return mat * scalar;
}


template<typename Scalar>
    requires Numeric<Scalar>
void swap(Matrix<Scalar>& a, Matrix<Scalar>& b) noexcept {
    a.swap(b);
}


template<typename Scalar>
    requires Numeric<Scalar>
std::ostream& operator<<(std::ostream& os, const Matrix<Scalar>& m) {
    for (Index i = 0; i < m.rows(); ++i) {
        os << "[ ";
        for (Index j = 0; j < m.cols(); ++j) {
            os << m(i, j);
            if (j + 1 < m.cols()) os << ", ";
        }
        os << " ]\n";
    }
    return os;
}

}

#endif //PARALLEL_LINEAR_ALGEBRA_LIB_MATRIX_OTHER_OPR_HPP
