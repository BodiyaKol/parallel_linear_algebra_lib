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

    const double* __restrict__ a = A.elements_.get();
    const double* __restrict__ b = B.elements_.get();
    double* __restrict__ c = C.elements_.get();

    const Index M = A.rows();
    const Index N = B.cols();
    const Index K = A.cols(); // same as B.rows()

#ifdef __AVX2__
    // sizeof(L1 cache) == (32KB)
    constexpr Index TILE_M = 16;
    constexpr Index TILE_K = 32;
    constexpr Index TILE_N = 64;  // mod 4 (AVX2 = 4 doubles)


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

        for (Index i = ii; i < i_end; ++i) {
        for (Index k = kk; k < k_end; ++k) {

            const double alpha = a[i * K + k];
            const __m256d valpha = _mm256_set1_pd(alpha);

            const double* b_row = b + k * N;
            double*       c_row = c + i * N;

            Index j = jj;

            // 16 doubles per iteration - 4 doubles per register
            for (; j + 15 < j_end; j += 16) {
                __builtin_prefetch(b_row + j + 64, 0, 1);
                __builtin_prefetch(c_row + j + 64, 1, 1);

                //                        Packed Doubles
                __m256d c0 = _mm256_load_pd(c_row + j);
                __m256d c1 = _mm256_load_pd(c_row + j + 4);
                __m256d c2 = _mm256_load_pd(c_row + j + 8);
                __m256d c3 = _mm256_load_pd(c_row + j + 12);

                __m256d b0 = _mm256_load_pd(b_row + j);
                __m256d b1 = _mm256_load_pd(b_row + j + 4);
                __m256d b2 = _mm256_load_pd(b_row + j + 8);
                __m256d b3 = _mm256_load_pd(b_row + j + 12);

                c0 = _mm256_fmadd_pd(valpha, b0, c0);
                c1 = _mm256_fmadd_pd(valpha, b1, c1);
                c2 = _mm256_fmadd_pd(valpha, b2, c2);
                c3 = _mm256_fmadd_pd(valpha, b3, c3);

                _mm256_store_pd(c_row + j,      c0);
                _mm256_store_pd(c_row + j + 4,  c1);
                _mm256_store_pd(c_row + j + 8,  c2);
                _mm256_store_pd(c_row + j + 12, c3);
            }

            // 4-double remnant
            for (; j + 3 < j_end; j += 4) {
                __m256d vc = _mm256_load_pd(c_row + j);
                __m256d vb = _mm256_load_pd(b_row + j);
                vc = _mm256_fmadd_pd(valpha, vb, vc);
                _mm256_store_pd(c_row + j, vc);
            }

            // scalar remnant
            for (; j < j_end; ++j) {
                c_row[j] += alpha * b_row[j];
            }
        }}
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

// TODO 8-11: Реалізувати складені оператори +=, -=, *=, /=
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

// Реалізувати транспонування: A^T
Matrix Matrix::transpose() const {
    return Matrix{};
}

// Транспонування in-place (тільки для квадратних)
void Matrix::transpose_inplace() {
    if (!is_square()) {
        throw NonSquareMatrixException(rows(), cols());
    }

}

// Отримати рядок як вектор
Vector Matrix::row(Index r) const {
    if (r >= rows()) {
        throw IndexOutOfRangeException(r, rows());
    }
    return Vector{};
}

// Отримати стовпець як вектор
Vector Matrix::col(Index c) const {
    if (c >= cols()) {
        throw IndexOutOfRangeException(c, cols());
    }
    
    return Vector{};
}

// Заповнити матрицю значенням
void Matrix::fill(Scalar value) {
    if (value == 0.0) std::memset(elements_.get(), 0, rows_ * cols_ * sizeof(double));
    else              std::fill(  elements_.get(), elements_.get() + rows_ * cols_, value);
}

// Зробити матрицю одиничною (identity)
void Matrix::set_identity() {
    if (!is_square()) {
        throw NonSquareMatrixException(rows(), cols());
    }
    
}

// Статичний метод: створити одиничну матрицю
Matrix Matrix::identity(Index n, StorageOrder order) {
    Matrix result(n, n, 0.0, order);
    for (Index i = 0; i < n; i++) {
        result(i, i) = 1.0;
    }
    return result;
}

// Обчислити норму
Scalar Matrix::norm() const {
    return 0.0;
}

// Множення скаляра на матрицю зліва: α * A
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
