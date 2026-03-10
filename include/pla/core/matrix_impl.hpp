#ifndef PARALLEL_LINEAR_ALGEBRA_LIB_MATRIX_OTHER_OPR_HPP
#define PARALLEL_LINEAR_ALGEBRA_LIB_MATRIX_OTHER_OPR_HPP

#include "pla"/core/matrix.h"
#include "pla/types/index.h""

namespace pla {

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
    if (value == Scalar{0}) std::memset(data(), Scalar{0}, size() * sizeof(Scalar));
    else                    std::fill  (data(), data() + size(), value);
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
    Matrix result(n, n, Scalar{0}, order);
    for (Index i = 0; i < n; i++) {
        result(i, i) = 1.0;
    }
    return result;
}


template<typename Scalar>
Matrix<Scalar> operator*(Scalar scalar, const Matrix<Scalar>& mat) {
    return mat * scalar;
}


template<typename Scalar>
    requires Numeric<Scalar>
Scalar Matrix<Scalar>::norm() const {
    return 0.0;
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

}

#endif //PARALLEL_LINEAR_ALGEBRA_LIB_MATRIX_OTHER_OPR_HPP