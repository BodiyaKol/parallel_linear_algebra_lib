#pragma once

#include <stdexcept>
#include "pla/algorithms/solve.h"

namespace pla {

template<typename Scalar>
    requires Numeric<Scalar>
Matrix<Scalar> inverse(const Matrix<Scalar>& A) {
    if (!A.is_square())
        throw std::invalid_argument("inverse: matrix must be square");

    Index n = A.rows();

    Matrix<Scalar> I = Matrix<Scalar>::identity(n);
    return solve(A, I);
}

}