#pragma once

#include "pla/core/matrix.h"
#include "pla/core/vector.h"
#include "pla/decompos/lu.h"
#include "pla/types/index.h"

namespace pla {
    // using LU
    template<typename Scalar>
        requires Numeric<Scalar>
    Vector<Scalar> solve(const Matrix<Scalar>& A, const Vector<Scalar>& b);

    template<typename Scalar>
        requires Numeric<Scalar>
    Vector<Scalar> solve(const LUResult<Scalar>& lu, const Vector<Scalar>& b);

    // AX = B(b1, b2, ...)
    template<typename Scalar>
        requires Numeric<Scalar>
    Matrix<Scalar> solve(const Matrix<Scalar>& A, const Matrix<Scalar>& B);

}

#include "pla/algorithms/solve_impl.hpp"