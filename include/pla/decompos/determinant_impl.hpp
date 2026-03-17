#pragma once

#include <stdexcept>
#include "pla/decompos/lu.h"

namespace pla {

    template<typename Scalar>
        requires Numeric<Scalar>
    Scalar determinant(const Matrix<Scalar>& A) {
        if (!A.is_square())
            throw std::invalid_argument("determinant: matrix must be square");

        // det(A) = product of U diagonal
        // since no pivoting: sign is always +1
        LUResult<Scalar> res = lu_blocked(A);

        Scalar det = Scalar{1};
        for (Index i = 0; i < A.rows(); ++i)
            det *= res.U(i, i);

        return det;
    }

}