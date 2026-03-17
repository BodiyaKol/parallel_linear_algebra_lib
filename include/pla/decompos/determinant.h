#pragma once

#include "pla/core/matrix.h"
#include "pla/decompos/lu.h"

namespace pla {

    template<typename Scalar>
        requires Numeric<Scalar>
    Scalar determinant(const Matrix<Scalar>& A);

}

#include "pla/decompos/determinant_impl.hpp"