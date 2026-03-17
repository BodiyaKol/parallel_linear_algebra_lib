#pragma once

#include "pla/core/matrix.h"
#include "pla/algorithms/solve.h"

namespace pla {

template<typename Scalar>
    requires Numeric<Scalar>
Matrix<Scalar> inverse(const Matrix<Scalar>& A);

}

#include "pla/algorithms/inverse_impl.hpp"