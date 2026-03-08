#pragma once

#include "../../types/matrix.h"
#include <vector>

namespace pla {

template<typename Scalar>
struct LUResult {
    Matrix<Scalar> L;
    Matrix<Scalar> U;
    std::vector<Index> perm;
};

template<typename Scalar>
LUResult<Scalar> lu_blocked(const Matrix<Scalar>& input, Index block_size = 32);

template<typename Scalar>
LUResult<Scalar> lu_naive(const Matrix<Scalar>& input);

}