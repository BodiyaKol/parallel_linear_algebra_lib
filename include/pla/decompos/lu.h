#pragma once

#include "pla/core/matrix.h"
#include "pla/types/index.h"
#include <vector>

namespace pla {

template<typename Scalar>
	requires Numeric<Scalar>
struct LUResult {
    Matrix<Scalar> L;
    Matrix<Scalar> U;
	Matrix<Scalar> LU_packed;
    std::vector<Index> perm;
};

template<typename Scalar>
	requires Numeric<Scalar>
LUResult<Scalar> lu_blocked(const Matrix<Scalar>& input, Index block_size = 32);

template<typename Scalar>
	requires Numeric<Scalar>
LUResult<Scalar> lu_naive(const Matrix<Scalar>& input);

}

#include "pla/decompos/lu_impl.hpp"
