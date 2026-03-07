#pragma once

#include "../../types/matrix.h"
#include <vector>
#include <stdexcept>
#include <cmath>

namespace pla {

    struct LUResult {
        Matrix L;
        Matrix U;
        std::vector<Index> perm;
    };

    LUResult lu_blocked(const Matrix& input, Index block_size = 32);
    LUResult lu_naive(const Matrix& input);

}