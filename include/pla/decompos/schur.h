#pragma once

#include "pla/core/matrix.h"
#include "pla/types/index.h"
#include "pla/decompos/hessenberg.h"

namespace pla {

    template<typename Scalar>
        requires Numeric<Scalar>
    struct RealSchurOptions {
        Scalar tolerance = static_cast<Scalar>(1e-10);
        Index max_iterations = 1000;
        bool accumulate_u = false;
    };

    template<typename Scalar>
        requires Numeric<Scalar>
    struct RealSchurResult {
        Matrix<Scalar> T;
        Matrix<Scalar> U;
        Index iterations = 0;
        bool converged = false;
        bool has_u = false;
    };

    template<typename Scalar>
        requires Numeric<Scalar>
    RealSchurResult<Scalar> real_schur(
        const Matrix<Scalar>& input,
        const RealSchurOptions<Scalar>& options = {}
    );

} // namespace pla

#include "pla/decompos/schur_impl.hpp"