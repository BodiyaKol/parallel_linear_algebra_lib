#pragma once

#include "pla/core/matrix.h"
#include "pla/core/vector.h"
#include "pla/types/index.h"

namespace pla {

    template<typename Scalar>
        requires Numeric<Scalar>
    struct HessenbergOptions {
        bool accumulate_q = false;
        Scalar tolerance = static_cast<Scalar>(1e-12);
    };

    template<typename Scalar>
        requires Numeric<Scalar>
    struct HessenbergResult {
        Matrix<Scalar> H;
        Matrix<Scalar> Q;
        bool has_q = false;
    };

    template<typename Scalar>
        requires Numeric<Scalar>
    struct HessenbergWorkspace {
        Vector<Scalar> v;
        Vector<Scalar> work;
    };

    template<typename Scalar>
        requires Numeric<Scalar>
    HessenbergResult<Scalar> hessenberg_reduce(
        const Matrix<Scalar>& input,
        const HessenbergOptions<Scalar>& options = {},
        HessenbergWorkspace<Scalar>* workspace = nullptr
    );

    template<typename Scalar>
        requires Numeric<Scalar>
    bool is_hessenberg(
        const Matrix<Scalar>& H,
        Scalar tolerance = static_cast<Scalar>(1e-10)
    );

} // namespace pla

#include "pla/decompos/hessenberg_impl.hpp"