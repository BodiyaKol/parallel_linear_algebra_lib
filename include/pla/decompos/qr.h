#pragma once

#include "pla/core/matrix.h"
#include "pla/types/index.h"

namespace pla {

    template<typename Scalar>
        requires Numeric<Scalar>
    struct QRResult {
        Matrix<Scalar> Q;
        Matrix<Scalar> R;
    };

    template<typename Scalar>
        requires Numeric<Scalar>
    QRResult<Scalar> qr_householder(const Matrix<Scalar>& input);

    template<typename Scalar>
        requires Numeric<Scalar>
    QRResult<Scalar> qr_givens(const Matrix<Scalar>& input);

    template<typename Scalar>
        requires Numeric<Scalar>
    QRResult<Scalar> qr(const Matrix<Scalar>& input);

}

#include "pla/decompos/qr_impl.hpp"
