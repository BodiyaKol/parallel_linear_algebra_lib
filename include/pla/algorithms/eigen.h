#pragma once

#include <complex>
#include <vector>

#include "pla/core/matrix.h"
#include "pla/core/vector.h"
#include "pla/types/index.h"
#include "pla/decompos/schur.h"

namespace pla {

    template<typename Scalar>
        requires Numeric<Scalar>
    struct EigenOptions {
        Scalar tolerance = static_cast<Scalar>(1e-10);
        Index max_iterations = 1000;
    };

    template<typename Scalar>
        requires Numeric<Scalar>
    struct EigenvaluesResult {
        std::vector<std::complex<Scalar>> values;
        Matrix<Scalar> schur_form;
        Index iterations = 0;
        bool converged = false;
    };

    template<typename Scalar>
        requires Numeric<Scalar>
    struct EigenResult {
        std::vector<std::complex<Scalar>> values;
        Matrix<Scalar> real_eigenvectors;
        std::vector<bool> has_real_eigenvector;
        Matrix<Scalar> schur_form;
        Index iterations = 0;
        bool converged = false;
    };

    template<typename Scalar>
        requires Numeric<Scalar>
    EigenvaluesResult<Scalar> eigenvalues_general(
        const Matrix<Scalar>& input,
        const EigenOptions<Scalar>& options = {}
    );

    template<typename Scalar>
        requires Numeric<Scalar>
    EigenResult<Scalar> eigen_general(
        const Matrix<Scalar>& input,
        const EigenOptions<Scalar>& options = {}
    );

} // namespace pla

#include "pla/algorithms/eigen_impl.hpp"