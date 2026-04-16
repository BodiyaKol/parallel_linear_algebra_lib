#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "pla/exceptions.h"

namespace pla {

namespace detail {

template<typename Scalar>
    requires Numeric<Scalar>
inline Scalar eigen_abs(Scalar x) {
    using std::abs;
    return abs(x);
}

template<typename Scalar>
    requires Numeric<Scalar>
inline std::vector<std::complex<Scalar>> extract_eigenvalues_from_real_schur(
    const Matrix<Scalar>& T,
    Scalar tolerance
) {
    if (!T.is_square()) {
        throw NonSquareMatrixException(T.rows(), T.cols());
    }

    const Index n = T.rows();
    std::vector<std::complex<Scalar>> values;
    values.reserve(static_cast<std::size_t>(n));

    Index i = 0;

    while (i < n) {
        if (i + 1 < n && eigen_abs(T(i + 1, i)) > tolerance) {
            const Scalar a = T(i, i);
            const Scalar b = T(i, i + 1);
            const Scalar c = T(i + 1, i);
            const Scalar d = T(i + 1, i + 1);

            const Scalar trace = a + d;
            const Scalar determinant = a * d - b * c;

            const std::complex<Scalar> discriminant =
                std::complex<Scalar>(trace * trace - static_cast<Scalar>(4) * determinant, Scalar{});

            const std::complex<Scalar> root = std::sqrt(discriminant);

            values.push_back((std::complex<Scalar>(trace, Scalar{}) + root) / static_cast<Scalar>(2));
            values.push_back((std::complex<Scalar>(trace, Scalar{}) - root) / static_cast<Scalar>(2));

            i += 2;
        } else {
            values.push_back(std::complex<Scalar>(T(i, i), Scalar{}));
            ++i;
        }
    }

    return values;
}

template<typename Scalar>
    requires Numeric<Scalar>
inline void swap_rows(Matrix<Scalar>& A, Index r1, Index r2) {
    if (r1 == r2) {
        return;
    }

    for (Index c = 0; c < A.cols(); ++c) {
        std::swap(A(r1, c), A(r2, c));
    }
}

template<typename Scalar>
    requires Numeric<Scalar>
inline Vector<Scalar> null_vector_rref(
    Matrix<Scalar> B,
    Scalar tolerance
) {
    const Index n = B.rows();

    std::vector<Index> pivot_columns;
    pivot_columns.reserve(static_cast<std::size_t>(n));

    Index row = 0;

    for (Index col = 0; col < n && row < n; ++col) {
        Index pivot = row;
        Scalar max_value = eigen_abs(B(row, col));

        for (Index r = row + 1; r < n; ++r) {
            const Scalar value = eigen_abs(B(r, col));
            if (value > max_value) {
                max_value = value;
                pivot = r;
            }
        }

        if (max_value <= tolerance) {
            continue;
        }

        swap_rows(B, row, pivot);

        const Scalar pivot_value = B(row, col);

        for (Index c = col; c < n; ++c) {
            B(row, c) /= pivot_value;
        }

        for (Index r = 0; r < n; ++r) {
            if (r == row) {
                continue;
            }

            const Scalar factor = B(r, col);

            if (eigen_abs(factor) <= tolerance) {
                continue;
            }

            for (Index c = col; c < n; ++c) {
                B(r, c) -= factor * B(row, c);
            }
        }

        pivot_columns.push_back(col);
        ++row;
    }

    std::vector<bool> is_pivot(static_cast<std::size_t>(n), false);

    for (Index c : pivot_columns) {
        is_pivot[static_cast<std::size_t>(c)] = true;
    }

    Index free_col = n;

    for (Index c = 0; c < n; ++c) {
        if (!is_pivot[static_cast<std::size_t>(c)]) {
            free_col = c;
            break;
        }
    }

    Vector<Scalar> x(n, Scalar{});

    if (free_col == n) {
        return x;
    }

    x[free_col] = static_cast<Scalar>(1);

    for (Index r = 0; r < pivot_columns.size(); ++r) {
        const Index pc = pivot_columns[static_cast<std::size_t>(r)];
        x[pc] = -B(r, free_col);
    }

    const Scalar norm = x.norm();

    if (norm > tolerance) {
        x /= norm;
    }

    return x;
}

template<typename Scalar>
    requires Numeric<Scalar>
inline Vector<Scalar> real_eigenvector_from_value(
    const Matrix<Scalar>& A,
    Scalar lambda,
    Scalar tolerance
) {
    const Index n = A.rows();

    Matrix<Scalar> B = A;

    for (Index i = 0; i < n; ++i) {
        B(i, i) -= lambda;
    }

    return null_vector_rref(B, tolerance);
}

} // namespace detail

template<typename Scalar>
    requires Numeric<Scalar>
EigenvaluesResult<Scalar> eigenvalues_general(
    const Matrix<Scalar>& input,
    const EigenOptions<Scalar>& options
) {
    if (!input.is_square()) {
        throw NonSquareMatrixException(input.rows(), input.cols());
    }

    RealSchurOptions<Scalar> schur_options;
    schur_options.tolerance = options.tolerance;
    schur_options.max_iterations = options.max_iterations;
    schur_options.accumulate_u = false;

    RealSchurResult<Scalar> schur = real_schur(input, schur_options);

    EigenvaluesResult<Scalar> result;
    result.schur_form = schur.T;
    result.iterations = schur.iterations;
    result.converged = schur.converged;
    result.values = detail::extract_eigenvalues_from_real_schur(
        schur.T,
        options.tolerance
    );

    return result;
}

template<typename Scalar>
    requires Numeric<Scalar>
EigenResult<Scalar> eigen_general(
    const Matrix<Scalar>& input,
    const EigenOptions<Scalar>& options
) {
    if (!input.is_square()) {
        throw NonSquareMatrixException(input.rows(), input.cols());
    }

    EigenvaluesResult<Scalar> values_result =
        eigenvalues_general(input, options);

    const Index n = input.rows();

    EigenResult<Scalar> result;
    result.values = values_result.values;
    result.schur_form = values_result.schur_form;
    result.iterations = values_result.iterations;
    result.converged = values_result.converged;

    result.real_eigenvectors = Matrix<Scalar>(
        n,
        static_cast<Index>(result.values.size()),
        input.order()
    );

    result.has_real_eigenvector.assign(result.values.size(), false);

    for (Index col = 0; col < result.values.size(); ++col) {
        const auto lambda = result.values[static_cast<std::size_t>(col)];

        if (std::abs(lambda.imag()) > options.tolerance) {
            continue;
        }

        Vector<Scalar> v = detail::real_eigenvector_from_value(
            input,
            lambda.real(),
            options.tolerance
        );

        if (v.norm() <= options.tolerance) {
            continue;
        }

        for (Index i = 0; i < n; ++i) {
            result.real_eigenvectors(i, col) = v[i];
        }

        result.has_real_eigenvector[static_cast<std::size_t>(col)] = true;
    }

    return result;
}

} // namespace pla