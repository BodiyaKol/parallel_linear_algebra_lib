#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "pla/exceptions.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace pla {

namespace detail {

template<typename Scalar>
    requires Numeric<Scalar>
inline Scalar hess_abs(Scalar x) {
    using std::abs;
    return abs(x);
}

template<typename Scalar>
    requires Numeric<Scalar>
inline Scalar hess_sign(Scalar x) {
    return x < Scalar{} ? static_cast<Scalar>(-1) : static_cast<Scalar>(1);
}

template<typename Scalar>
    requires Numeric<Scalar>
inline bool build_householder_vector(
    const Matrix<Scalar>& H,
    Index start_row,
    Index col,
    Vector<Scalar>& v,
    Scalar tolerance
) {
    const Index n = H.rows();
    const Index m = n - start_row;

    v.resize(m);

    Scalar norm_x = Scalar{};

    for (Index i = 0; i < m; ++i) {
        const Scalar value = H(start_row + i, col);
        v[i] = value;
        norm_x += value * value;
    }

    using std::sqrt;
    norm_x = sqrt(norm_x);

    if (norm_x <= tolerance) {
        return false;
    }

    const Scalar alpha = -hess_sign(v[0]) * norm_x;
    v[0] -= alpha;

    Scalar norm_v = Scalar{};

    for (Index i = 0; i < m; ++i) {
        norm_v += v[i] * v[i];
    }

    norm_v = sqrt(norm_v);

    if (norm_v <= tolerance) {
        return false;
    }

    const Scalar inv_norm = static_cast<Scalar>(1) / norm_v;

    for (Index i = 0; i < m; ++i) {
        v[i] *= inv_norm;
    }

    return true;
}

template<typename Scalar>
    requires Numeric<Scalar>
inline void apply_householder_left(
    Matrix<Scalar>& H,
    const Vector<Scalar>& v,
    Index row0,
    Index col0,
    Vector<Scalar>& work
) {
    const Index m = v.dimension();
    const Index cols = H.cols() - col0;

    work.resize(cols);

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) if(cols > 256 && m > 128)
#endif
    for (std::int64_t jj = 0; jj < static_cast<std::int64_t>(cols); ++jj) {
        const Index j = static_cast<Index>(jj);

        Scalar dot = Scalar{};
        for (Index i = 0; i < m; ++i) {
            dot += v[i] * H(row0 + i, col0 + j);
        }

        work[j] = dot;
    }

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) if(m * cols > 65536)
#endif
    for (std::int64_t ii = 0; ii < static_cast<std::int64_t>(m); ++ii) {
        const Index i = static_cast<Index>(ii);
        const Scalar factor = static_cast<Scalar>(2) * v[i];

        if (factor == Scalar{}) {
            continue;
        }

        for (Index j = 0; j < cols; ++j) {
            H(row0 + i, col0 + j) -= factor * work[j];
        }
    }
}

template<typename Scalar>
    requires Numeric<Scalar>
inline void apply_householder_right(
    Matrix<Scalar>& H,
    const Vector<Scalar>& v,
    Index row0,
    Index col0,
    Vector<Scalar>& work
) {
    const Index rows = H.rows() - row0;
    const Index m = v.dimension();

    work.resize(rows);

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) if(rows > 256 && m > 128)
#endif
    for (std::int64_t ii = 0; ii < static_cast<std::int64_t>(rows); ++ii) {
        const Index i = static_cast<Index>(ii);

        Scalar dot = Scalar{};
        for (Index j = 0; j < m; ++j) {
            dot += H(row0 + i, col0 + j) * v[j];
        }

        work[i] = dot;
    }

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) if(rows * m > 65536)
#endif
    for (std::int64_t ii = 0; ii < static_cast<std::int64_t>(rows); ++ii) {
        const Index i = static_cast<Index>(ii);
        const Scalar factor = static_cast<Scalar>(2) * work[i];

        if (factor == Scalar{}) {
            continue;
        }

        for (Index j = 0; j < m; ++j) {
            H(row0 + i, col0 + j) -= factor * v[j];
        }
    }
}

} // namespace detail

template<typename Scalar>
    requires Numeric<Scalar>
HessenbergResult<Scalar> hessenberg_reduce(
    const Matrix<Scalar>& input,
    const HessenbergOptions<Scalar>& options,
    HessenbergWorkspace<Scalar>* workspace
) {
    if (!input.is_square()) {
        throw NonSquareMatrixException(input.rows(), input.cols());
    }

    const Index n = input.rows();

    HessenbergResult<Scalar> result;
    result.H = input;
    result.has_q = options.accumulate_q;

    if (options.accumulate_q) {
        result.Q = Matrix<Scalar>::identity(n, input.order());
    }

    if (n <= 2) {
        return result;
    }

    HessenbergWorkspace<Scalar> local_workspace;
    HessenbergWorkspace<Scalar>& ws =
        workspace != nullptr ? *workspace : local_workspace;

    ws.v.resize(n);
    ws.work.resize(n);

    Matrix<Scalar>& H = result.H;

    for (Index k = 0; k + 2 < n; ++k) {
        const Index start_row = k + 1;

        const bool built = detail::build_householder_vector(
            H,
            start_row,
            k,
            ws.v,
            options.tolerance
        );

        if (!built) {
            continue;
        }

        detail::apply_householder_left(
            H,
            ws.v,
            start_row,
            k,
            ws.work
        );

        detail::apply_householder_right(
            H,
            ws.v,
            0,
            start_row,
            ws.work
        );

        if (options.accumulate_q) {
            detail::apply_householder_right(
                result.Q,
                ws.v,
                0,
                start_row,
                ws.work
            );
        }

        for (Index i = k + 2; i < n; ++i) {
            H(i, k) = Scalar{};
        }
    }

    return result;
}

template<typename Scalar>
    requires Numeric<Scalar>
bool is_hessenberg(const Matrix<Scalar>& H, Scalar tolerance) {
    if (!H.is_square()) {
        return false;
    }

    const Index n = H.rows();

    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j + 1 < i; ++j) {
            if (detail::hess_abs(H(i, j)) > tolerance) {
                return false;
            }
        }
    }

    return true;
}

} // namespace pla
