#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "pla/exceptions.h"

namespace pla {

namespace detail {

template<typename Scalar>
    requires Numeric<Scalar>
inline Scalar schur_abs(Scalar x) {
    using std::abs;
    return abs(x);
}

template<typename Scalar>
    requires Numeric<Scalar>
inline void givens_rotation(Scalar a, Scalar b, Scalar& c, Scalar& s) {
    using std::hypot;

    if (b == Scalar{}) {
        c = static_cast<Scalar>(1);
        s = Scalar{};
        return;
    }

    const Scalar r = hypot(a, b);
    c = a / r;
    s = b / r;
}

template<typename Scalar>
    requires Numeric<Scalar>
inline bool negligible_subdiag(
    const Matrix<Scalar>& T,
    Index i,
    Scalar tolerance
) {
    const Scalar scale =
        schur_abs(T(i - 1, i - 1)) +
        schur_abs(T(i, i)) +
        static_cast<Scalar>(1);

    return schur_abs(T(i, i - 1)) <= tolerance * scale;
}

template<typename Scalar>
    requires Numeric<Scalar>
inline bool is_quasi_triangular(
    const Matrix<Scalar>& T,
    Scalar tolerance
) {
    const Index n = T.rows();

    for (Index i = 1; i < n; ++i) {
        if (schur_abs(T(i, i - 1)) <= tolerance) {
            continue;
        }

        if (i + 1 < n && schur_abs(T(i + 1, i)) > tolerance) {
            return false;
        }

        ++i;
    }

    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j + 1 < i; ++j) {
            if (schur_abs(T(i, j)) > tolerance) {
                return false;
            }
        }
    }

    return true;
}

template<typename Scalar>
    requires Numeric<Scalar>
inline Scalar wilkinson_shift(
    const Matrix<Scalar>& T,
    Index lo,
    Index hi
) {
    if (hi <= lo) {
        return T(lo, lo);
    }

    const Scalar a = T(hi - 1, hi - 1);
    const Scalar b = T(hi - 1, hi);
    const Scalar c = T(hi, hi - 1);
    const Scalar d = T(hi, hi);

    const Scalar tr = a + d;
    const Scalar det = a * d - b * c;
    const Scalar disc = tr * tr - static_cast<Scalar>(4) * det;

    if (disc < Scalar{}) {
        return d;
    }

    using std::sqrt;
    const Scalar root = sqrt(disc);

    const Scalar mu1 = (tr + root) / static_cast<Scalar>(2);
    const Scalar mu2 = (tr - root) / static_cast<Scalar>(2);

    return schur_abs(mu1 - d) < schur_abs(mu2 - d) ? mu1 : mu2;
}

template<typename Scalar>
    requires Numeric<Scalar>
inline void clean_hessenberg(
    Matrix<Scalar>& T,
    Index lo,
    Index hi
) {
    for (Index i = lo; i <= hi; ++i) {
        for (Index j = lo; j + 1 < i; ++j) {
            T(i, j) = Scalar{};
        }
    }
}

template<typename Scalar>
    requires Numeric<Scalar>
inline void shifted_hessenberg_qr_window_inplace(
    Matrix<Scalar>& H,
    Index lo,
    Index hi,
    Scalar shift,
    std::vector<Scalar>& cs,
    std::vector<Scalar>& sn
) {
    if (hi <= lo) {
        return;
    }

    const Index m = hi - lo + 1;
    cs.assign(static_cast<std::size_t>(m - 1), Scalar{});
    sn.assign(static_cast<std::size_t>(m - 1), Scalar{});

    for (Index i = lo; i <= hi; ++i) {
        H(i, i) -= shift;
    }

    for (Index k = lo; k < hi; ++k) {
        Scalar c{};
        Scalar s{};

        givens_rotation(H(k, k), H(k + 1, k), c, s);

        cs[static_cast<std::size_t>(k - lo)] = c;
        sn[static_cast<std::size_t>(k - lo)] = s;

        for (Index j = k; j <= hi; ++j) {
            const Scalar x = H(k, j);
            const Scalar y = H(k + 1, j);

            H(k, j)     = c * x + s * y;
            H(k + 1, j) = -s * x + c * y;
        }

        H(k + 1, k) = Scalar{};
    }

    for (Index k = lo; k < hi; ++k) {
        const Scalar c = cs[static_cast<std::size_t>(k - lo)];
        const Scalar s = sn[static_cast<std::size_t>(k - lo)];

        const Index row_end = std::min(hi, k + 2);

        for (Index i = lo; i <= row_end; ++i) {
            const Scalar x = H(i, k);
            const Scalar y = H(i, k + 1);

            H(i, k)     = c * x + s * y;
            H(i, k + 1) = -s * x + c * y;
        }
    }

    for (Index i = lo; i <= hi; ++i) {
        H(i, i) += shift;
    }

    clean_hessenberg(H, lo, hi);
}

template<typename Scalar>
    requires Numeric<Scalar>
inline Index find_active_hi(
    Matrix<Scalar>& T,
    Index hi,
    Scalar tolerance
) {
    while (hi > 0) {
        if (negligible_subdiag(T, hi, tolerance)) {
            T(hi, hi - 1) = Scalar{};
            --hi;
            continue;
        }

        break;
    }

    return hi;
}

template<typename Scalar>
    requires Numeric<Scalar>
inline Index find_active_lo(
    const Matrix<Scalar>& T,
    Index hi,
    Scalar tolerance
) {
    Index lo = 0;

    for (Index i = hi; i > 0; --i) {
        if (negligible_subdiag(T, i, tolerance)) {
            lo = i;
            break;
        }
    }

    return lo;
}

} // namespace detail

template<typename Scalar>
    requires Numeric<Scalar>
RealSchurResult<Scalar> real_schur(
    const Matrix<Scalar>& input,
    const RealSchurOptions<Scalar>& options
) {
    if (!input.is_square()) {
        throw NonSquareMatrixException(input.rows(), input.cols());
    }

    const Index n = input.rows();

    HessenbergOptions<Scalar> hess_options;
    hess_options.accumulate_q = false;
    hess_options.tolerance = options.tolerance;

    auto hess = hessenberg_reduce(input, hess_options);

    RealSchurResult<Scalar> result;
    result.T = hess.H;
    result.has_u = false;

    if (n <= 1 || detail::is_quasi_triangular(result.T, options.tolerance)) {
        result.converged = true;
        result.iterations = 0;
        return result;
    }

    std::vector<Scalar> cs;
    std::vector<Scalar> sn;
    cs.reserve(static_cast<std::size_t>(n));
    sn.reserve(static_cast<std::size_t>(n));

    Index hi = n - 1;
    Index iterations = 0;

    while (hi > 0 && iterations < options.max_iterations) {
        hi = detail::find_active_hi(result.T, hi, options.tolerance);

        if (hi == 0) {
            break;
        }

        if (hi == 1) {
            break;
        }

        const Index lo = detail::find_active_lo(result.T, hi, options.tolerance);

        if (hi <= lo) {
            if (hi == 0) {
                break;
            }
            --hi;
            continue;
        }

        if (hi == lo + 1) {
            // Keep 2x2 block as a real Schur block.
            if (lo > 0 && detail::negligible_subdiag(result.T, lo, options.tolerance)) {
                result.T(lo, lo - 1) = Scalar{};
            }
            hi = lo > 0 ? lo - 1 : 0;
            continue;
        }

        Scalar shift = detail::wilkinson_shift(result.T, lo, hi);

        // Exceptional shift to avoid long stagnation.
        if (iterations > 0 && iterations % 30 == 0) {
            shift += static_cast<Scalar>(0.75) *
                     detail::schur_abs(result.T(hi, hi - 1));
        }

        detail::shifted_hessenberg_qr_window_inplace(
            result.T,
            lo,
            hi,
            shift,
            cs,
            sn
        );

        for (Index i = lo + 1; i <= hi; ++i) {
            if (detail::negligible_subdiag(result.T, i, options.tolerance)) {
                result.T(i, i - 1) = Scalar{};
            }
        }

        ++iterations;
    }

    result.iterations = iterations;
    result.converged =
        iterations < options.max_iterations &&
        detail::is_quasi_triangular(result.T, options.tolerance);

    return result;
}

} // namespace pla
