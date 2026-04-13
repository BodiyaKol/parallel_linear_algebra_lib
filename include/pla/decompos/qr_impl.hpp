#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "pla/core/matrix.h"
#include "pla/core/vector.h"
#include "pla/types/index.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace pla {

// AlignedAllocator for std::vector to ensure 64-byte alignment for SIMD operations
template<typename T, std::size_t Alignment = 64>
struct AlignedAllocator {
    using value_type = T;

    template<typename U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };

    AlignedAllocator() noexcept = default;

    template<typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        return static_cast<T*>(
            ::operator new(n * sizeof(T), std::align_val_t{Alignment}));
    }

    void deallocate(T* p, std::size_t) noexcept {
        ::operator delete(p, std::align_val_t{Alignment});
    }

    template<typename U, std::size_t A2>
    bool operator==(const AlignedAllocator<U, A2>&) const noexcept { return Alignment == A2; }

    template<typename U, std::size_t A2>
    bool operator!=(const AlignedAllocator<U, A2>& o) const noexcept { return !(*this == o); }
};

template<typename Scalar>
using AlignedVec = std::vector<Scalar, AlignedAllocator<Scalar, 64>>;

static constexpr Index QR_BLOCK = 48;

// QRWorkspace holds temporary buffers for the blocked Householder QR algorithm
template<typename Scalar>
struct QRWorkspace {
    AlignedVec<Scalar> Y_col;
    AlignedVec<Scalar> taus;
    AlignedVec<Scalar> W;
    AlignedVec<Scalar> T_mat;
    AlignedVec<Scalar> v_buf; 

    void resize(Index m, Index n, Index block) {
        const Index max_dim = std::max(m, n);
        Y_col.assign(static_cast<std::size_t>(m * block),      Scalar{0});
        taus.assign (static_cast<std::size_t>(block),           Scalar{0});
        W.assign    (static_cast<std::size_t>(block * max_dim), Scalar{0});
        T_mat.assign(static_cast<std::size_t>(block * block),   Scalar{0});
        v_buf.assign(static_cast<std::size_t>(m),               Scalar{0});
    }
};

// simd_dot computes the dot product of two vectors using SIMD instructions when possible
template<typename Scalar>
[[gnu::always_inline]] inline
Scalar simd_dot(const Scalar* __restrict__ x,
                const Scalar* __restrict__ y,
                Index len) noexcept
{
    Scalar acc{0};

#ifdef __AVX2__
    if constexpr (std::is_same_v<Scalar, double>) {
        __m256d vacc = _mm256_setzero_pd();
        Index i = 0;
        for (; i + 3 < len; i += 4) {
            __m256d vx = _mm256_loadu_pd(x + i);
            __m256d vy = _mm256_loadu_pd(y + i);
            vacc = _mm256_fmadd_pd(vx, vy, vacc);
        }
        double tmp[4];
        _mm256_storeu_pd(tmp, vacc);
        acc = tmp[0] + tmp[1] + tmp[2] + tmp[3];
        for (; i < len; ++i) acc += x[i] * y[i];
        return acc;
    } else if constexpr (std::is_same_v<Scalar, float>) {
        __m256 vacc = _mm256_setzero_ps();
        Index i = 0;
        for (; i + 7 < len; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vy = _mm256_loadu_ps(y + i);
            vacc = _mm256_fmadd_ps(vx, vy, vacc);
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, vacc);
        for (int k = 0; k < 8; ++k) acc += tmp[k];
        for (; i < len; ++i) acc += x[i] * y[i];
        return acc;
    }
#endif
    #pragma omp simd reduction(+:acc)
    for (Index i = 0; i < len; ++i) acc += x[i] * y[i];
    return acc;
}

// compute_householder
// Computes Householder vector v and scalar tau such that
//   (I - tau * v * v^T) * x  =  -sign(x[0]) * ||x|| * e_1
template<typename Scalar>
[[nodiscard]] bool compute_householder(
    const Scalar* col_ptr,
    Index col_stride,
    Index len,
    Scalar* v,
    Scalar& tau) noexcept
{
    for (Index i = 0; i < len; ++i)
        v[i] = col_ptr[i * col_stride];

    Scalar sigma = 0;
    for (Index i = 1; i < len; ++i)
        sigma += v[i] * v[i];

    Scalar x0 = v[0];

    if (sigma == 0 && x0 >= 0) {
        tau = 0;
        return false;
    }

    Scalar norm = std::sqrt(x0 * x0 + sigma);
    Scalar beta = (x0 <= 0) ? norm : -norm;

    Scalar inv = 1.0 / (x0 - beta);

    v[0] = 1;
    for (Index i = 1; i < len; ++i)
        v[i] *= inv;

    tau = (beta - x0) / beta;

    return true;
}

// build_T_matrix constructs the T matrix for a block of Householder reflectors
template<typename Scalar>
    requires Numeric<Scalar>
static void build_T_matrix(
    const Scalar* __restrict__ Y_col,
    Index panel_rows,
    Index nb,
    const Scalar* __restrict__ taus,
    Scalar* __restrict__ T_data) noexcept
{
    std::fill(T_data, T_data + nb * nb, Scalar{0});

    for (Index i = 0; i < nb; ++i) {
        T_data[i + i * nb] = taus[i];
        if (i == 0) continue;

        const Scalar* __restrict__ yi = Y_col + i * panel_rows;

        Scalar z[QR_BLOCK] = {};
        for (Index j = 0; j < i; ++j)
            z[j] = simd_dot(Y_col + j * panel_rows, yi, panel_rows);

        Scalar Tz[QR_BLOCK] = {};
        for (Index j = 0; j < i; ++j) {
            if (z[j] == Scalar{0}) continue;
            for (Index r = 0; r <= j; ++r)
                Tz[r] += T_data[r + j * nb] * z[j];
        }

        for (Index r = 0; r < i; ++r)
            T_data[r + i * nb] = -taus[i] * Tz[r];
    }
}

// apply_block_reflector applies the block Householder transformation to a matrix A
template<typename Scalar>
    requires Numeric<Scalar>
static void apply_block_reflector(
    Matrix<Scalar>& A,
    const Scalar* __restrict__ Y_col,
    const Scalar* __restrict__ T_data,
    Index panel_rows,
    Index nb,
    Index row0,
    Index col0,
    Scalar* __restrict__ W_buf) noexcept
{
    const Index n     = A.cols();
    const Index ncols = n - col0;
    const Index nrows = panel_rows;

    if (ncols <= 0 || nrows <= 0 || nb == 0) return;

    Scalar* __restrict__ Adata =
        static_cast<Scalar*>(__builtin_assume_aligned(A.data(), 64));
    const Index lda = n;

    // Step 1:  W = Y^T * A[row0:, col0:]
    //          W[i, j] = sum_r  Y[r,i] * A[row0+r, col0+j]
    std::fill(W_buf, W_buf + nb * ncols, Scalar{0});

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided) if(nrows * ncols > 8192)
    #endif
    for (Index i = 0; i < nb; ++i) {
        const Scalar* __restrict__ yi = Y_col + i * panel_rows;
        Scalar* __restrict__       wi = W_buf  + i * ncols;

        for (Index r = 0; r < nrows; ++r) {
            const Scalar yir = yi[r];
            if (yir == Scalar{0}) continue;

            const Scalar* __restrict__ arow = Adata + (row0 + r) * lda + col0;

#ifdef __AVX2__
            if constexpr (std::is_same_v<Scalar, double>) {
                const __m256d vy = _mm256_set1_pd(yir);
                Index j = 0;
                for (; j + 15 < ncols; j += 16) {
                    __builtin_prefetch(arow + j + 64, 0, 1);
                    _mm256_storeu_pd(wi+j,    _mm256_fmadd_pd(vy, _mm256_loadu_pd(arow+j),    _mm256_loadu_pd(wi+j)));
                    _mm256_storeu_pd(wi+j+4,  _mm256_fmadd_pd(vy, _mm256_loadu_pd(arow+j+4),  _mm256_loadu_pd(wi+j+4)));
                    _mm256_storeu_pd(wi+j+8,  _mm256_fmadd_pd(vy, _mm256_loadu_pd(arow+j+8),  _mm256_loadu_pd(wi+j+8)));
                    _mm256_storeu_pd(wi+j+12, _mm256_fmadd_pd(vy, _mm256_loadu_pd(arow+j+12), _mm256_loadu_pd(wi+j+12)));
                }
                for (; j + 3 < ncols; j += 4)
                    _mm256_storeu_pd(wi+j, _mm256_fmadd_pd(vy, _mm256_loadu_pd(arow+j), _mm256_loadu_pd(wi+j)));
                for (; j < ncols; ++j) wi[j] += yir * arow[j];
            } else if constexpr (std::is_same_v<Scalar, float>) {
                const __m256 vy = _mm256_set1_ps(yir);
                Index j = 0;
                for (; j + 31 < ncols; j += 32) {
                    __builtin_prefetch(arow + j + 128, 0, 1);
                    _mm256_storeu_ps(wi+j,    _mm256_fmadd_ps(vy, _mm256_loadu_ps(arow+j),    _mm256_loadu_ps(wi+j)));
                    _mm256_storeu_ps(wi+j+8,  _mm256_fmadd_ps(vy, _mm256_loadu_ps(arow+j+8),  _mm256_loadu_ps(wi+j+8)));
                    _mm256_storeu_ps(wi+j+16, _mm256_fmadd_ps(vy, _mm256_loadu_ps(arow+j+16), _mm256_loadu_ps(wi+j+16)));
                    _mm256_storeu_ps(wi+j+24, _mm256_fmadd_ps(vy, _mm256_loadu_ps(arow+j+24), _mm256_loadu_ps(wi+j+24)));
                }
                for (; j + 7 < ncols; j += 8)
                    _mm256_storeu_ps(wi+j, _mm256_fmadd_ps(vy, _mm256_loadu_ps(arow+j), _mm256_loadu_ps(wi+j)));
                for (; j < ncols; ++j) wi[j] += yir * arow[j];
            } else {
                for (Index j = 0; j < ncols; ++j) wi[j] += yir * arow[j];
            }
#else
            #pragma omp simd
            for (Index j = 0; j < ncols; ++j) wi[j] += yir * arow[j];
#endif
        }
    }

    // Step 2:  W <- T^T * W
    for (Index i = nb; i-- > 0;) {
        Scalar* __restrict__ wi = W_buf + i * ncols;

        const Scalar Tii = T_data[i + i * nb];
#ifdef __AVX2__
        if constexpr (std::is_same_v<Scalar, double>) {
            const __m256d vt = _mm256_set1_pd(Tii);
            Index j = 0;
            for (; j + 3 < ncols; j += 4)
                _mm256_storeu_pd(wi+j, _mm256_mul_pd(vt, _mm256_loadu_pd(wi+j)));
            for (; j < ncols; ++j) wi[j] *= Tii;
        } else if constexpr (std::is_same_v<Scalar, float>) {
            const __m256 vt = _mm256_set1_ps(Tii);
            Index j = 0;
            for (; j + 7 < ncols; j += 8)
                _mm256_storeu_ps(wi+j, _mm256_mul_ps(vt, _mm256_loadu_ps(wi+j)));
            for (; j < ncols; ++j) wi[j] *= Tii;
        } else {
            for (Index j = 0; j < ncols; ++j) wi[j] *= Tii;
        }
#else
        #pragma omp simd
        for (Index j = 0; j < ncols; ++j) wi[j] *= Tii;
#endif
        for (Index l = 0; l < i; ++l) {
            const Scalar Tli = T_data[l + i * nb];
            if (Tli == Scalar{0}) continue;
            const Scalar* __restrict__ wl = W_buf + l * ncols;

#ifdef __AVX2__
            if constexpr (std::is_same_v<Scalar, double>) {
                const __m256d vt = _mm256_set1_pd(Tli);
                Index j = 0;
                for (; j + 3 < ncols; j += 4)
                    _mm256_storeu_pd(wi+j, _mm256_fmadd_pd(vt, _mm256_loadu_pd(wl+j), _mm256_loadu_pd(wi+j)));
                for (; j < ncols; ++j) wi[j] += Tli * wl[j];
            } else if constexpr (std::is_same_v<Scalar, float>) {
                const __m256 vt = _mm256_set1_ps(Tli);
                Index j = 0;
                for (; j + 7 < ncols; j += 8)
                    _mm256_storeu_ps(wi+j, _mm256_fmadd_ps(vt, _mm256_loadu_ps(wl+j), _mm256_loadu_ps(wi+j)));
                for (; j < ncols; ++j) wi[j] += Tli * wl[j];
            } else {
                for (Index j = 0; j < ncols; ++j) wi[j] += Tli * wl[j];
            }
#else
            #pragma omp simd
            for (Index j = 0; j < ncols; ++j) wi[j] += Tli * wl[j];
#endif
        }
    }

    // Step 3:  A <- A - Y * W
    //          A[row0+r, col0+j] -= sum_i  Y[r,i] * W[i,j]
    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided) if(nrows * ncols > 8192)
    #endif
    for (Index r = 0; r < nrows; ++r) {
        Scalar* __restrict__ arow = Adata + (row0 + r) * lda + col0;

        for (Index i = 0; i < nb; ++i) {
            const Scalar yir = Y_col[r + i * panel_rows];
            if (yir == Scalar{0}) continue;

            const Scalar* __restrict__ wi = W_buf + i * ncols;

#ifdef __AVX2__
            if constexpr (std::is_same_v<Scalar, double>) {
                const __m256d vy = _mm256_set1_pd(yir);
                Index j = 0;
                for (; j + 15 < ncols; j += 16) {
                    __builtin_prefetch(wi + j + 64, 0, 1);
                    _mm256_storeu_pd(arow+j,    _mm256_fnmadd_pd(vy, _mm256_loadu_pd(wi+j),    _mm256_loadu_pd(arow+j)));
                    _mm256_storeu_pd(arow+j+4,  _mm256_fnmadd_pd(vy, _mm256_loadu_pd(wi+j+4),  _mm256_loadu_pd(arow+j+4)));
                    _mm256_storeu_pd(arow+j+8,  _mm256_fnmadd_pd(vy, _mm256_loadu_pd(wi+j+8),  _mm256_loadu_pd(arow+j+8)));
                    _mm256_storeu_pd(arow+j+12, _mm256_fnmadd_pd(vy, _mm256_loadu_pd(wi+j+12), _mm256_loadu_pd(arow+j+12)));
                }
                for (; j + 3 < ncols; j += 4)
                    _mm256_storeu_pd(arow+j, _mm256_fnmadd_pd(vy, _mm256_loadu_pd(wi+j), _mm256_loadu_pd(arow+j)));
                for (; j < ncols; ++j) arow[j] -= yir * wi[j];
            } else if constexpr (std::is_same_v<Scalar, float>) {
                const __m256 vy = _mm256_set1_ps(yir);
                Index j = 0;
                for (; j + 31 < ncols; j += 32) {
                    __builtin_prefetch(wi + j + 128, 0, 1);
                    _mm256_storeu_ps(arow+j,    _mm256_fnmadd_ps(vy, _mm256_loadu_ps(wi+j),    _mm256_loadu_ps(arow+j)));
                    _mm256_storeu_ps(arow+j+8,  _mm256_fnmadd_ps(vy, _mm256_loadu_ps(wi+j+8),  _mm256_loadu_ps(arow+j+8)));
                    _mm256_storeu_ps(arow+j+16, _mm256_fnmadd_ps(vy, _mm256_loadu_ps(wi+j+16), _mm256_loadu_ps(arow+j+16)));
                    _mm256_storeu_ps(arow+j+24, _mm256_fnmadd_ps(vy, _mm256_loadu_ps(wi+j+24), _mm256_loadu_ps(arow+j+24)));
                }
                for (; j + 7 < ncols; j += 8)
                    _mm256_storeu_ps(arow+j, _mm256_fnmadd_ps(vy, _mm256_loadu_ps(wi+j), _mm256_loadu_ps(arow+j)));
                for (; j < ncols; ++j) arow[j] -= yir * wi[j];
            } else {
                for (Index j = 0; j < ncols; ++j) arow[j] -= yir * wi[j];
            }
#else
            #pragma omp simd
            for (Index j = 0; j < ncols; ++j) arow[j] -= yir * wi[j];
#endif
        }
    }
}

// apply_block_reflector_right applies the block Householder transformation from the right side: A <- A * (I - Y T^T Y^T)
template<typename Scalar>
    requires Numeric<Scalar>
static void apply_block_reflector_right(
    Matrix<Scalar>& Q,
    const Scalar* __restrict__ Y_col,
    const Scalar* __restrict__ T_data,
    Index panel_rows,
    Index nb,
    Index row0,
    Scalar* __restrict__ W_buf) noexcept
{
    // Q is m x m (row-major, lda = m)
    // Q <- Q * (I - Y T^T Y^T)
    //    = Q - (Q * Y) * T^T * Y^T
    //
    // Y_col[j * panel_rows + r] = Y[r, j]   col-major, r in [0, panel_rows)
    // T_data[i + j * nb]        = T[i, j]   col-major, upper-triangular
    // W_buf: scratch, size >= m * nb  (W stored row-major: W[i,j] = W_buf[i*nb+j])

    const Index m   = Q.rows();
    const Index lda = m;   // Q is square, row-major

    if (panel_rows <= 0 || nb == 0) return;

    Scalar* __restrict__ Qdata =
        static_cast<Scalar*>(__builtin_assume_aligned(Q.data(), 64));

    // Step 1:  W = Q * Y    (m x nb, row-major: W[i,j] = W_buf[i*nb+j])
    //          W[i, j] = sum_{r=0}^{panel_rows-1}  Q[i, row0+r] * Y[r, j]
    std::fill(W_buf, W_buf + m * nb, Scalar{0});

    for (Index i = 0; i < m; ++i) {
        const Scalar* __restrict__ qrow = Qdata + i * lda + row0;
        Scalar*       __restrict__ wrow = W_buf  + i * nb;
        for (Index r = 0; r < panel_rows; ++r) {
            const Scalar qir = qrow[r];
            if (qir == Scalar{0}) continue;
            for (Index j = 0; j < nb; ++j)
                wrow[j] += qir * Y_col[j * panel_rows + r];
        }
    }

    // Step 2:  W <- W * T^T
    for (Index i = 0; i < m; ++i) {
        Scalar* __restrict__ wrow = W_buf + i * nb;
        for (Index j = nb; j-- > 0;) {
            Scalar s = T_data[j + j * nb] * wrow[j];
            for (Index l = 0; l < j; ++l)
                s += T_data[l + j * nb] * wrow[l];
            wrow[j] = s;
        }
    }

    // Step 3:  Q[:, row0:row0+panel_rows] -= W * Y^T
    //          Q[i, row0+r] -= sum_{j=0}^{nb-1}  W[i,j] * Y[r,j]
    for (Index i = 0; i < m; ++i) {
        Scalar*       __restrict__ qrow = Qdata + i * lda + row0;
        const Scalar* __restrict__ wrow = W_buf  + i * nb;
        for (Index j = 0; j < nb; ++j) {
            const Scalar wij = wrow[j];
            if (wij == Scalar{0}) continue;
            for (Index r = 0; r < panel_rows; ++r)
                qrow[r] -= wij * Y_col[j * panel_rows + r];
        }
    }
}

// panel_qr_step performs one step of the blocked Householder QR factorization on a panel of the matrix R
template<typename Scalar>
    requires Numeric<Scalar>
static void panel_qr_step(
    Matrix<Scalar>& R,
    Index k,
    Index nb,
    Scalar* __restrict__ Y_col,
    Scalar* __restrict__ taus,
    Scalar* __restrict__ v_buf) noexcept
{
    const Index m          = R.rows();
    const Index n          = R.cols();
    const Index panel_rows = m - k;
    const Index panel_cols = std::min(nb, n - k);

    Scalar* __restrict__ Rdata =
        static_cast<Scalar*>(__builtin_assume_aligned(R.data(), 64));

    for (Index j = 0; j < panel_cols; ++j) {
        const Index col = k + j;
        const Index len = m - col;

        const bool active = compute_householder(
            Rdata + col * n + col,
            n,
            len,
            v_buf,
            taus[j]);

        Scalar* __restrict__ yj = Y_col + j * panel_rows;
        std::fill(yj, yj + panel_rows, Scalar{0});
        if (!active) continue;

        yj[j] = Scalar{1};
        for (Index i = 1; i < len; ++i)
            yj[j + i] = v_buf[i];

        {
            Scalar vtx{0};
            for (Index i = 0; i < len; ++i)
                vtx += v_buf[i] * Rdata[(col + i) * n + col];
            const Scalar tau_j = taus[j];
            for (Index i = 0; i < len; ++i)
                Rdata[(col + i) * n + col] -= tau_j * v_buf[i] * vtx;
        }
        for (Index i = 1; i < len; ++i)
            Rdata[(col + i) * n + col] = Scalar{0};

        const Index trail_start = col + 1;
        const Index trail_end   = std::min(k + nb, n);
        const Index trail_width = trail_end - trail_start;
        if (trail_width <= 0) continue;

        Scalar w[QR_BLOCK] = {};
        const Scalar tau_j = taus[j];

        for (Index i = 0; i < len; ++i) {
            const Scalar vi = v_buf[i];
            if (vi == Scalar{0}) continue;
            const Scalar* __restrict__ arow = Rdata + (col + i) * n + trail_start;
            #pragma omp simd
            for (Index jj = 0; jj < trail_width; ++jj)
                w[jj] += vi * arow[jj];
        }
        for (Index i = 0; i < len; ++i) {
            const Scalar fac = tau_j * v_buf[i];
            if (fac == Scalar{0}) continue;
            Scalar* __restrict__ arow = Rdata + (col + i) * n + trail_start;
            #pragma omp simd
            for (Index jj = 0; jj < trail_width; ++jj)
                arow[jj] -= fac * w[jj];
        }
    }
}

// qr_householder performs the QR factorization of a matrix using the blocked Householder algorithm
template<typename Scalar>
    requires Numeric<Scalar>
QRResult<Scalar> qr_householder(const Matrix<Scalar>& input) {
    if (input.rows() == 0 || input.cols() == 0)
        throw std::invalid_argument("qr_householder: matrix must not be empty");

    const Index m = input.rows();
    const Index n = input.cols();

    Matrix<Scalar> R = input;
    Matrix<Scalar> Q = Matrix<Scalar>::identity(m);

    QRWorkspace<Scalar> ws;
    ws.resize(m, n, QR_BLOCK);

    const Index steps = std::min(m, n);

    for (Index k = 0; k < steps; k += QR_BLOCK) {
        const Index nb         = std::min(static_cast<Index>(QR_BLOCK), steps - k);
        const Index panel_rows = m - k;

        Scalar* Y_col  = ws.Y_col.data();
        Scalar* taus   = ws.taus.data();
        Scalar* T_data = ws.T_mat.data();
        Scalar* W_buf  = ws.W.data();
        Scalar* v_buf  = ws.v_buf.data();

        std::fill(Y_col,  Y_col  + panel_rows * nb, Scalar{0});
        std::fill(taus,   taus   + nb,               Scalar{0});
        std::fill(T_data, T_data + nb * nb,           Scalar{0});

        panel_qr_step(R, k, nb, Y_col, taus, v_buf);
        build_T_matrix(Y_col, panel_rows, nb, taus, T_data);

        if (k + nb < n)
            apply_block_reflector(R, Y_col, T_data, panel_rows, nb, k, k + nb, W_buf);

        apply_block_reflector_right(Q, Y_col, T_data, panel_rows, nb, k, W_buf);
    }

    #pragma omp parallel for schedule(static) if(m * n > 65536)
    for (Index i = 1; i < m; ++i) {
        const Index jend = std::min(i, n);
        Scalar* row = R.data() + i * n;
        #pragma omp simd
        for (Index j = 0; j < jend; ++j)
            row[j] = Scalar{0};
    }

    return {Q, R};
}

template<typename Scalar>
    requires Numeric<Scalar>
QRResult<Scalar> qr_givens(const Matrix<Scalar>& input) {
    return qr_householder(input);
}

template<typename Scalar>
    requires Numeric<Scalar>
QRResult<Scalar> qr(const Matrix<Scalar>& input) {
    return qr_householder(input);
}

} // namespace pla
