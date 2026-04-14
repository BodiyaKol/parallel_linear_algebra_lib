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

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace pla {

template<typename T, std::size_t Alignment = 64>
struct AlignedAllocator {
    using value_type = T;
    template<typename U> struct rebind { using other = AlignedAllocator<U, Alignment>; };
    AlignedAllocator() noexcept = default;
    template<typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{Alignment}));
    }
    void deallocate(T* p, std::size_t) noexcept {
        ::operator delete(p, std::align_val_t{Alignment});
    }
    template<typename U, std::size_t A2>
    bool operator==(const AlignedAllocator<U,A2>&) const noexcept { return Alignment==A2; }
    template<typename U, std::size_t A2>
    bool operator!=(const AlignedAllocator<U,A2>& o) const noexcept { return !(*this==o); }
};

template<typename Scalar>
using AlignedVec = std::vector<Scalar, AlignedAllocator<Scalar, 64>>;

// OMP parallel region spawn costs ~5-15 µs on typical Linux.
// Only worth it when work > ~50K doubles of FMA.
static constexpr Index OMP_ROWS_THRESHOLD  = 96;    // min rows to parallelise
static constexpr Index OMP_WORK_THRESHOLD  = 16384; // min rows*cols to parallelise
static constexpr Index QR_BLOCK            = 64;

template<typename Scalar>
struct QRWorkspace {
    AlignedVec<Scalar> Y_col;
    AlignedVec<Scalar> taus;
    AlignedVec<Scalar> W;
    AlignedVec<Scalar> W_Q;
    AlignedVec<Scalar> T_mat;
    AlignedVec<Scalar> v_buf;
    AlignedVec<Scalar> thread_W;
    int nthreads{1};

    void resize(Index m, Index n, Index block) {
#ifdef _OPENMP
        nthreads = omp_get_max_threads();
#endif
        const Index max_dim = std::max(m, n);
        Y_col.assign   (static_cast<std::size_t>(m * block),                    Scalar{0});
        taus.assign    (static_cast<std::size_t>(block),                        Scalar{0});
        W.assign       (static_cast<std::size_t>(block * max_dim),              Scalar{0});
        W_Q.assign     (static_cast<std::size_t>(m * block),                    Scalar{0});
        T_mat.assign   (static_cast<std::size_t>(block * block),                Scalar{0});
        v_buf.assign   (static_cast<std::size_t>(m),                            Scalar{0});
        thread_W.assign(static_cast<std::size_t>(nthreads * block * max_dim),   Scalar{0});
    }
};

// ---------------------------------------------------------------
//  SIMD primitives — AVX-512 primary, AVX2 fallback, NEON, scalar last
// ---------------------------------------------------------------
template<typename Scalar>
[[gnu::always_inline]] inline
Scalar simd_dot(const Scalar* __restrict__ x,
                const Scalar* __restrict__ y,
                Index len) noexcept
{
    Scalar acc{0};
#if defined(__AVX512F__)
    if constexpr (std::is_same_v<Scalar, double>) {
        __m512d v0 = _mm512_setzero_pd(), v1 = _mm512_setzero_pd(),
                v2 = _mm512_setzero_pd(), v3 = _mm512_setzero_pd();
        Index i = 0;
        for (; i + 31 < len; i += 32) {
            v0 = _mm512_fmadd_pd(_mm512_loadu_pd(x+i),    _mm512_loadu_pd(y+i),    v0);
            v1 = _mm512_fmadd_pd(_mm512_loadu_pd(x+i+8),  _mm512_loadu_pd(y+i+8),  v1);
            v2 = _mm512_fmadd_pd(_mm512_loadu_pd(x+i+16), _mm512_loadu_pd(y+i+16), v2);
            v3 = _mm512_fmadd_pd(_mm512_loadu_pd(x+i+24), _mm512_loadu_pd(y+i+24), v3);
        }
        v0 = _mm512_add_pd(_mm512_add_pd(v0,v1), _mm512_add_pd(v2,v3));
        for (; i + 7 < len; i += 8)
            v0 = _mm512_fmadd_pd(_mm512_loadu_pd(x+i), _mm512_loadu_pd(y+i), v0);
        acc = _mm512_reduce_add_pd(v0);
        for (; i < len; ++i) acc += x[i]*y[i];
        return acc;
    } else if constexpr (std::is_same_v<Scalar, float>) {
        __m512 v0 = _mm512_setzero_ps(), v1 = _mm512_setzero_ps();
        Index i = 0;
        for (; i + 31 < len; i += 32) {
            v0 = _mm512_fmadd_ps(_mm512_loadu_ps(x+i),    _mm512_loadu_ps(y+i),    v0);
            v1 = _mm512_fmadd_ps(_mm512_loadu_ps(x+i+16), _mm512_loadu_ps(y+i+16), v1);
        }
        v0 = _mm512_add_ps(v0,v1);
        for (; i + 15 < len; i += 16)
            v0 = _mm512_fmadd_ps(_mm512_loadu_ps(x+i), _mm512_loadu_ps(y+i), v0);
        acc = _mm512_reduce_add_ps(v0);
        for (; i < len; ++i) acc += x[i]*y[i];
        return acc;
    }
#elif defined(__AVX2__)
    if constexpr (std::is_same_v<Scalar, double>) {
        __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd(),
                v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();
        Index i = 0;
        for (; i + 15 < len; i += 16) {
            v0 = _mm256_fmadd_pd(_mm256_loadu_pd(x+i),    _mm256_loadu_pd(y+i),    v0);
            v1 = _mm256_fmadd_pd(_mm256_loadu_pd(x+i+4),  _mm256_loadu_pd(y+i+4),  v1);
            v2 = _mm256_fmadd_pd(_mm256_loadu_pd(x+i+8),  _mm256_loadu_pd(y+i+8),  v2);
            v3 = _mm256_fmadd_pd(_mm256_loadu_pd(x+i+12), _mm256_loadu_pd(y+i+12), v3);
        }
        v0 = _mm256_add_pd(_mm256_add_pd(v0,v1),_mm256_add_pd(v2,v3));
        for (; i + 3 < len; i += 4)
            v0 = _mm256_fmadd_pd(_mm256_loadu_pd(x+i),_mm256_loadu_pd(y+i),v0);
        double tmp[4]; _mm256_storeu_pd(tmp,v0);
        acc = tmp[0]+tmp[1]+tmp[2]+tmp[3];
        for (; i < len; ++i) acc += x[i]*y[i];
        return acc;
    } else if constexpr (std::is_same_v<Scalar, float>) {
        __m256 v0 = _mm256_setzero_ps(), v1 = _mm256_setzero_ps();
        Index i = 0;
        for (; i + 15 < len; i += 16) {
            v0 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i),   _mm256_loadu_ps(y+i),   v0);
            v1 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+8), _mm256_loadu_ps(y+i+8), v1);
        }
        v0 = _mm256_add_ps(v0,v1);
        for (; i + 7 < len; i += 8)
            v0 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i),_mm256_loadu_ps(y+i),v0);
        float tmp[8]; _mm256_storeu_ps(tmp,v0);
        for (int k=0;k<8;++k) acc+=tmp[k];
        for (; i < len; ++i) acc += x[i]*y[i];
        return acc;
    }
#elif defined(__ARM_NEON)
    if constexpr (std::is_same_v<Scalar, double>) {
        float64x2_t v0 = vdupq_n_f64(0.0), v1 = vdupq_n_f64(0.0),
                    v2 = vdupq_n_f64(0.0), v3 = vdupq_n_f64(0.0);
        Index i = 0;
        for (; i + 7 < len; i += 8) {
            v0 = vfmaq_f64(v0, vld1q_f64(x+i),   vld1q_f64(y+i));
            v1 = vfmaq_f64(v1, vld1q_f64(x+i+2), vld1q_f64(y+i+2));
            v2 = vfmaq_f64(v2, vld1q_f64(x+i+4), vld1q_f64(y+i+4));
            v3 = vfmaq_f64(v3, vld1q_f64(x+i+6), vld1q_f64(y+i+6));
        }
        v0 = vaddq_f64(vaddq_f64(v0, v1), vaddq_f64(v2, v3));
        for (; i + 1 < len; i += 2)
            v0 = vfmaq_f64(v0, vld1q_f64(x+i), vld1q_f64(y+i));
        acc = vaddvq_f64(v0);
        for (; i < len; ++i) acc += x[i]*y[i];
        return acc;
    } else if constexpr (std::is_same_v<Scalar, float>) {
        float32x4_t v0 = vdupq_n_f32(0.f), v1 = vdupq_n_f32(0.f),
                    v2 = vdupq_n_f32(0.f), v3 = vdupq_n_f32(0.f);
        Index i = 0;
        for (; i + 15 < len; i += 16) {
            v0 = vfmaq_f32(v0, vld1q_f32(x+i),    vld1q_f32(y+i));
            v1 = vfmaq_f32(v1, vld1q_f32(x+i+4),  vld1q_f32(y+i+4));
            v2 = vfmaq_f32(v2, vld1q_f32(x+i+8),  vld1q_f32(y+i+8));
            v3 = vfmaq_f32(v3, vld1q_f32(x+i+12), vld1q_f32(y+i+12));
        }
        v0 = vaddq_f32(vaddq_f32(v0, v1), vaddq_f32(v2, v3));
        for (; i + 3 < len; i += 4)
            v0 = vfmaq_f32(v0, vld1q_f32(x+i), vld1q_f32(y+i));
        acc = vaddvq_f32(v0);
        for (; i < len; ++i) acc += x[i]*y[i];
        return acc;
    }
#endif
    #pragma omp simd reduction(+:acc)
    for (Index i = 0; i < len; ++i) acc += x[i]*y[i];
    return acc;
}

template<typename Scalar>
[[gnu::always_inline]] inline
void simd_axpy(Scalar* __restrict__ dst,
               const Scalar* __restrict__ src,
               Scalar alpha, Index len) noexcept
{
    if (len <= 0) return;
#if defined(__AVX512F__)
    if constexpr (std::is_same_v<Scalar, double>) {
        const __m512d va = _mm512_set1_pd(alpha);
        Index j = 0;
        for (; j + 31 < len; j += 32) {
            __builtin_prefetch(src+j+128, 0, 1);
            _mm512_storeu_pd(dst+j,    _mm512_fmadd_pd(va,_mm512_loadu_pd(src+j),    _mm512_loadu_pd(dst+j)));
            _mm512_storeu_pd(dst+j+8,  _mm512_fmadd_pd(va,_mm512_loadu_pd(src+j+8),  _mm512_loadu_pd(dst+j+8)));
            _mm512_storeu_pd(dst+j+16, _mm512_fmadd_pd(va,_mm512_loadu_pd(src+j+16), _mm512_loadu_pd(dst+j+16)));
            _mm512_storeu_pd(dst+j+24, _mm512_fmadd_pd(va,_mm512_loadu_pd(src+j+24), _mm512_loadu_pd(dst+j+24)));
        }
        for (; j + 7 < len; j += 8)
            _mm512_storeu_pd(dst+j, _mm512_fmadd_pd(va,_mm512_loadu_pd(src+j),_mm512_loadu_pd(dst+j)));
        for (; j < len; ++j) dst[j] += alpha*src[j];
        return;
    } else if constexpr (std::is_same_v<Scalar, float>) {
        const __m512 va = _mm512_set1_ps(alpha);
        Index j = 0;
        for (; j + 63 < len; j += 64) {
            _mm512_storeu_ps(dst+j,    _mm512_fmadd_ps(va,_mm512_loadu_ps(src+j),    _mm512_loadu_ps(dst+j)));
            _mm512_storeu_ps(dst+j+16, _mm512_fmadd_ps(va,_mm512_loadu_ps(src+j+16), _mm512_loadu_ps(dst+j+16)));
            _mm512_storeu_ps(dst+j+32, _mm512_fmadd_ps(va,_mm512_loadu_ps(src+j+32), _mm512_loadu_ps(dst+j+32)));
            _mm512_storeu_ps(dst+j+48, _mm512_fmadd_ps(va,_mm512_loadu_ps(src+j+48), _mm512_loadu_ps(dst+j+48)));
        }
        for (; j + 15 < len; j += 16)
            _mm512_storeu_ps(dst+j, _mm512_fmadd_ps(va,_mm512_loadu_ps(src+j),_mm512_loadu_ps(dst+j)));
        for (; j < len; ++j) dst[j] += alpha*src[j];
        return;
    }
#elif defined(__AVX2__)
    if constexpr (std::is_same_v<Scalar, double>) {
        const __m256d va = _mm256_set1_pd(alpha);
        Index j = 0;
        for (; j + 15 < len; j += 16) {
            __builtin_prefetch(src+j+64,  0, 1);
            _mm256_storeu_pd(dst+j,    _mm256_fmadd_pd(va,_mm256_loadu_pd(src+j),    _mm256_loadu_pd(dst+j)));
            _mm256_storeu_pd(dst+j+4,  _mm256_fmadd_pd(va,_mm256_loadu_pd(src+j+4),  _mm256_loadu_pd(dst+j+4)));
            _mm256_storeu_pd(dst+j+8,  _mm256_fmadd_pd(va,_mm256_loadu_pd(src+j+8),  _mm256_loadu_pd(dst+j+8)));
            _mm256_storeu_pd(dst+j+12, _mm256_fmadd_pd(va,_mm256_loadu_pd(src+j+12), _mm256_loadu_pd(dst+j+12)));
        }
        for (; j + 3 < len; j += 4)
            _mm256_storeu_pd(dst+j, _mm256_fmadd_pd(va,_mm256_loadu_pd(src+j),_mm256_loadu_pd(dst+j)));
        for (; j < len; ++j) dst[j] += alpha*src[j];
        return;
    } else if constexpr (std::is_same_v<Scalar, float>) {
        const __m256 va = _mm256_set1_ps(alpha);
        Index j = 0;
        for (; j + 31 < len; j += 32) {
            _mm256_storeu_ps(dst+j,    _mm256_fmadd_ps(va,_mm256_loadu_ps(src+j),    _mm256_loadu_ps(dst+j)));
            _mm256_storeu_ps(dst+j+8,  _mm256_fmadd_ps(va,_mm256_loadu_ps(src+j+8),  _mm256_loadu_ps(dst+j+8)));
            _mm256_storeu_ps(dst+j+16, _mm256_fmadd_ps(va,_mm256_loadu_ps(src+j+16), _mm256_loadu_ps(dst+j+16)));
            _mm256_storeu_ps(dst+j+24, _mm256_fmadd_ps(va,_mm256_loadu_ps(src+j+24), _mm256_loadu_ps(dst+j+24)));
        }
        for (; j + 7 < len; j += 8)
            _mm256_storeu_ps(dst+j, _mm256_fmadd_ps(va,_mm256_loadu_ps(src+j),_mm256_loadu_ps(dst+j)));
        for (; j < len; ++j) dst[j] += alpha*src[j];
        return;
    }
#elif defined(__ARM_NEON)
    if constexpr (std::is_same_v<Scalar, double>) {
        const float64x2_t va = vdupq_n_f64(alpha);
        Index j = 0;
        for (; j + 7 < len; j += 8) {
            __builtin_prefetch(src+j+32, 0, 1);
            vst1q_f64(dst+j,   vfmaq_f64(vld1q_f64(dst+j),   va, vld1q_f64(src+j)));
            vst1q_f64(dst+j+2, vfmaq_f64(vld1q_f64(dst+j+2), va, vld1q_f64(src+j+2)));
            vst1q_f64(dst+j+4, vfmaq_f64(vld1q_f64(dst+j+4), va, vld1q_f64(src+j+4)));
            vst1q_f64(dst+j+6, vfmaq_f64(vld1q_f64(dst+j+6), va, vld1q_f64(src+j+6)));
        }
        for (; j + 1 < len; j += 2)
            vst1q_f64(dst+j, vfmaq_f64(vld1q_f64(dst+j), va, vld1q_f64(src+j)));
        for (; j < len; ++j) dst[j] += alpha*src[j];
        return;
    } else if constexpr (std::is_same_v<Scalar, float>) {
        const float32x4_t va = vdupq_n_f32(alpha);
        Index j = 0;
        for (; j + 15 < len; j += 16) {
            __builtin_prefetch(src+j+64, 0, 1);
            vst1q_f32(dst+j,    vfmaq_f32(vld1q_f32(dst+j),    va, vld1q_f32(src+j)));
            vst1q_f32(dst+j+4,  vfmaq_f32(vld1q_f32(dst+j+4),  va, vld1q_f32(src+j+4)));
            vst1q_f32(dst+j+8,  vfmaq_f32(vld1q_f32(dst+j+8),  va, vld1q_f32(src+j+8)));
            vst1q_f32(dst+j+12, vfmaq_f32(vld1q_f32(dst+j+12), va, vld1q_f32(src+j+12)));
        }
        for (; j + 3 < len; j += 4)
            vst1q_f32(dst+j, vfmaq_f32(vld1q_f32(dst+j), va, vld1q_f32(src+j)));
        for (; j < len; ++j) dst[j] += alpha*src[j];
        return;
    }
#endif
    #pragma omp simd
    for (Index j = 0; j < len; ++j) dst[j] += alpha*src[j];
}

template<typename Scalar>
[[gnu::always_inline]] inline
void simd_naxpy(Scalar* __restrict__ dst,
                const Scalar* __restrict__ src,
                Scalar alpha, Index len) noexcept
{
    if (len <= 0) return;
#if defined(__AVX512F__)
    if constexpr (std::is_same_v<Scalar, double>) {
        const __m512d va = _mm512_set1_pd(alpha);
        Index j = 0;
        for (; j + 31 < len; j += 32) {
            __builtin_prefetch(src+j+128, 0, 1);
            _mm512_storeu_pd(dst+j,    _mm512_fnmadd_pd(va,_mm512_loadu_pd(src+j),    _mm512_loadu_pd(dst+j)));
            _mm512_storeu_pd(dst+j+8,  _mm512_fnmadd_pd(va,_mm512_loadu_pd(src+j+8),  _mm512_loadu_pd(dst+j+8)));
            _mm512_storeu_pd(dst+j+16, _mm512_fnmadd_pd(va,_mm512_loadu_pd(src+j+16), _mm512_loadu_pd(dst+j+16)));
            _mm512_storeu_pd(dst+j+24, _mm512_fnmadd_pd(va,_mm512_loadu_pd(src+j+24), _mm512_loadu_pd(dst+j+24)));
        }
        for (; j + 7 < len; j += 8)
            _mm512_storeu_pd(dst+j, _mm512_fnmadd_pd(va,_mm512_loadu_pd(src+j),_mm512_loadu_pd(dst+j)));
        for (; j < len; ++j) dst[j] -= alpha*src[j];
        return;
    } else if constexpr (std::is_same_v<Scalar, float>) {
        const __m512 va = _mm512_set1_ps(alpha);
        Index j = 0;
        for (; j + 63 < len; j += 64) {
            _mm512_storeu_ps(dst+j,    _mm512_fnmadd_ps(va,_mm512_loadu_ps(src+j),    _mm512_loadu_ps(dst+j)));
            _mm512_storeu_ps(dst+j+16, _mm512_fnmadd_ps(va,_mm512_loadu_ps(src+j+16), _mm512_loadu_ps(dst+j+16)));
            _mm512_storeu_ps(dst+j+32, _mm512_fnmadd_ps(va,_mm512_loadu_ps(src+j+32), _mm512_loadu_ps(dst+j+32)));
            _mm512_storeu_ps(dst+j+48, _mm512_fnmadd_ps(va,_mm512_loadu_ps(src+j+48), _mm512_loadu_ps(dst+j+48)));
        }
        for (; j + 15 < len; j += 16)
            _mm512_storeu_ps(dst+j, _mm512_fnmadd_ps(va,_mm512_loadu_ps(src+j),_mm512_loadu_ps(dst+j)));
        for (; j < len; ++j) dst[j] -= alpha*src[j];
        return;
    }
#elif defined(__AVX2__)
    if constexpr (std::is_same_v<Scalar, double>) {
        const __m256d va = _mm256_set1_pd(alpha);
        Index j = 0;
        for (; j + 15 < len; j += 16) {
            __builtin_prefetch(src+j+64,  0, 1);
            _mm256_storeu_pd(dst+j,    _mm256_fnmadd_pd(va,_mm256_loadu_pd(src+j),    _mm256_loadu_pd(dst+j)));
            _mm256_storeu_pd(dst+j+4,  _mm256_fnmadd_pd(va,_mm256_loadu_pd(src+j+4),  _mm256_loadu_pd(dst+j+4)));
            _mm256_storeu_pd(dst+j+8,  _mm256_fnmadd_pd(va,_mm256_loadu_pd(src+j+8),  _mm256_loadu_pd(dst+j+8)));
            _mm256_storeu_pd(dst+j+12, _mm256_fnmadd_pd(va,_mm256_loadu_pd(src+j+12), _mm256_loadu_pd(dst+j+12)));
        }
        for (; j + 3 < len; j += 4)
            _mm256_storeu_pd(dst+j, _mm256_fnmadd_pd(va,_mm256_loadu_pd(src+j),_mm256_loadu_pd(dst+j)));
        for (; j < len; ++j) dst[j] -= alpha*src[j];
        return;
    } else if constexpr (std::is_same_v<Scalar, float>) {
        const __m256 va = _mm256_set1_ps(alpha);
        Index j = 0;
        for (; j + 31 < len; j += 32) {
            _mm256_storeu_ps(dst+j,    _mm256_fnmadd_ps(va,_mm256_loadu_ps(src+j),    _mm256_loadu_ps(dst+j)));
            _mm256_storeu_ps(dst+j+8,  _mm256_fnmadd_ps(va,_mm256_loadu_ps(src+j+8),  _mm256_loadu_ps(dst+j+8)));
            _mm256_storeu_ps(dst+j+16, _mm256_fnmadd_ps(va,_mm256_loadu_ps(src+j+16), _mm256_loadu_ps(dst+j+16)));
            _mm256_storeu_ps(dst+j+24, _mm256_fnmadd_ps(va,_mm256_loadu_ps(src+j+24), _mm256_loadu_ps(dst+j+24)));
        }
        for (; j + 7 < len; j += 8)
            _mm256_storeu_ps(dst+j, _mm256_fnmadd_ps(va,_mm256_loadu_ps(src+j),_mm256_loadu_ps(dst+j)));
        for (; j < len; ++j) dst[j] -= alpha*src[j];
        return;
    }
#elif defined(__ARM_NEON)
    if constexpr (std::is_same_v<Scalar, double>) {
        const float64x2_t va = vdupq_n_f64(alpha);
        Index j = 0;
        for (; j + 7 < len; j += 8) {
            __builtin_prefetch(src+j+32, 0, 1);
            vst1q_f64(dst+j,   vfmsq_f64(vld1q_f64(dst+j),   va, vld1q_f64(src+j)));
            vst1q_f64(dst+j+2, vfmsq_f64(vld1q_f64(dst+j+2), va, vld1q_f64(src+j+2)));
            vst1q_f64(dst+j+4, vfmsq_f64(vld1q_f64(dst+j+4), va, vld1q_f64(src+j+4)));
            vst1q_f64(dst+j+6, vfmsq_f64(vld1q_f64(dst+j+6), va, vld1q_f64(src+j+6)));
        }
        for (; j + 1 < len; j += 2)
            vst1q_f64(dst+j, vfmsq_f64(vld1q_f64(dst+j), va, vld1q_f64(src+j)));
        for (; j < len; ++j) dst[j] -= alpha*src[j];
        return;
    } else if constexpr (std::is_same_v<Scalar, float>) {
        const float32x4_t va = vdupq_n_f32(alpha);
        Index j = 0;
        for (; j + 15 < len; j += 16) {
            __builtin_prefetch(src+j+64, 0, 1);
            vst1q_f32(dst+j,    vfmsq_f32(vld1q_f32(dst+j),    va, vld1q_f32(src+j)));
            vst1q_f32(dst+j+4,  vfmsq_f32(vld1q_f32(dst+j+4),  va, vld1q_f32(src+j+4)));
            vst1q_f32(dst+j+8,  vfmsq_f32(vld1q_f32(dst+j+8),  va, vld1q_f32(src+j+8)));
            vst1q_f32(dst+j+12, vfmsq_f32(vld1q_f32(dst+j+12), va, vld1q_f32(src+j+12)));
        }
        for (; j + 3 < len; j += 4)
            vst1q_f32(dst+j, vfmsq_f32(vld1q_f32(dst+j), va, vld1q_f32(src+j)));
        for (; j < len; ++j) dst[j] -= alpha*src[j];
        return;
    }
#endif
    #pragma omp simd
    for (Index j = 0; j < len; ++j) dst[j] -= alpha*src[j];
}

// fused_row_dot: w[j] += dot(row[0..pr), Y[:,j]) for ALL j in one pass over row.
// Y_col is col-major: Y[:,j] = Y_col + j*panel_rows  (contiguous, length panel_rows).
// Y_tr  is row-major transposed copy: Y_tr[j*panel_rows + r] = Y_col[j*panel_rows + r]
//   → same data, but when we iterate j in inner loop, Y_tr[j*panel_rows + r] for
//     fixed r is stride panel_rows (non-sequential). So we keep the outer-r / inner-j
//     structure and rely on compiler SIMD on the j loop (nb≤64 fits in registers).
// With -O3 -march=native the #pragma omp simd on j is auto-vectorised to 8 zmm FMAs.
template<typename Scalar>
[[gnu::always_inline]] inline
void fused_row_dot(const Scalar* __restrict__ row,
                   const Scalar* __restrict__ Y_col,
                   Scalar* __restrict__       w,
                   Index panel_rows, Index nb) noexcept
{
#if defined(__AVX512F__)
    if constexpr (std::is_same_v<Scalar, double>) {
        for (Index r = 0; r < panel_rows; ++r) {
            const Scalar rv = row[r];
            if (rv == Scalar{0}) continue;
            const __m512d vrv = _mm512_set1_pd(rv);
            Index j = 0;
            for (; j + 7 < nb; j += 8) {
                __m512d yw; double* ywp = reinterpret_cast<double*>(&yw);
                for (int jj = 0; jj < 8; ++jj) ywp[jj] = Y_col[(j+jj)*panel_rows + r];
                __m512d wv = _mm512_loadu_pd(w + j);
                _mm512_storeu_pd(w + j, _mm512_fmadd_pd(vrv, yw, wv));
            }
            for (; j < nb; ++j) w[j] += rv * Y_col[j * panel_rows + r];
        }
        return;
    }
#elif defined(__ARM_NEON)
    if constexpr (std::is_same_v<Scalar, double>) {
        const Index nb4 = nb & ~Index{3};
        Index r = 0;
        for (; r + 1 < panel_rows; r += 2) {
            const float64x2_t rr = vld1q_f64(row + r);
            Index j = 0;
            for (; j < nb4; j += 4) {
                w[j]   += vaddvq_f64(vmulq_f64(rr, vld1q_f64(Y_col + j*panel_rows + r)));
                w[j+1] += vaddvq_f64(vmulq_f64(rr, vld1q_f64(Y_col + (j+1)*panel_rows + r)));
                w[j+2] += vaddvq_f64(vmulq_f64(rr, vld1q_f64(Y_col + (j+2)*panel_rows + r)));
                w[j+3] += vaddvq_f64(vmulq_f64(rr, vld1q_f64(Y_col + (j+3)*panel_rows + r)));
            }
            for (; j < nb; ++j)
                w[j] += vaddvq_f64(vmulq_f64(rr, vld1q_f64(Y_col + j*panel_rows + r)));
        }

        for (; r < panel_rows; ++r) {
            const Scalar rv = row[r];
            if (rv == Scalar{0}) continue;
            for (Index j = 0; j < nb; ++j)
                w[j] += rv * Y_col[j * panel_rows + r];
        }
        return;
    } else if constexpr (std::is_same_v<Scalar, float>) {
        const Index nb4 = nb & ~Index{3};
        Index r = 0;
        for (; r + 3 < panel_rows; r += 4) {
            const float32x4_t rr = vld1q_f32(row + r);
            Index j = 0;
            for (; j < nb4; j += 4) {
                w[j]   += vaddvq_f32(vmulq_f32(rr, vld1q_f32(Y_col + j*panel_rows + r)));
                w[j+1] += vaddvq_f32(vmulq_f32(rr, vld1q_f32(Y_col + (j+1)*panel_rows + r)));
                w[j+2] += vaddvq_f32(vmulq_f32(rr, vld1q_f32(Y_col + (j+2)*panel_rows + r)));
                w[j+3] += vaddvq_f32(vmulq_f32(rr, vld1q_f32(Y_col + (j+3)*panel_rows + r)));
            }
            for (; j < nb; ++j)
                w[j] += vaddvq_f32(vmulq_f32(rr, vld1q_f32(Y_col + j*panel_rows + r)));
        }
        for (; r < panel_rows; ++r) {
            const Scalar rv = row[r];
            if (rv == Scalar{0}) continue;
            for (Index j = 0; j < nb; ++j)
                w[j] += rv * Y_col[j * panel_rows + r];
        }
        return;
    }
#endif
    for (Index r = 0; r < panel_rows; ++r) {
        const Scalar rv = row[r];
        if (rv == Scalar{0}) continue;
        #pragma omp simd
        for (Index j = 0; j < nb; ++j)
            w[j] += rv * Y_col[j * panel_rows + r];
    }
}

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
            const Scalar zj = z[j];
            #pragma omp simd
            for (Index r = 0; r <= j; ++r)
                Tz[r] += T_data[r + j * nb] * zj;
        }
        #pragma omp simd
        for (Index r = 0; r < i; ++r)
            T_data[r + i * nb] = -taus[i] * Tz[r];
    }
}

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
    Scalar* __restrict__ W_buf,
    Scalar* __restrict__ thread_W_base,
    int nthreads,
    Index max_dim) noexcept
{
    const Index n     = A.cols();
    const Index ncols = n - col0;
    const Index nrows = panel_rows;

    if (ncols <= 0 || nrows <= 0 || nb == 0) return;

    Scalar* __restrict__ Adata = A.data();
    const Index lda = n;

    const bool do_par = (nrows >= OMP_ROWS_THRESHOLD)
                     && (nrows * ncols >= OMP_WORK_THRESHOLD);

    std::fill(W_buf, W_buf + nb * ncols, Scalar{0});

    // Step 1: W = Y^T * A_sub
    #ifdef _OPENMP
    if (do_par) {
        #pragma omp parallel
        {
            const int tid  = omp_get_thread_num();
            const int nthr = omp_get_num_threads();
            Scalar* tw = thread_W_base + static_cast<std::size_t>(tid) * nb * max_dim;
            std::fill(tw, tw + nb * ncols, Scalar{0});

            #pragma omp for schedule(static) nowait
            for (Index r = 0; r < nrows; ++r) {
                const Scalar* __restrict__ arow = Adata + (row0 + r) * lda + col0;
                for (Index i = 0; i < nb; ++i) {
                    const Scalar yir = Y_col[r + i * panel_rows];
                    if (yir == Scalar{0}) continue;
                    simd_axpy(tw + i * ncols, arow, yir, ncols);
                }
            }

            const Index cols_per = (ncols + nthr - 1) / nthr;
            const Index c0   = std::min(static_cast<Index>(tid) * cols_per, ncols);
            const Index c1   = std::min(c0 + cols_per, ncols);
            const Index clen = c1 - c0;

            #pragma omp barrier

            if (clen > 0) {
                for (int t = 0; t < nthr; ++t) {
                    const Scalar* src = thread_W_base + static_cast<std::size_t>(t) * nb * max_dim;
                    for (Index i = 0; i < nb; ++i)
                        simd_axpy(W_buf + i * ncols + c0, src + i * ncols + c0, Scalar{1}, clen);
                }
            }
            #pragma omp barrier
        }
    } else
    #endif
    {
        // Serial path — parallelise Step 1 over rows of A (nrows independent outputs
        // would race on wi, so instead: parallel over i with each thread owning wi)
        // Actually rows of A are read-only in Step1; wi = W_buf+i*ncols are distinct.
        // Safe to parallelise over i because each i writes its own wi row of W_buf.
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(nrows * ncols >= 4096)
        #endif
        for (Index i = 0; i < nb; ++i) {
            const Scalar* __restrict__ yi = Y_col + i * panel_rows;
            Scalar* __restrict__       wi = W_buf + i * ncols;
            for (Index r = 0; r < nrows; ++r) {
                const Scalar yir = yi[r];
                if (yir == Scalar{0}) continue;
                simd_axpy(wi, Adata + (row0 + r) * lda + col0, yir, ncols);
            }
        }
    }

    // Step 2: W <- T^T * W  (serial dependency chain between i, nb×nb tiny)
    for (Index i = nb; i-- > 0;) {
        Scalar* __restrict__ wi = W_buf + i * ncols;
        const Scalar Tii = T_data[i + i * nb];
        #pragma omp simd
        for (Index j = 0; j < ncols; ++j) wi[j] *= Tii;
        for (Index l = 0; l < i; ++l) {
            const Scalar Tli = T_data[l + i * nb];
            if (Tli == Scalar{0}) continue;
            simd_axpy(wi, W_buf + l * ncols, Tli, ncols);
        }
    }

    // Step 3: A_sub -= Y * W
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(do_par)
    #endif
    for (Index r = 0; r < nrows; ++r) {
        Scalar* __restrict__ arow = Adata + (row0 + r) * lda + col0;
        for (Index i = 0; i < nb; ++i) {
            const Scalar yir = Y_col[r + i * panel_rows];
            if (yir == Scalar{0}) continue;
            simd_naxpy(arow, W_buf + i * ncols, yir, ncols);
        }
    }
}

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
    const Index m   = Q.rows();
    const Index lda = m;

    if (panel_rows <= 0 || nb == 0) return;

    Scalar* __restrict__ Qdata = Q.data();

    const bool do_par = (m >= OMP_ROWS_THRESHOLD)
                     && (m * panel_rows >= OMP_WORK_THRESHOLD);

    // Step 1 + 2 + 3 fused per Q-row — each row fully independent
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(do_par)
    #endif
    for (Index i = 0; i < m; ++i) {
        Scalar* __restrict__ qrow = Qdata + i * lda + row0;  // non-const: Step 3 writes here
        Scalar* __restrict__ wrow = W_buf + i * nb;

        // Step 1: w = Q[i, row0:] * Y  (fused single pass)
        for (Index j = 0; j < nb; ++j) wrow[j] = Scalar{0};
        fused_row_dot(qrow, Y_col, wrow, panel_rows, nb);

        // Step 2: w <- w * T^T  (nb×nb upper triangular, in-place per row)
        for (Index j = nb; j-- > 0;) {
            Scalar s = T_data[j + j * nb] * wrow[j];
            #pragma omp simd reduction(+:s)
            for (Index l = 0; l < j; ++l)
                s += T_data[l + j * nb] * wrow[l];
            wrow[j] = s;
        }

        // Step 3: Q[i, row0:] -= w * Y^T
        for (Index j = 0; j < nb; ++j) {
            const Scalar wij = wrow[j];
            if (wij == Scalar{0}) continue;
            simd_naxpy(qrow, Y_col + j * panel_rows, wij, panel_rows);
        }
    }
}

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

    Scalar* __restrict__ Rdata = R.data();

    for (Index j = 0; j < panel_cols; ++j) {
        const Index col = k + j;
        const Index len = m - col;

        Scalar tau_j{0};
        {
            const Scalar* src = Rdata + col * n + col;
            Scalar sigma{0};
            for (Index i = 1; i < len; ++i) {
                v_buf[i] = src[i * n];
                sigma   += v_buf[i] * v_buf[i];
            }
            const Scalar x0 = src[0];
            v_buf[0] = x0;

            if (sigma == Scalar{0} && x0 >= Scalar{0}) {
                taus[j] = Scalar{0};
                Scalar* yj = Y_col + j * panel_rows;
                std::fill(yj, yj + panel_rows, Scalar{0});
                continue;
            }

            const Scalar norm = std::sqrt(x0*x0 + sigma);
            const Scalar beta = (x0 <= Scalar{0}) ? norm : -norm;
            const Scalar inv  = Scalar{1} / (x0 - beta);
            v_buf[0] = Scalar{1};
            for (Index i = 1; i < len; ++i) v_buf[i] *= inv;
            tau_j = (beta - x0) / beta;
        }
        taus[j] = tau_j;

        Scalar* __restrict__ yj = Y_col + j * panel_rows;
        std::fill(yj, yj + panel_rows, Scalar{0});
        yj[j] = Scalar{1};
        for (Index i = 1; i < len; ++i) yj[j + i] = v_buf[i];

        {
            Scalar vtx{0};
            #pragma omp simd reduction(+:vtx)
            for (Index i = 0; i < len; ++i)
                vtx += v_buf[i] * Rdata[(col+i)*n + col];
            const Scalar tau_vtx = tau_j * vtx;
            #pragma omp simd
            for (Index i = 0; i < len; ++i)
                Rdata[(col+i)*n + col] -= tau_vtx * v_buf[i];
        }
        #pragma omp simd
        for (Index i = 1; i < len; ++i) Rdata[(col+i)*n + col] = Scalar{0};

        const Index trail_start = col + 1;
        const Index trail_end   = std::min(k + nb, n);
        const Index trail_width = trail_end - trail_start;
        if (trail_width <= 0) continue;

        Scalar w[QR_BLOCK] = {};
        for (Index i = 0; i < len; ++i) {
            const Scalar vi = v_buf[i];
            if (vi == Scalar{0}) continue;
            simd_axpy(w, Rdata + (col+i)*n + trail_start, vi, trail_width);
        }
        for (Index i = 0; i < len; ++i) {
            const Scalar fac = tau_j * v_buf[i];
            if (fac == Scalar{0}) continue;
            simd_naxpy(Rdata + (col+i)*n + trail_start, w, fac, trail_width);
        }
    }
}

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

    const Index max_dim = std::max(m, n);
    const Index steps   = std::min(m, n);

    for (Index k = 0; k < steps; k += QR_BLOCK) {
        const Index nb         = std::min(static_cast<Index>(QR_BLOCK), steps - k);
        const Index panel_rows = m - k;

        Scalar* Y_col    = ws.Y_col.data();
        Scalar* taus     = ws.taus.data();
        Scalar* T_data   = ws.T_mat.data();
        Scalar* W_buf    = ws.W.data();
        Scalar* W_Q      = ws.W_Q.data();
        Scalar* v_buf    = ws.v_buf.data();
        Scalar* thread_W = ws.thread_W.data();

        std::fill(Y_col,  Y_col  + panel_rows * nb, Scalar{0});
        std::fill(taus,   taus   + nb,               Scalar{0});
        std::fill(T_data, T_data + nb * nb,           Scalar{0});

        panel_qr_step(R, k, nb, Y_col, taus, v_buf);
        build_T_matrix(Y_col, panel_rows, nb, taus, T_data);

        if (k + nb < n)
            apply_block_reflector(R, Y_col, T_data, panel_rows, nb, k, k+nb,
                                  W_buf, thread_W, ws.nthreads, max_dim);

        apply_block_reflector_right(Q, Y_col, T_data, panel_rows, nb, k, W_Q);
    }

    #pragma omp parallel for schedule(static) if(m * n > 32768)
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
