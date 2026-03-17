#ifndef PARALLEL_LINEAR_ALGEBRA_LIB_LU_IMPL_H
#define PARALLEL_LINEAR_ALGEBRA_LIB_LU_IMPL_H

#include <vector>
#include <cmath>
#include <stdexcept>

#include "pla/types/index.h"

namespace pla {

template<typename Scalar>
    requires Numeric<Scalar>
LUResult<Scalar> lu_naive(const Matrix<Scalar>& input) {
    if (!input.is_square())
        throw std::invalid_argument("LU: matrix must be square");

    Index n = input.rows();
    Matrix<Scalar> A = input;
    std::vector<Index> perm(n);
    for (Index i = 0; i < n; i++) perm[i] = i;

    for (Index k = 0; k < n; k++) {
        Scalar pivot = A(k, k);
        if (std::abs(pivot) < 1e-12)
            throw std::runtime_error("LU naive: zero pivot");

        for (Index i = k + 1; i < n; i++) {
            A(i, k) /= pivot;
            for (Index j = k + 1; j < n; j++)
                A(i, j) -= A(i, k) * A(k, j);
        }
    }

    Matrix<Scalar> L = Matrix<Scalar>::identity(n);
    Matrix<Scalar> U(n, n, Scalar{0});

    for (Index i = 0; i < n; i++)
        for (Index j = 0; j < n; j++) {
            if (i > j) L(i, j) = A(i, j);
            else       U(i, j) = A(i, j);
        }

    return {L, U, perm};
}

template<typename Scalar>
    requires Numeric<Scalar>
static Matrix<Scalar> extract_block(const Matrix<Scalar>& A, Index row_start, Index row_end,
                                            Index col_start, Index col_end) {
    Index rows = row_end - row_start;
    Index cols = col_end - col_start;
    Matrix block{rows, cols, 0.0};

    for (Index i = 0; i < rows; i++)
        for (Index j = 0; j < cols; j++)
            block(i, j) = A(row_start + i, col_start + j);

    return block;
}

template<typename Scalar>
    requires Numeric<Scalar>
static void subtract_block(Matrix<Scalar>& A, const Matrix<Scalar>& block,
                                              Index row_start, Index col_start) {
    for (Index i = 0; i < block.rows(); i++)
        for (Index j = 0; j < block.cols(); j++)
            A(row_start + i, col_start + j) -= block(i, j);
}


template<typename Scalar>
    requires Numeric<Scalar>
static void lu_diagonal_block(Matrix<Scalar>& A, Index block_row,
                               Index block_col, Index block_size) {

    Index n = std::min(block_size, std::min(A.rows() - block_row, A.cols() - block_col));

    for (Index k = 0; k < n; k++) {
        Index r = block_row + k;
        Index c = block_col + k;

        Scalar pivot = A(r, c);
        if (std::abs(pivot) < 1e-12)
            throw std::runtime_error("LU: zero pivot");

        for (Index i = r + 1; i < block_row + n; i++) {
            A(i, c) /= pivot;
            for (Index j = c + 1; j < block_col + n; j++)
                A(i, j) -= A(i, c) * A(r, j);
        }
    }
}

// L00 * U0i = A0i
// (L00_r * U0i_j)_{rj} = A_{rj}
// L00 * x_k = Ai_k(columns)
// => U(r,j) + sum_{i=block_row..r-1} L(r,i) * U(i,j) = A(r,j)
// => U(r,j) = A(r,j) - sum_{i<r} L(r,i) * U(i,j)
template<typename Scalar>
    requires Numeric<Scalar>
static void update_right(Matrix<Scalar>& A, Index block_row, Index block_col, Index block_size,
                                         Index col_start, Index col_end) {

    for (Index k = 0; k < block_size; k++) {
        Index r = block_row + k;

        for (Index j = col_start; j < col_end; j++) {
            for (Index i = block_row; i < r; i++)
                A(r, j) -= A(r, i) * A(i, j);
        }
    }
}

// Li0 * U00 = Ai0
// (Li0_i * U00_k)_{ik} = A_{ik}
// x_r * U00 = Ai0_r (rows)
// => L(i,k) * U(k,k) + sum_{j=k+1..block_size-1} L(i,j) * U(j,k) = A(i,k)
// => L(i,k) = (A(i,k) - sum_{j>k} L(i,j) * U(j,k)) / U(k,k)
template<typename Scalar>
    requires Numeric<Scalar>
static void update_bottom(Matrix<Scalar>& A, Index block_row, Index block_col, Index block_size,
                                                    Index row_start, Index row_end) {
    for (Index i = row_start; i < row_end; i++) {
        for (Index k = 0; k < block_size; k++) {

            Index c = block_col + k;
            A(i, c) /= A(c, c);

            for (Index kk = k + 1; kk < block_size; kk++)
                A(i, block_col + kk) -= A(i, c) * A(c, block_col + kk);
        }
    }
}

// A_sub -= L_bottom * U_right
template<typename Scalar>
    requires Numeric<Scalar>
static void update_submatrix(Matrix<Scalar>& A, Index row_start, Index row_end,
                                     Index col_start, Index col_end,
                                               Index k_start,   Index k_end) {

    Matrix<Scalar> L_bottom = extract_block(A, row_start, row_end, k_start, k_end);
    Matrix<Scalar> U_right  = extract_block(A, k_start,   k_end,  col_start, col_end);

    Matrix<Scalar> product = L_bottom * U_right;

    // A_sub -= L_bottom * U_right
    subtract_block(A, product, row_start, col_start);
}


template<typename Scalar>
    requires Numeric<Scalar>
LUResult<Scalar> lu_blocked(const Matrix<Scalar>& input, Index block_size) {
    if (!input.is_square())
        throw std::invalid_argument("LU: matrix must be square");

    Index n = input.rows();
    Matrix A = input;

    std::vector<Index> perm(n);
    for (Index i = 0; i < n; i++) perm[i] = i;

    for (Index k = 0; k < n; k += block_size) {
        Index k_end = std::min(k + block_size, n);
        Index bs    = k_end - k;

        lu_diagonal_block(A, k, k, bs);

        if (k_end >= n) continue;

        #pragma omp parallel for schedule(static)
        for (Index j = k_end; j < n; j += block_size) {
            Index j_end = std::min(j + block_size, n);
            update_right(A, k, k, bs, j, j_end);
        }

        #pragma omp parallel for schedule(static)
        for (Index i = k_end; i < n; i += block_size) {
            Index i_end = std::min(i + block_size, n);
            update_bottom(A, k, k, bs, i, i_end);
        }

        #pragma omp parallel for schedule(dynamic) collapse(2)
        for (Index i = k_end; i < n; i += block_size) {
            for (Index j = k_end; j < n; j += block_size) {
                Index i_end = std::min(i + block_size, n);
                Index j_end = std::min(j + block_size, n);
                update_submatrix(A, i, i_end, j, j_end, k, k_end);
            }
        }
    }

    Matrix<Scalar> L = Matrix<Scalar>::identity(n);
    Matrix<Scalar> U(n, n, 0.0);

    for (Index i = 0; i < n; i++) {
        for (Index j = 0; j < n; j++) {
            if (i > j)
                L(i, j) = A(i, j);  // L
            else
                U(i, j) = A(i, j);  // U
        }
    }

    return {L, U, perm};
}

}

#endif //PARALLEL_LINEAR_ALGEBRA_LIB_LU_IMPL_H