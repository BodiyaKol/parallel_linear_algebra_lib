#ifndef PARALLEL_LINEAR_ALGEBRA_LIB_TEST_MATRIX_OPR_H
#define PARALLEL_LINEAR_ALGEBRA_LIB_TEST_MATRIX_OPR_H

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <sstream>
#include <type_traits>

#include "pla.h"


template<typename T>
static pla::Matrix<T> make_matrix(int rows, int cols,
                                  std::initializer_list<T> values,
                                  pla::StorageOrder order = pla::StorageOrder::RowMajor)
{
    pla::Matrix<T> m(rows, cols, order);
    auto it = values.begin();
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            if (it != values.end())
                m(i, j) = *it++;
    return m;
}

template<typename T>
static bool matrices_near(const pla::Matrix<T>& A,
                           const pla::Matrix<T>& B,
                           T tol = T(1e-9))
{
    if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            if (std::abs(A(i, j) - B(i, j)) > tol) return false;
    return true;
}


template<typename T>
class MatrixTest : public ::testing::Test {};

using ScalarTypes = ::testing::Types<double, float>;
TYPED_TEST_SUITE(MatrixTest, ScalarTypes);

TYPED_TEST(MatrixTest, DefaultConstructor)
{
    pla::Matrix<TypeParam> m;
    EXPECT_EQ(m.rows(), 0);
    EXPECT_EQ(m.cols(), 0);
    EXPECT_EQ(m.size(), 0);
    EXPECT_EQ(m.data(), nullptr);
}

TYPED_TEST(MatrixTest, SizeConstructorZeroInitialised)
{
    pla::Matrix<TypeParam> m(3, 4);
    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 4);
    EXPECT_EQ(m.size(), 12);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            EXPECT_EQ(m(i, j), TypeParam(0));
}

TYPED_TEST(MatrixTest, FillConstructor)
{
    pla::Matrix<TypeParam> m(2, 3, TypeParam(7));
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_EQ(m(i, j), TypeParam(7));
}

TYPED_TEST(MatrixTest, StorageOrderRowMajor)
{
    pla::Matrix<TypeParam> m(2, 3, pla::StorageOrder::RowMajor);
    EXPECT_EQ(m.order(), pla::StorageOrder::RowMajor);
    m(0, 0) = 1; m(0, 1) = 2; m(0, 2) = 3;
    m(1, 0) = 4; m(1, 1) = 5; m(1, 2) = 6;
    // In row-major the raw buffer must be [ 1 2 3 4 5 6 ]
    const TypeParam* p = m.data();
    EXPECT_EQ(p[0], TypeParam(1));
    EXPECT_EQ(p[1], TypeParam(2));
    EXPECT_EQ(p[3], TypeParam(4));
}

TYPED_TEST(MatrixTest, StorageOrderColMajor)
{
    pla::Matrix<TypeParam> m(2, 3, pla::StorageOrder::ColMajor);
    EXPECT_EQ(m.order(), pla::StorageOrder::ColMajor);
    m(0, 0) = 1; m(0, 1) = 2; m(0, 2) = 3;
    m(1, 0) = 4; m(1, 1) = 5; m(1, 2) = 6;
    // In col-major the raw buffer must be [ 1 4 2 5 3 6 ]
    const TypeParam* p = m.data();
    EXPECT_EQ(p[0], TypeParam(1));
    EXPECT_EQ(p[1], TypeParam(4));
    EXPECT_EQ(p[2], TypeParam(2));
}

TYPED_TEST(MatrixTest, IsSquare)
{
    pla::Matrix<TypeParam> sq(4, 4);
    pla::Matrix<TypeParam> rect(3, 4);
    EXPECT_TRUE(sq.is_square());
    EXPECT_FALSE(rect.is_square());
}

TYPED_TEST(MatrixTest, CopyConstructorDeepCopy)
{
    auto A = make_matrix<TypeParam>(2, 2, {1, 2, 3, 4});
    pla::Matrix<TypeParam> B(A);

    EXPECT_TRUE(matrices_near(A, B));

    B(0, 0) = TypeParam(99);
    EXPECT_EQ(A(0, 0), TypeParam(1));
}

TYPED_TEST(MatrixTest, CopyAssignmentDeepCopy)
{
    auto A = make_matrix<TypeParam>(2, 2, {1, 2, 3, 4});
    pla::Matrix<TypeParam> B(2, 2);
    B = A;

    EXPECT_TRUE(matrices_near(A, B));
    B(1, 1) = TypeParam(42);
    EXPECT_EQ(A(1, 1), TypeParam(4));
}

TYPED_TEST(MatrixTest, SelfAssignment)
{
    auto A = make_matrix<TypeParam>(2, 2, {1, 2, 3, 4});
    A = A; // must not crash or corrupt
    EXPECT_EQ(A(0, 0), TypeParam(1));
    EXPECT_EQ(A(1, 1), TypeParam(4));
}

TYPED_TEST(MatrixTest, MoveConstructor)
{
    auto A = make_matrix<TypeParam>(2, 2, {1, 2, 3, 4});
    pla::Matrix<TypeParam> B(std::move(A));

    EXPECT_EQ(B(0, 0), TypeParam(1));
    EXPECT_EQ(B(1, 1), TypeParam(4));
    // A is in valid-but-unspecified state; just don't crash
}

TYPED_TEST(MatrixTest, MoveAssignment)
{
    auto A = make_matrix<TypeParam>(2, 2, {5, 6, 7, 8});
    pla::Matrix<TypeParam> B;
    B = std::move(A);
    EXPECT_EQ(B(0, 0), TypeParam(5));
}

TYPED_TEST(MatrixTest, ElementAccessMutable)
{
    pla::Matrix<TypeParam> m(3, 3);
    m(1, 2) = TypeParam(3.14);
    EXPECT_NEAR(static_cast<double>(m(1, 2)), 3.14, 1e-5);
}

TYPED_TEST(MatrixTest, ElementAccessConst)
{
    const auto m = make_matrix<TypeParam>(2, 2, {10, 20, 30, 40});
    EXPECT_EQ(m(0, 1), TypeParam(20));
    EXPECT_EQ(m(1, 0), TypeParam(30));
}

TYPED_TEST(MatrixTest, AtValidIndex)
{
    auto m = make_matrix<TypeParam>(2, 2, {1, 2, 3, 4});
    EXPECT_NO_THROW((void)m.at(1, 1));
    EXPECT_EQ(m.at(0, 1), TypeParam(2));
}

TYPED_TEST(MatrixTest, AtInvalidIndexThrows)
{
    pla::Matrix<TypeParam> m(2, 2);
    EXPECT_THROW((void)m.at(2, 0), std::out_of_range);
    EXPECT_THROW((void)m.at(0, 2), std::out_of_range);
    EXPECT_THROW((void)m.at(-1, 0), std::out_of_range);
}

TYPED_TEST(MatrixTest, Clear)
{
    pla::Matrix<TypeParam> m(3, 3, TypeParam(1));
    m.clear();
    EXPECT_EQ(m.rows(), 0);
    EXPECT_EQ(m.cols(), 0);
    EXPECT_EQ(m.data(), nullptr);
}

TYPED_TEST(MatrixTest, MemberSwap)
{
    auto A = make_matrix<TypeParam>(2, 2, {1, 2, 3, 4});
    auto B = make_matrix<TypeParam>(3, 1, {9, 8, 7});

    A.swap(B);

    EXPECT_EQ(A.rows(), 3);
    EXPECT_EQ(A.cols(), 1);
    EXPECT_EQ(A(0, 0), TypeParam(9));

    EXPECT_EQ(B.rows(), 2);
    EXPECT_EQ(B(0, 0), TypeParam(1));
}

TYPED_TEST(MatrixTest, FreeSwap)
{
    auto A = make_matrix<TypeParam>(2, 2, {1, 2, 3, 4});
    auto B = make_matrix<TypeParam>(1, 3, {7, 8, 9});
    pla::swap(A, B);
    EXPECT_EQ(A.cols(), 3);
    EXPECT_EQ(B.rows(), 2);
}

TYPED_TEST(MatrixTest, Fill)
{
    pla::Matrix<TypeParam> m(3, 4);
    m.fill(TypeParam(5));
    for (int i = 0; i < m.rows(); ++i)
        for (int j = 0; j < m.cols(); ++j)
            EXPECT_EQ(m(i, j), TypeParam(5));
}

TYPED_TEST(MatrixTest, SetIdentitySquare)
{
    pla::Matrix<TypeParam> m(4, 4, TypeParam(99));
    m.set_identity();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            EXPECT_EQ(m(i, j), (i == j) ? TypeParam(1) : TypeParam(0));
}

TYPED_TEST(MatrixTest, StaticIdentity)
{
    auto I = pla::Matrix<TypeParam>::identity(3);
    EXPECT_EQ(I.rows(), 3);
    EXPECT_EQ(I.cols(), 3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_EQ(I(i, j), (i == j) ? TypeParam(1) : TypeParam(0));
}

TYPED_TEST(MatrixTest, Addition)
{
    auto A = make_matrix<TypeParam>(2, 2, {1, 2, 3, 4});
    auto B = make_matrix<TypeParam>(2, 2, {5, 6, 7, 8});
    auto C = A + B;
    EXPECT_EQ(C(0, 0), TypeParam(6));
    EXPECT_EQ(C(0, 1), TypeParam(8));
    EXPECT_EQ(C(1, 0), TypeParam(10));
    EXPECT_EQ(C(1, 1), TypeParam(12));
}

TYPED_TEST(MatrixTest, Subtraction)
{
    auto A = make_matrix<TypeParam>(2, 2, {5, 6, 7, 8});
    auto B = make_matrix<TypeParam>(2, 2, {1, 2, 3, 4});
    auto C = A - B;
    EXPECT_EQ(C(0, 0), TypeParam(4));
    EXPECT_EQ(C(1, 1), TypeParam(4));
}

TYPED_TEST(MatrixTest, UnaryNegation)
{
    auto A = make_matrix<TypeParam>(2, 2, {1, -2, 3, -4});
    auto B = -A;
    EXPECT_EQ(B(0, 0), TypeParam(-1));
    EXPECT_EQ(B(0, 1), TypeParam(2));
    EXPECT_EQ(B(1, 0), TypeParam(-3));
    EXPECT_EQ(B(1, 1), TypeParam(4));
}

TYPED_TEST(MatrixTest, ScalarMultiplyRight)
{
    auto A = make_matrix<TypeParam>(2, 2, {1, 2, 3, 4});
    auto B = A * TypeParam(3);
    EXPECT_EQ(B(0, 0), TypeParam(3));
    EXPECT_EQ(B(1, 1), TypeParam(12));
}

TYPED_TEST(MatrixTest, ScalarMultiplyLeft)
{
    auto A = make_matrix<TypeParam>(2, 2, {1, 2, 3, 4});
    auto B = TypeParam(2) * A;
    EXPECT_EQ(B(0, 1), TypeParam(4));
}

TYPED_TEST(MatrixTest, ScalarDivision)
{
    auto A = make_matrix<TypeParam>(2, 2, {2, 4, 6, 8});
    auto B = A / TypeParam(2);
    EXPECT_EQ(B(0, 0), TypeParam(1));
    EXPECT_EQ(B(1, 1), TypeParam(4));
}

TYPED_TEST(MatrixTest, CompoundAddAssign)
{
    auto A = make_matrix<TypeParam>(2, 2, {1, 2, 3, 4});
    auto B = make_matrix<TypeParam>(2, 2, {1, 1, 1, 1});
    A += B;
    EXPECT_EQ(A(0, 0), TypeParam(2));
    EXPECT_EQ(A(1, 1), TypeParam(5));
}

TYPED_TEST(MatrixTest, CompoundSubAssign)
{
    auto A = make_matrix<TypeParam>(2, 2, {5, 6, 7, 8});
    auto B = make_matrix<TypeParam>(2, 2, {1, 1, 1, 1});
    A -= B;
    EXPECT_EQ(A(0, 0), TypeParam(4));
    EXPECT_EQ(A(1, 1), TypeParam(7));
}

TYPED_TEST(MatrixTest, CompoundScalarMulAssign)
{
    auto A = make_matrix<TypeParam>(2, 2, {1, 2, 3, 4});
    A *= TypeParam(10);
    EXPECT_EQ(A(0, 0), TypeParam(10));
    EXPECT_EQ(A(1, 0), TypeParam(30));
}

TYPED_TEST(MatrixTest, CompoundScalarDivAssign)
{
    auto A = make_matrix<TypeParam>(2, 2, {10, 20, 30, 40});
    A /= TypeParam(10);
    EXPECT_EQ(A(0, 0), TypeParam(1));
    EXPECT_EQ(A(1, 1), TypeParam(4));
}

TYPED_TEST(MatrixTest, MatVecProduct)
{
    // [ 1 2 ]   [ 1 ]   [ 5 ]
    // [ 3 4 ] * [ 2 ] = [11 ]
    auto A = make_matrix<TypeParam>(2, 2, {1, 2, 3, 4});
    pla::Vector<TypeParam> v(2);
    v[0] = TypeParam(1);
    v[1] = TypeParam(2);

    auto r = A * v;
    EXPECT_NEAR(static_cast<double>(r[0]), 5.0, 1e-5);
    EXPECT_NEAR(static_cast<double>(r[1]), 11.0, 1e-5);
}

TYPED_TEST(MatrixTest, MatMatProduct2x2)
{
    // [1 2] * [5 6] = [19 22]
    // [3 4]   [7 8]   [43 50]
    auto A = make_matrix<TypeParam>(2, 2, {1, 2, 3, 4});
    auto B = make_matrix<TypeParam>(2, 2, {5, 6, 7, 8});
    auto C = A * B;

    EXPECT_NEAR(static_cast<double>(C(0, 0)), 19.0, 1e-4);
    EXPECT_NEAR(static_cast<double>(C(0, 1)), 22.0, 1e-4);
    EXPECT_NEAR(static_cast<double>(C(1, 0)), 43.0, 1e-4);
    EXPECT_NEAR(static_cast<double>(C(1, 1)), 50.0, 1e-4);
}

TYPED_TEST(MatrixTest, MatMatProductRectangular)
{
    // (2×3) * (3×2)
    auto A = make_matrix<TypeParam>(2, 3, {1, 2, 3,
                                           4, 5, 6});
    auto B = make_matrix<TypeParam>(3, 2, {7,  8,
                                           9,  10,
                                           11, 12});
    auto C = A * B;
    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 2);
    EXPECT_NEAR(static_cast<double>(C(0, 0)), 58.0, 1e-4);
    EXPECT_NEAR(static_cast<double>(C(0, 1)), 64.0, 1e-4);
    EXPECT_NEAR(static_cast<double>(C(1, 0)), 139.0, 1e-4);
    EXPECT_NEAR(static_cast<double>(C(1, 1)), 154.0, 1e-4);
}

TYPED_TEST(MatrixTest, MatMatProductWithIdentityLeft)
{
    auto I = pla::Matrix<TypeParam>::identity(3);
    auto A = make_matrix<TypeParam>(3, 3, {1, 2, 3,
                                           4, 5, 6,
                                           7, 8, 9});
    auto C = I * A;
    EXPECT_TRUE(matrices_near(C, A, TypeParam(1e-5)));
}

TYPED_TEST(MatrixTest, MatMatProductWithIdentityRight)
{
    auto A = make_matrix<TypeParam>(3, 3, {1, 2, 3,
                                           4, 5, 6,
                                           7, 8, 9});
    auto I = pla::Matrix<TypeParam>::identity(3);
    auto C = A * I;
    EXPECT_TRUE(matrices_near(C, A, TypeParam(1e-5)));
}

TYPED_TEST(MatrixTest, TransposeSquare)
{
    auto A = make_matrix<TypeParam>(2, 2, {1, 2, 3, 4});
    auto T = A.transpose();
    EXPECT_EQ(T(0, 1), TypeParam(3));
    EXPECT_EQ(T(1, 0), TypeParam(2));
}

TYPED_TEST(MatrixTest, TransposeRectangular)
{
    // (2×3)^T == (3×2)
    auto A = make_matrix<TypeParam>(2, 3, {1, 2, 3, 4, 5, 6});
    auto T = A.transpose();
    EXPECT_EQ(T.rows(), 3);
    EXPECT_EQ(T.cols(), 2);
    EXPECT_EQ(T(0, 0), TypeParam(1));
    EXPECT_EQ(T(1, 0), TypeParam(2));
    EXPECT_EQ(T(2, 1), TypeParam(6));
}

TYPED_TEST(MatrixTest, TransposeInplace)
{
    auto A = make_matrix<TypeParam>(2, 2, {1, 2, 3, 4});
    A.transpose_inplace();
    EXPECT_EQ(A(0, 1), TypeParam(3));
    EXPECT_EQ(A(1, 0), TypeParam(2));
}

TYPED_TEST(MatrixTest, DoubleTranposeIsOriginal)
{
    auto A = make_matrix<TypeParam>(3, 2, {1, 2, 3, 4, 5, 6});
    auto T = A.transpose().transpose();
    EXPECT_TRUE(matrices_near(A, T, TypeParam(1e-9)));
}

TYPED_TEST(MatrixTest, RowExtractor)
{
    auto A = make_matrix<TypeParam>(3, 3, {1, 2, 3,
                                           4, 5, 6,
                                           7, 8, 9});
    auto r = A.row(1);
    EXPECT_EQ(r[0], TypeParam(4));
    EXPECT_EQ(r[1], TypeParam(5));
    EXPECT_EQ(r[2], TypeParam(6));
}

TYPED_TEST(MatrixTest, ColExtractor)
{
    auto A = make_matrix<TypeParam>(3, 3, {1, 2, 3,
                                           4, 5, 6,
                                           7, 8, 9});
    auto c = A.col(2);
    EXPECT_EQ(c[0], TypeParam(3));
    EXPECT_EQ(c[1], TypeParam(6));
    EXPECT_EQ(c[2], TypeParam(9));
}

TYPED_TEST(MatrixTest, NormKnownValue)
{
    // ||[3, 4]|| == 5
    auto A = make_matrix<TypeParam>(1, 2, {3, 4});
    EXPECT_NEAR(static_cast<double>(A.norm()), 5.0, 1e-5);
}

TYPED_TEST(MatrixTest, NormZeroMatrix)
{
    pla::Matrix<TypeParam> Z(3, 3);
    EXPECT_NEAR(static_cast<double>(Z.norm()), 0.0, 1e-9);
}

TYPED_TEST(MatrixTest, NormIdentity)
{
    constexpr int n = 4;
    auto I = pla::Matrix<TypeParam>::identity(n);
    EXPECT_NEAR(static_cast<double>(I.norm()), std::sqrt(static_cast<double>(n)), 1e-5);
}

TYPED_TEST(MatrixTest, EqualityTrue)
{
    auto A = make_matrix<TypeParam>(2, 2, {1, 2, 3, 4});
    auto B = make_matrix<TypeParam>(2, 2, {1, 2, 3, 4});
    EXPECT_TRUE(A == B);
    EXPECT_FALSE(A != B);
}

TYPED_TEST(MatrixTest, EqualityFalse)
{
    auto A = make_matrix<TypeParam>(2, 2, {1, 2, 3, 4});
    auto B = make_matrix<TypeParam>(2, 2, {1, 2, 3, 5});
    EXPECT_FALSE(A == B);
    EXPECT_TRUE(A != B);
}

TYPED_TEST(MatrixTest, EqualityDifferentShape)
{
    pla::Matrix<TypeParam> A(2, 3);
    pla::Matrix<TypeParam> B(3, 2);
    EXPECT_FALSE(A == B);
    EXPECT_TRUE(A != B);
}

TYPED_TEST(MatrixTest, StreamOutputDoesNotCrash)
{
    auto A = make_matrix<TypeParam>(2, 2, {1, 2, 3, 4});
    std::ostringstream oss;
    EXPECT_NO_THROW(oss << A);
    EXPECT_FALSE(oss.str().empty());
}

TYPED_TEST(MatrixTest, OneByOne)
{
    pla::Matrix<TypeParam> m(1, 1, TypeParam(42));
    EXPECT_EQ(m(0, 0), TypeParam(42));
    auto T = m.transpose();
    EXPECT_EQ(T(0, 0), TypeParam(42));
    auto sq = m * m;
    EXPECT_NEAR(static_cast<double>(sq(0, 0)), 42.0 * 42.0, 1e-4);
}

TYPED_TEST(MatrixTest, AdditionCommutativity)
{
    auto A = make_matrix<TypeParam>(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto B = make_matrix<TypeParam>(3, 3, {9, 8, 7, 6, 5, 4, 3, 2, 1});
    EXPECT_TRUE(matrices_near<TypeParam>(A + B, B + A, TypeParam(1e-9)));
}

TYPED_TEST(MatrixTest, ScalarArithmeticRoundTrip)
{
    auto A = make_matrix<TypeParam>(2, 2, {4, 8, 12, 16});
    auto B = (A * TypeParam(3)) / TypeParam(3);
    EXPECT_TRUE(matrices_near(A, B, TypeParam(1e-5)));
}

TYPED_TEST(MatrixTest, TransposeOfProductEqualsProductOfTransposes)
{
    auto A = make_matrix<TypeParam>(2, 3, {1, 2, 3, 4, 5, 6});
    auto B = make_matrix<TypeParam>(3, 2, {1, 0, 0, 1, 1, 1});
    auto ABt = (A * B).transpose();
    auto BtAt = B.transpose() * A.transpose();
    EXPECT_TRUE(matrices_near(ABt, BtAt, TypeParam(1e-5)));
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#endif //PARALLEL_LINEAR_ALGEBRA_LIB_TEST_MATRIX_OPR_H