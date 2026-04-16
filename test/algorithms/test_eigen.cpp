#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <vector>
#include <cmath>

#include "pla/pla.h"

namespace {

template<typename Scalar>
bool near(Scalar a, Scalar b, Scalar eps = static_cast<Scalar>(1e-6)) {
    return std::abs(a - b) <= eps;
}

template<typename Scalar>
bool contains_eigenvalue(
    const std::vector<std::complex<Scalar>>& values,
    std::complex<Scalar> target,
    Scalar eps = static_cast<Scalar>(1e-6)
) {
    return std::any_of(values.begin(), values.end(), [&](const auto& value) {
        return std::abs(value - target) <= eps;
    });
}

template<typename Scalar>
Scalar residual_norm(
    const pla::Matrix<Scalar>& A,
    const pla::Vector<Scalar>& v,
    Scalar lambda
) {
    pla::Vector<Scalar> Av = A * v;
    pla::Vector<Scalar> lv = v * lambda;
    return (Av - lv).norm();
}

} // namespace

TEST(EigenGeneralTest, DiagonalMatrixEigenvalues) {
    pla::Matrix<double> A(3, 3);
    A(0, 0) = 1.0;
    A(1, 1) = 2.0;
    A(2, 2) = 3.0;

    auto result = pla::eigenvalues_general(A);

    ASSERT_EQ(result.values.size(), 3);

    EXPECT_TRUE(contains_eigenvalue(result.values, {1.0, 0.0}));
    EXPECT_TRUE(contains_eigenvalue(result.values, {2.0, 0.0}));
    EXPECT_TRUE(contains_eigenvalue(result.values, {3.0, 0.0}));
}

TEST(EigenGeneralTest, UpperTriangularMatrixEigenvalues) {
    pla::Matrix<double> A(3, 3);
    A(0, 0) = 4.0;
    A(0, 1) = 7.0;
    A(0, 2) = -2.0;

    A(1, 1) = -1.0;
    A(1, 2) = 5.0;

    A(2, 2) = 9.0;

    auto result = pla::eigenvalues_general(A);

    ASSERT_EQ(result.values.size(), 3);

    EXPECT_TRUE(contains_eigenvalue(result.values, {4.0, 0.0}));
    EXPECT_TRUE(contains_eigenvalue(result.values, {-1.0, 0.0}));
    EXPECT_TRUE(contains_eigenvalue(result.values, {9.0, 0.0}));
}

TEST(EigenGeneralTest, ComplexConjugatePairEigenvalues) {
    pla::Matrix<double> A(2, 2);
    A(0, 0) = 0.0;
    A(0, 1) = -1.0;
    A(1, 0) = 1.0;
    A(1, 1) = 0.0;

    auto result = pla::eigenvalues_general(A);

    ASSERT_EQ(result.values.size(), 2);

    EXPECT_TRUE(contains_eigenvalue(result.values, {0.0, 1.0}));
    EXPECT_TRUE(contains_eigenvalue(result.values, {0.0, -1.0}));
}

TEST(EigenGeneralTest, RealEigenvectorsSatisfyAvEqualsLambdaV) {
    pla::Matrix<double> A(2, 2);
    A(0, 0) = 2.0;
    A(0, 1) = 0.0;
    A(1, 0) = 0.0;
    A(1, 1) = 3.0;

    auto result = pla::eigen_general(A);

    ASSERT_EQ(result.values.size(), 2);
    ASSERT_EQ(result.has_real_eigenvector.size(), 2);

    for (std::size_t col = 0; col < result.values.size(); ++col) {
        ASSERT_TRUE(result.has_real_eigenvector[col]);

        const double lambda = result.values[col].real();

        pla::Vector<double> v(2);
        v[0] = result.real_eigenvectors(0, col);
        v[1] = result.real_eigenvectors(1, col);

        EXPECT_GT(v.norm(), 1e-9);
        EXPECT_LT(residual_norm(A, v, lambda), 1e-6);
    }
}

TEST(EigenGeneralTest, ComplexEigenvaluesDoNotProduceRealEigenvectors) {
    pla::Matrix<double> A(2, 2);
    A(0, 0) = 0.0;
    A(0, 1) = -1.0;
    A(1, 0) = 1.0;
    A(1, 1) = 0.0;

    auto result = pla::eigen_general(A);

    ASSERT_EQ(result.values.size(), 2);
    ASSERT_EQ(result.has_real_eigenvector.size(), 2);

    EXPECT_FALSE(result.has_real_eigenvector[0]);
    EXPECT_FALSE(result.has_real_eigenvector[1]);
}

TEST(EigenGeneralTest, RejectsNonSquareMatrix) {
    pla::Matrix<double> A(2, 3);

    EXPECT_THROW(
        pla::eigenvalues_general(A),
        pla::NonSquareMatrixException
    );
}