#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <vector>

#include "pla/pla.h"

static void fill_random_pla(pla::Matrix<>& M, std::mt19937& rng,
                             std::uniform_real_distribution<double>& dist) {
    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j)
            M(i, j) = dist(rng);
}

static void sync_to_eigen(const pla::Matrix<>& src, Eigen::MatrixXd& dst) {
    for (int i = 0; i < src.rows(); ++i)
        for (int j = 0; j < src.cols(); ++j)
            dst(i, j) = src(i, j);
}



TEST(Determinant, Identity) {
    for (int n : {1, 2, 4, 8, 16}) {
        pla::Matrix<> A = pla::Matrix<>::identity(n);
        EXPECT_NEAR(pla::determinant(A), 1.0, 1e-12) << "n=" << n;
    }
}

TEST(Determinant, KnownSmall2x2) {
    // det | 1 2 | = 1*4 - 2*3 = -2
    //     | 3 4 |
    pla::Matrix<> A(2, 2);
    A(0,0) = 1; A(0,1) = 2;
    A(1,0) = 3; A(1,1) = 4;
    EXPECT_NEAR(pla::determinant(A), -2.0, 1e-12);
}

TEST(Determinant, KnownSmall3x3) {
    // det | 2 -1  0 |
    //     | 1  3  2 | = 2*(3*1 - 2*0) - (-1)*(1*1 - 2*2) + 0 = 6 - 3 = 3
    //     | 2  0  1 |
    pla::Matrix<> A(3, 3);
    A(0,0) = 2; A(0,1) = -1; A(0,2) = 0;
    A(1,0) = 1; A(1,1) =  3; A(1,2) = 2;
    A(2,0) = 2; A(2,1) =  0; A(2,2) = 1;
    EXPECT_NEAR(pla::determinant(A), 3.0, 1e-10);
}

TEST(Determinant, DiagonalMatrix) {
    // det = product of diagonal elements
    const int n = 5;
    pla::Matrix<> A(n, n, 0.0);
    double expected = 1.0;
    for (int i = 0; i < n; ++i) {
        A(i, i) = static_cast<double>(i + 2);
        expected *= static_cast<double>(i + 2);
    }
    EXPECT_NEAR(pla::determinant(A), expected, 1e-9);
}

TEST(Determinant, ThrowsOnNonSquare) {
    pla::Matrix<> A(3, 4);
    EXPECT_THROW(pla::determinant(A), std::invalid_argument);
}

TEST(Determinant, MatchesEigen) {
    std::mt19937 rng(9999);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    for (int n : {2, 4, 8, 16, 32, 64}) {
        pla::Matrix<> A(n, n);
        fill_random_pla(A, rng, dist);

        Eigen::MatrixXd Ae(n, n);
        sync_to_eigen(A, Ae);

        double det_pla   = pla::determinant(A);
        double det_eigen = Ae.determinant();

        double tol = std::abs(det_eigen) * 1e-6;
        EXPECT_NEAR(det_pla, det_eigen, tol) << "n=" << n;
    }
}

TEST(Determinant, ScalarMultiply) {
    // det(k*A) = k^n * det(A)
    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> dist(-3.0, 3.0);

    const int n = 4;
    const double k = 2.0;

    pla::Matrix<> A(n, n);
    fill_random_pla(A, rng, dist);

    pla::Matrix<> kA = A * k;

    double det_A  = pla::determinant(A);
    double det_kA = pla::determinant(kA);

    double expected = std::pow(k, n) * det_A;
    EXPECT_NEAR(det_kA, expected, std::abs(expected) * 1e-9);
}


TEST(Determinant, PerformanceCompareEigen) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<int> sizes = {32, 64, 128, 256, 512};
    const int repeats = 4;

    for (int n : sizes) {
        pla::Matrix<> A(n, n);
        fill_random_pla(A, rng, dist);

        Eigen::MatrixXd Ae(n, n);
        sync_to_eigen(A, Ae);

        pla::determinant(A);
        Ae.determinant();

        double total_pla = 0.0;
        for (int r = 0; r < repeats; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            pla::determinant(A);
            auto t1 = std::chrono::high_resolution_clock::now();
            total_pla += std::chrono::duration<double>(t1 - t0).count();
        }

        double total_eigen = 0.0;
        for (int r = 0; r < repeats; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            Ae.determinant();
            auto t1 = std::chrono::high_resolution_clock::now();
            total_eigen += std::chrono::duration<double>(t1 - t0).count();
        }

        double avg_pla   = total_pla   / repeats;
        double avg_eigen = total_eigen / repeats;

        std::cout << "n=" << n
                  << ": pla avg=" << avg_pla << "s"
                  << ", eigen avg=" << avg_eigen << "s"
                  << ", ratio=" << (avg_pla / avg_eigen)
                  << std::endl;
    }

    SUCCEED();
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}