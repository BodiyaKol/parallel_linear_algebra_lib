#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <random>
#include <chrono>
#include <iostream>
#include <vector>

#include "pla.h"

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



TEST(Inverse, IdentityInverseIsIdentity) {
    for (int n : {1, 2, 4, 8}) {
        pla::Matrix<> A = pla::Matrix<>::identity(n);
        pla::Matrix<> Ainv = pla::inverse(A);

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double expected = (i == j) ? 1.0 : 0.0;
                EXPECT_NEAR(Ainv(i, j), expected, 1e-12)
                    << "n=" << n << " at (" << i << "," << j << ")";
            }
    }
}

TEST(Inverse, KnownSmall2x2) {
    // inv | 2 1 | = | 1.5  -0.5 |
    //     | 1 1 |   | -1    1   |
    pla::Matrix<> A(2, 2);
    A(0,0) = 2; A(0,1) = 1;
    A(1,0) = 1; A(1,1) = 1;

    pla::Matrix<> Ainv = pla::inverse(A);

    EXPECT_NEAR(Ainv(0,0),  1.0, 1e-12);
    EXPECT_NEAR(Ainv(0,1), -1.0, 1e-12);
    EXPECT_NEAR(Ainv(1,0), -1.0, 1e-12);
    EXPECT_NEAR(Ainv(1,1),  2.0, 1e-12);
}

TEST(Inverse, ATimesInverseIsIdentity) {
    // A * A^-1 == I
    std::mt19937 rng(1111);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    for (int n : {4, 8, 16, 32, 64}) {
        pla::Matrix<> A(n, n);
        fill_random_pla(A, rng, dist);

        pla::Matrix<> Ainv = pla::inverse(A);
        pla::Matrix<> AAinv = A * Ainv;

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double expected = (i == j) ? 1.0 : 0.0;
                EXPECT_NEAR(AAinv(i, j), expected, 1e-8)
                    << "A*Ainv at (" << i << "," << j << ") n=" << n;
            }
    }
}

TEST(Inverse, InverseTimesAIsIdentity) {
    // A^-1 * A == I
    std::mt19937 rng(2222);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    for (int n : {4, 8, 16, 32}) {
        pla::Matrix<> A(n, n);
        fill_random_pla(A, rng, dist);

        pla::Matrix<> Ainv = pla::inverse(A);
        pla::Matrix<> AinvA = Ainv * A;

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double expected = (i == j) ? 1.0 : 0.0;
                EXPECT_NEAR(AinvA(i, j), expected, 1e-8)
                    << "Ainv*A at (" << i << "," << j << ") n=" << n;
            }
    }
}

TEST(Inverse, MatchesEigen) {
    std::mt19937 rng(3333);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    for (int n : {4, 8, 16, 32, 64}) {
        pla::Matrix<> A(n, n);
        fill_random_pla(A, rng, dist);

        Eigen::MatrixXd Ae(n, n);
        sync_to_eigen(A, Ae);

        pla::Matrix<> Ainv      = pla::inverse(A);
        Eigen::MatrixXd Ainv_e  = Ae.inverse();

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                EXPECT_NEAR(Ainv(i, j), Ainv_e(i, j), 1e-8)
                    << "at (" << i << "," << j << ") n=" << n;
    }
}

TEST(Inverse, ThrowsOnNonSquare) {
    pla::Matrix<> A(3, 4);
    EXPECT_THROW(pla::inverse(A), std::invalid_argument);
}

TEST(Inverse, PerformanceCompareEigen) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<int> sizes = {32, 64, 128, 256, 512};
    const int repeats = 4;

    for (int n : sizes) {
        pla::Matrix<> A(n, n);
        fill_random_pla(A, rng, dist);

        Eigen::MatrixXd Ae(n, n);
        sync_to_eigen(A, Ae);

        // warmup
        { pla::Matrix<> r = pla::inverse(A); (void)r(0,0); }
        { volatile double s = Ae.inverse().sum(); (void)s; }

        double total_pla = 0.0;
        for (int r = 0; r < repeats; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            pla::Matrix<> res = pla::inverse(A);
            auto t1 = std::chrono::high_resolution_clock::now();
            volatile double sink = res(0, 0);
            (void)sink;
            total_pla += std::chrono::duration<double>(t1 - t0).count();
        }

        double total_eigen = 0.0;
        for (int r = 0; r < repeats; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            Eigen::MatrixXd res = Ae.inverse();
            auto t1 = std::chrono::high_resolution_clock::now();
            volatile double sink = res(0, 0);
            (void)sink;
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

TEST(Inverse, OpenMPCheck) {
#ifdef _OPENMP
    std::cout << "OpenMP enabled, max threads: " << omp_get_max_threads() << std::endl;
#else
    std::cout << "OpenMP NOT enabled" << std::endl;
#endif
    SUCCEED();
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}