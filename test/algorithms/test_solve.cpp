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

static void sync_vec_to_eigen(const pla::Vector<>& src, Eigen::VectorXd& dst) {
    for (int i = 0; i < src.dimension(); ++i)
        dst(i) = src[i];
}



TEST(Solve, KnownSmall2x2) {
    // | 2 1 | | x0 |   | 5 |     x0=2, x1=1
    // | 1 3 | | x1 | = | 5 |
    pla::Matrix<> A(2, 2);
    A(0,0) = 2; A(0,1) = 1;
    A(1,0) = 1; A(1,1) = 3;

    pla::Vector<> b(2);
    b[0] = 5; b[1] = 5;

    pla::Vector<> x = pla::solve(A, b);

    EXPECT_NEAR(x[0], 2.0, 1e-12);
    EXPECT_NEAR(x[1], 1.0, 1e-12);
}

TEST(Solve, KnownSmall3x3) {
    // | 1 0 0 | | x0 |   | 1 |     x = (1,2,3)
    // | 0 2 0 | | x1 | = | 4 |
    // | 0 0 3 | | x2 |   | 9 |
    pla::Matrix<> A(3, 3, 0.0);
    A(0,0) = 1; A(1,1) = 2; A(2,2) = 3;

    pla::Vector<> b(3);
    b[0] = 1; b[1] = 4; b[2] = 9;

    pla::Vector<> x = pla::solve(A, b);

    EXPECT_NEAR(x[0], 1.0, 1e-12);
    EXPECT_NEAR(x[1], 2.0, 1e-12);
    EXPECT_NEAR(x[2], 3.0, 1e-12);
}

TEST(Solve, ResidualIsSmall) {
    // ||Ax - b|| < tol
    std::mt19937 rng(1111);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    for (int n : {4, 8, 16, 32, 64}) {
        pla::Matrix<> A(n, n);
        fill_random_pla(A, rng, dist);

        pla::Vector<> b(n);
        for (int i = 0; i < n; ++i)
            b[i] = dist(rng);

        pla::Vector<> x = pla::solve(A, b);

        // r = Ax - b
        pla::Vector<> r = A * x;
        double residual = 0.0;
        for (int i = 0; i < n; ++i)
            residual += (r[i] - b[i]) * (r[i] - b[i]);
        residual = std::sqrt(residual);

        EXPECT_LT(residual, 1e-8) << "n=" << n;
    }
}

TEST(Solve, MatchesEigen) {
    std::mt19937 rng(2222);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    for (int n : {4, 8, 16, 32, 64, 128}) {
        pla::Matrix<> A(n, n);
        fill_random_pla(A, rng, dist);

        pla::Vector<> b(n);
        for (int i = 0; i < n; ++i)
            b[i] = dist(rng);

        Eigen::MatrixXd Ae(n, n);
        Eigen::VectorXd be(n);
        sync_to_eigen(A, Ae);
        sync_vec_to_eigen(b, be);

        pla::Vector<> x_pla    = pla::solve(A, b);
        Eigen::VectorXd x_eigen = Ae.lu().solve(be);

        for (int i = 0; i < n; ++i)
            EXPECT_NEAR(x_pla[i], x_eigen(i), 1e-8)
                << "mismatch at i=" << i << " n=" << n;
    }
}

TEST(Solve, ReusingLU) {
    std::mt19937 rng(3333);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    const int n = 32;
    pla::Matrix<> A(n, n);
    fill_random_pla(A, rng, dist);

    pla::Vector<> b(n);
    for (int i = 0; i < n; ++i)
        b[i] = dist(rng);

    pla::Vector<> x1 = pla::solve(A, b);

    auto lu = pla::lu_naive(A);
    pla::Vector<> x2 = pla::solve(lu, b);

    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(x1[i], x2[i], 1e-12) << "i=" << i;
}

TEST(Solve, MultipleRHS) {
    // AX = B
    std::mt19937 rng(4444);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    const int n = 32, nrhs = 8;
    pla::Matrix<> A(n, n);
    pla::Matrix<> B(n, nrhs);
    fill_random_pla(A, rng, dist);
    fill_random_pla(B, rng, dist);

    pla::Matrix<> X = pla::solve(A, B);

    for (int col = 0; col < nrhs; ++col) {
        pla::Vector<> b_col(n);
        for (int i = 0; i < n; ++i) b_col[i] = B(i, col);

        pla::Vector<> x_col(n);
        for (int i = 0; i < n; ++i) x_col[i] = X(i, col);

        pla::Vector<> r = A * x_col;
        double residual = 0.0;
        for (int i = 0; i < n; ++i)
            residual += (r[i] - b_col[i]) * (r[i] - b_col[i]);

        EXPECT_LT(std::sqrt(residual), 1e-8) << "col=" << col;
    }
}

TEST(Solve, ThrowsOnNonSquare) {
    pla::Matrix<> A(3, 4);
    pla::Vector<> b(3);
    EXPECT_THROW(pla::solve(A, b), std::invalid_argument);
}

TEST(Solve, ThrowsOnSizeMismatch) {
    pla::Matrix<> A(4, 4);
    pla::Vector<> b(3);
    EXPECT_THROW(pla::solve(A, b), std::invalid_argument);
}


TEST(Solve, PerformanceCompareEigen) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<int> sizes = {32, 64, 128, 256, 512, 1024};
    const int repeats = 4;

    for (int n : sizes) {
        pla::Matrix<> A(n, n);
        fill_random_pla(A, rng, dist);

        pla::Vector<> b(n);
        for (int i = 0; i < n; ++i) b[i] = dist(rng);

        Eigen::MatrixXd Ae(n, n);
        Eigen::VectorXd be(n);
        sync_to_eigen(A, Ae);
        sync_vec_to_eigen(b, be);

        pla::solve(A, b);
        Ae.lu().solve(be);

        double total_pla = 0.0;
        for (int r = 0; r < repeats; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            pla::solve(A, b);
            auto t1 = std::chrono::high_resolution_clock::now();
            total_pla += std::chrono::duration<double>(t1 - t0).count();
        }

        double total_eigen = 0.0;
        for (int r = 0; r < repeats; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            Ae.lu().solve(be);
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